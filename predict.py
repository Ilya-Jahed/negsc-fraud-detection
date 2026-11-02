"""
predict.py -- "Predict" panel.

Ports the downstream classification stage from NEGSC.ipynb cells 33-52:
embed train/test graphs with the trained NEGSC Model, train a fresh
MLPPredictor as the final classifier on top of those embeddings, predict,
decode labels, and report metrics + confusion matrix.

.cuda() calls become .to(DEVICE), consistent with the rest of the project.
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

import negsc
from negsc import MLPPredictor

from device import get_device

DEVICE = get_device()


# ---------------------------------------------------------------------------
# Embedding (NEGSC.ipynb cell 44, embedding part)
# ---------------------------------------------------------------------------

def embed_graphs(model, full_train_graph, G_test):
    """
    Embed the full training graph and the test graph with the trained
    NEGSC model's encoder. Mirrors NEGSC.ipynb cell 44 (embedding lines).

    Returns (train_embs, test_embs, train_lbls, test_lbls).

    MEMORY NOTE -- divergence from the notebook, and why
    ----------------------------------------------------
    The notebook runs this under autograd. On the paper's 24 GB card that is
    fine; here the test graph alone is ~340k flows (~680k directed edges), and
    the multi-head layer keeps all three heads' intermediates alive at once.
    Measured on that graph, the retained activations are roughly:

        feat_cat [E,117]   ~300 MB
        W out    [E,78]    ~200 MB
        messages [E,39]    ~100 MB
        attention[E,39]    ~100 MB
                                   ~0.7 GB per head, x3 heads

    plus ~660 MB of graph tensors for both graphs. The first full run died
    exactly here, on a 9.5 GB MIG slice, with an allocator error.

    None of it is needed: the embeddings are consumed by a downstream
    classifier that trains its own weights, so no gradient has to flow back
    through the encoder. Running under no_grad removes the activations
    entirely. `model.eval()` changes nothing numerically -- this encoder has no
    dropout or batch-norm -- it is set to make the intent explicit.

    The training graph's feature tensors are also released before the test
    forward: the classifier still needs that graph for its edges and labels,
    but not for `ndata['h']` or `edata['h']`.
    """
    model.eval()
    with torch.no_grad():
        train_embs = model.embed(full_train_graph, full_train_graph.ndata['h'],
                                 full_train_graph.edata['h']).detach()
        train_lbls = full_train_graph.edata['label']

        # Free ~460 MB before the test-graph forward pass. Safe: the
        # downstream classifier overwrites ndata['h'] with its own input and
        # only reads edata['label'], which is kept.
        full_train_graph.ndata.pop('h', None)
        full_train_graph.edata.pop('h', None)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        test_embs = model.embed(G_test, G_test.ndata['feature'],
                                G_test.edata['h']).detach()
        test_lbls = G_test.edata['label']

    return train_embs, test_embs, train_lbls, test_lbls


# ---------------------------------------------------------------------------
# Downstream classifier training (NEGSC.ipynb cell 44, classifier part)
# ---------------------------------------------------------------------------

def train_classifier(train_embs, train_lbls, in_features: int, n_classes: int,
                      g_for_predictor, num_steps: int = 10000):
    """
    Train a fresh MLPPredictor as the final edge classifier on top of the
    frozen NEGSC embeddings. Mirrors NEGSC.ipynb cell 44 (the `log`/`opt`
    training loop). Note: this MLPPredictor is a *separate* instance from
    the one used internally in negsc.Model._edge_features -- it gets
    actually trained here, unlike that one.

    `g_for_predictor` is the graph passed to `log(g, train_embs)` each
    step (the source reuses the training graph `g` here).
    """
    log = MLPPredictor(in_features, n_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.0)
    log.to(DEVICE)

    xent = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        log.train()
        opt.zero_grad()

        logits = log(g_for_predictor, train_embs)
        loss = xent(logits, train_lbls)
        loss.backward(retain_graph=True)
        opt.step()

    return log


# ---------------------------------------------------------------------------
# Prediction (NEGSC.ipynb cell 44, tail)
# ---------------------------------------------------------------------------

def predict(log, G_test, test_embs):
    """Run the trained classifier on the test graph and argmax to labels."""
    logits = log(G_test, test_embs)
    preds = torch.argmax(logits, dim=1)
    return preds


def decode_labels(label_encoder, test_lbls, preds):
    """
    Move predictions/labels to CPU and inverse-transform back to string
    class names. Mirrors NEGSC.ipynb cells 45-47.
    """
    preds = preds.to('cpu')
    test_lbls = test_lbls.to('cpu')
    test_lbls = label_encoder.inverse_transform(test_lbls)
    preds = label_encoder.inverse_transform(preds)
    return test_lbls, preds


# ---------------------------------------------------------------------------
# Evaluation (NEGSC.ipynb cells 49-52)
# ---------------------------------------------------------------------------

def print_classification_report(test_lbls, preds, target_names):
    """Mirrors NEGSC.ipynb cell 52."""
    print(classification_report(test_lbls, preds, target_names=target_names, digits=4))


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                           cmap=None, normalize=True, save_path=None):
    """
    Ported from NEGSC.ipynb cell 49 (first plot_confusion_matrix
    definition) unchanged, with an added optional save_path so the caller
    controls where the figure is written instead of a hardcoded filename.
    """
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate(test_lbls, preds, save_path='outputs/confusion_matrix.png'):
    """
    Convenience wrapper: print classification_report and plot+save the
    confusion matrix, mirroring NEGSC.ipynb cells 50/52.

    Returns a metrics dict. `macro_f1` is included alongside the weighted
    figures because the weighted average is dominated by DDoS/DoS and hides
    exactly the minority-class collapse this dataset is prone to (Theft has
    73 samples in the paper's 3% sample). See the benchmark protocol,
    ROADMAP Phase 7.
    """
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)

    target_names = np.unique(test_lbls)
    plot_confusion_matrix(
        cm=confusion_matrix(test_lbls, preds),
        normalize=True,
        target_names=target_names,
        title="Confusion Matrix",
        save_path=save_path,
    )
    print_classification_report(test_lbls, preds, target_names)

    metrics = dict(
        accuracy=float(accuracy_score(test_lbls, preds)),
        weighted_precision=float(precision_score(test_lbls, preds, average="weighted", zero_division=0)),
        weighted_recall=float(recall_score(test_lbls, preds, average="weighted", zero_division=0)),
        weighted_f1=float(f1_score(test_lbls, preds, average="weighted", zero_division=0)),
        macro_precision=float(precision_score(test_lbls, preds, average="macro", zero_division=0)),
        macro_recall=float(recall_score(test_lbls, preds, average="macro", zero_division=0)),
        macro_f1=float(f1_score(test_lbls, preds, average="macro", zero_division=0)),
    )
    metrics["per_class"] = {
        str(cls): dict(
            recall=float(recall_score(test_lbls, preds, labels=[cls], average="macro", zero_division=0)),
            f1=float(f1_score(test_lbls, preds, labels=[cls], average="macro", zero_division=0)),
            support=int(np.sum(np.asarray(test_lbls) == cls)),
        )
        for cls in target_names
    }
    return metrics