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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Embedding (NEGSC.ipynb cell 44, embedding part)
# ---------------------------------------------------------------------------

def embed_graphs(model, full_train_graph, G_test):
    """
    Embed the full training graph and the test graph with the trained
    NEGSC model's encoder. Mirrors NEGSC.ipynb cell 44 (embedding lines).

    Returns (train_embs, test_embs, train_lbls, test_lbls).
    """
    embeds = model.embed(full_train_graph, full_train_graph.ndata['h'],
                          full_train_graph.edata['h']).detach()
    train_embs = embeds
    test_embs = model.embed(G_test, G_test.ndata['feature'], G_test.edata['h'])

    train_lbls = full_train_graph.edata['label']
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
    plt.show()


def evaluate(test_lbls, preds, save_path='outputs/confusion_matrix.png'):
    """
    Convenience wrapper: print classification_report and plot+save the
    confusion matrix, mirroring NEGSC.ipynb cells 50/52.
    """
    target_names = np.unique(test_lbls)
    plot_confusion_matrix(
        cm=confusion_matrix(test_lbls, preds),
        normalize=True,
        target_names=target_names,
        title="Confusion Matrix",
        save_path=save_path,
    )
    print_classification_report(test_lbls, preds, target_names)