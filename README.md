# negsc-nids

Reimplementation of **NEGSC**: a Node-Edge Graph Attention encoder (**NEGAT**)
combined with graph-contrastive self-supervised learning (Wasserstein
distance + Gromov-Wasserstein distance losses) for **network intrusion
detection** on NetFlow data (NF-BoT-IoT-v2).

> **Paper:** Xu, R., Wu, G., Wang, W., Gao, X., He, A., Zhang, Z. (2024).
> *Applying self-supervised learning to network intrusion detection for
> network flows with graph neural network.* Computer Networks 248, 110495.
> Authors' code: https://github.com/renj-xu/NEGSC

**What problem this solves:** GNN-based NIDS usually need labeled traffic
to train on, which is expensive and can't keep up with new attack types.
Prior self-supervised GNN-based NIDS (e.g. Anomal-E) only manage *binary*
classification ("malicious or not") because their contrastive objective
doesn't carry enough signal to separate attack *types*. NEGSC is presented
as the first GNN-based self-supervised method to do full **multiclass**
classification of network flows (Benign / DDoS / DoS / Reconnaissance /
Theft), via two changes to a general graph-contrastive framework (GSC):
an edge-aware encoder (**NEGAT**, since in NetFlow data the informative
signal — protocol, flags, byte/packet counts — lives on edges/flows, not
on nodes/IP:port endpoints), and a new contrastive objective (**NEGSC**)
comparing subgraphs via both edge-feature distance (Wasserstein Distance)
and topology distance (Gromov-Wasserstein Distance).

This repo mirrors the paper's original notebooks (`NEGSC.ipynb` for
training, `test.ipynb` for inference) one-to-one in logic, split into
readable modules that map directly onto the panels of the architecture
diagram:

| Module | Diagram panel |
|---|---|
| `data.py` | Data Flow / Graph Representation |
| `negat.py` | NEGAT (Encoder) |
| `negsc.py` | NEGSC (sampling, generative NEGAT, contrastive WD/GWD loss) |
| `predict.py` | Predict |
| `main.py` | full training pipeline (= `NEGSC.ipynb`) |
| `run_inference.py` | inference on a saved checkpoint (= `test.ipynb`) |

## Status

All modules implemented, end to end (both `NEGSC.ipynb` and `test.ipynb`
are fully ported):

- [x] `data.py` — data loading, preprocessing, encoding, DGL graph construction
- [x] `negat.py` — NEGAT encoder (GATlayer, MultiHeadGATLayer, GAT)
- [x] `negsc.py` — subgraph sampling, NEGSC contrastive model (WD/GWD loss), train()
- [x] `predict.py` — embedding, downstream classifier, metrics, confusion matrix
- [x] `main.py` — full end-to-end training pipeline (= `NEGSC.ipynb`)
- [x] `run_inference.py` — inference on a saved checkpoint (= `test.ipynb`)

## Setup

```bash
conda create -n negsc python=3.10 -y
conda activate negsc
pip install --upgrade pip

# DGL (CPU, Windows): version 1.1.2 is required for compatibility
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/repo.html

pip install -r requirements.txt
```

> **Windows note:** newer DGL versions can fail to load (`dgl.dll` not
> found) due to a compatibility issue. `dgl==1.1.2` is confirmed working.

See [Running on GPU](#running-on-gpu-later-not-needed-for-now) below if
you're setting up on a CUDA machine instead.

## Running on GPU later (not needed for now)

The code auto-detects the device
(`DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
in `main.py`), so no code changes are needed to move from CPU to GPU —
only the install steps differ:

```bash
# 1. install a CUDA-enabled torch build matching your driver's CUDA version
#    (check with `nvidia-smi`, then pick your combo at
#    https://pytorch.org/get-started/locally/), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. install the matching DGL wheel for that same torch + CUDA version
#    (see https://www.dgl.ai/pages/start.html), e.g.:
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# 3. verify both picked up the GPU:
python -c "import torch; print(torch.cuda.is_available())"   # should be True
```

If torch's and DGL's CUDA versions don't match, DGL tends to fail silently
back to CPU rather than error — always confirm with step 3.

## Data

Download `NF-BoT-IoT-v2.csv` and place it at:

```
data_raw/NF-BoT-IoT-v2.csv
```

## Running

### Training (= `NEGSC.ipynb`)

```bash
python main.py
```

This runs the full pipeline: preprocesses the data, builds training
graphs, pre-trains the NEGAT/NEGSC contrastive encoder, then trains a
downstream MLP classifier on the frozen embeddings, evaluates it
(classification report + confusion matrix saved to
`outputs/confusion_matrix.png`), and saves checkpoints to
`checkpoints/model.pth` and `checkpoints/log.pth`.

Expect a lot of console output — the contrastive loss debug logging and
the 10,000-step downstream classifier loop are both verbose, matching the
source notebook's behavior.

### Inference on a saved checkpoint (= `test.ipynb`)

```bash
python run_inference.py
```

Requires `checkpoints/model.pth` and `checkpoints/log.pth` to already
exist (produced by `python main.py`). This script re-runs preprocessing
(without downsampling — `test.ipynb` uses the raw csv as-is, unlike
`main.py`'s 3%-per-class sample) and re-derives the target encoder and
scaler fresh on the resulting `X_train`, then loads the checkpoints,
embeds the test graph, predicts, and reports metrics + a confusion matrix
saved to `outputs/confusion_matrix_inference.png`. See the "Known
discrepancies" section below for why the encoder/scaler are re-derived
rather than loaded from the training run.

## Known discrepancies vs. the paper

This project's goal so far has been to reproduce the reference notebooks
**exactly**, bugs included, before any refactor. Close reading turned up
several places where the *code* quietly diverges from what the *paper*
describes or implies. None of these have been "fixed" — each is
replicated faithfully and flagged in-code where it occurs:

| # | Location | What the paper implies | What the code actually does |
|---|---|---|---|
| 1 | `main.py`, model/optimizer construction | Train with `tau=5`, `lr=1e-4`, `weight_decay=1e-5` (defined as config constants) | `negsc.Model(Encoder, gene)` and `torch.optim.Adam(model.parameters())` are called **without** passing these in, so they silently fall back to library defaults (`tau≈0.5`, `lr=1e-3`, `weight_decay=0`). The configured values are dead. |
| 2 | `predict.py`, `embed_graphs` | Node features read from a consistent key | Training graphs store node features under `ndata['h']`; the test graph uses `ndata['feature']` instead — a naming inconsistency inherited from the source notebook, handled with a special case rather than a fix. |
| 3 | `predict.py`, `train_classifier` | — | `loss.backward(retain_graph=True)` is called even though nothing downstream needs the graph kept alive (inputs are already `.detach()`-ed). Likely an unnecessary carry-over that costs extra memory for no benefit. |
| 4 | `negat.py`, `GAT.__init__` | Hidden/output width should follow the `out_dim` argument | `MultiHeadGATLayer(in_dim, e_dim, 39, num_heads)` hardcodes the output width to `39` regardless of `out_dim`. Harmless here only because this dataset always has 39 edge features anyway. |
| 5 | `negsc.py`, `Model.sub_loss_batch` | `L = L_edges + L_topology` (two terms) | A third term (node-level WD loss) is computed and logged but never added into the returned loss — this actually **matches** the paper's two-term formula; the extra computation looks like dead code inherited from the base GSC method, costing time for nothing. |
| 6 | `negsc.py`, `Model._edge_features` | The edge embedder should be a trained part of the model | The edge-embedding `MLPPredictor` is instantiated fresh with random weights on **every training step**, never registered with the optimizer, and discarded after use. One of the two loss terms that drives training is therefore computed on a permanently-untrained random projection — confirmed intentional in the source. |

## Reference

Architecture diagram and original notebooks provided as the basis for this
reimplementation; see `NEGSC.ipynb` / `test.ipynb` for the source logic each
module was ported from.