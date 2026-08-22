# negsc-nids

Reimplementation of **NEGSC**: a Node-Edge Graph Attention encoder (**NEGAT**)
combined with graph-contrastive self-supervised learning (Wasserstein
distance + Gromov-Wasserstein distance losses) for **network intrusion
detection** on NetFlow data (NF-BoT-IoT-v2).

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

- [x] `data.py` — data loading, preprocessing, encoding, DGL graph construction
- [x] `main.py` — wired up through the data phase only (rest marked `TODO`)
- [ ] `negat.py`
- [ ] `negsc.py`
- [ ] `predict.py`
- [ ] `run_inference.py`

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

## Running (current progress)

Only the data pipeline is wired in so far. Running `main.py` at this point
loads the raw csv, preprocesses it, and builds the training graph chunks,
printing a summary of each graph:

```bash
python main.py
```

Expected output looks like:

```
[data] built <N> training graph chunks on cpu
  graph[0]: nodes=... edges=... edge_feat_dim=...
  graph[1]: nodes=... edges=... edge_feat_dim=...
  ...
```

The remaining pipeline (NEGAT encoder, NEGSC contrastive training, and the
downstream Predict stage) is stubbed out with `TODO` comments in `main.py`
and will be filled in as `negat.py`, `negsc.py`, and `predict.py` are added.

## Reference

Architecture diagram and original notebooks provided as the basis for this
reimplementation; see `NEGSC.ipynb` / `test.ipynb` for the source logic each
module was ported from.