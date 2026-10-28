# Environment record

Verified **2026-10-28** on the development machine. Update this file whenever a
dependency, driver or interpreter changes — a result is only reproducible
against a recorded environment.

## What is installed and verified

| Item | Value | Status |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200 | — |
| Python (delivery env) | 3.12.3 | ☑ verified |
| torch | 2.3.0+**cpu** | ⚠ CPU build only |
| dgl | 2.2.1 | ☑ verified |
| numpy | 1.26.4 | ☑ pinned by torch 2.3 |
| pandas | 2.3.3 | ☑ |
| scikit-learn | 1.7.2 | ☑ |
| category_encoders | 2.11.2 | ☑ installed during Phase 1 (was missing — blocker B2) |
| networkx / scipy | 3.7 / 1.16.2 | ☑ |
| pyarrow | **not installed** | see note below |
| GPU | NVIDIA RTX 3050 Laptop, 6 GB | present; driver 560.94, CUDA 12.6 |
| RAM | 16.9 GB total | ⚠ the 6.05 GB CSV does not fit |
| Disk | C: 12.5 GB free, D: 57.7 GB free | artifacts on D: |

Exact versions: `requirements.lock.txt` (182 packages).

## Environments

### `negsc-ref` — Python 3.10, CUDA  (reference env, created 2026-10-28)

| Item | Value |
|---|---|
| Python | 3.10.22 |
| torch | **2.3.0+cu121** — CUDA available: **True** |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 6 GB |
| dgl | 2.2.1 |
| numpy / sklearn / pandas | 1.26.4 / 1.7.2 / 2.3.3 |
| category_encoders | 2.8.1 |

Verified: `torch.ones(2).cuda().device` → `cuda:0`.

This is the environment the parity gate runs in. Python 3.10 is not cosmetic:
on ≥3.11 `random.sample(set, k)` raises, and the released notebook passes a
set. Pinning 3.10 lets the authors' notebook run essentially as published.

**Three pins that are not obvious** (learned the hard way; they are in
`create_env.sh`):

| Pin | Why |
|---|---|
| `torchdata==0.7.1` | DGL 2.2.1 imports `torchdata.datapipes`, removed in torchdata ≥ 0.10 |
| `pyyaml` | DGL's `graphbolt` imports it at load time |
| `pydantic` | same — `graphbolt.impl.ondisk_metadata` |

Without any one of them, `import dgl` fails with a `ModuleNotFoundError`.

Recreate with `bash create_env.sh`.

### ⚠ DGL has no Windows CUDA wheels

Verified 2026-10-28 by listing the wheel index:

| Index | Wheels | Windows CUDA wheels |
|---|---|---|
| `data.dgl.ai/wheels/torch-2.3/cu121/repo.html` | 25 | **0** — all `manylinux1_x86_64` |
| `data.dgl.ai/wheels/repo.html` | 1302 | 355, all CPU |

So on Windows a machine can have `torch 2.3.0+cu121` reporting `cuda True` and
still fail the moment a graph moves to the GPU:

```
DGLError: Device API cuda is not enabled. Please install the cuda version of dgl.
```

`device.py` now probes DGL before choosing a device and falls back to CPU with
a warning rather than crashing after data preparation. **GPU training of a DGL
model requires WSL2 or native Linux**; the lab GPU is not usable from native
Windows for this project.

Practical consequence: the parity gate and even the full 32-chunk run are CPU
work here. At ~50 subgraphs per epoch a step is 1–3 s, so a full training run is
hours, not days — the Windows limitation costs less than it sounds.

### `negsc-dev` — Python 3.12, CPU  (delivery env)

The environment used for all Phase 1 work. Same packages, `torch 2.3.0+cpu`.

## Why pyarrow is absent

`pyarrow` 26 requires NumPy ≥ 2.0. This environment is pinned to 1.26 by
`torch 2.3.0`, and installing pyarrow leaves `sklearn` unable to import
(`sklearn.utils.fixes` imports pyarrow when present). It was installed and
removed during Phase 1. Consequence: **the dataset cache is CSV, not parquet**
(`make_cache.py`). No functional difference for this project.

## Verified capabilities

| Capability | Status | Evidence |
|---|---|---|
| End-to-end pipeline on synthetic data | ☑ | `pytest tests/ -q` → **12 passed in `negsc-ref` (py3.10) and 12 passed in `negsc-dev` (py3.12)** |
| End-to-end pipeline on the real cache | ☑ | `outputs/run_real_step.json` |
| Dense and sparse adjacency numerically identical | ☑ | `tests/test_sparse_adj.py` (5 tests) |
| Same seed → same losses, in and across processes | ☑ | `tests/test_determinism.py` |
| DGL CUDA probe falls back cleanly | ☑ | `device.py`; the py3.10 env has CUDA torch + CPU DGL and the suite passes |
| Peak RAM during a 200k-row run | ~2.5 GB | measured, block-parallel |
| One contrastive step, real chunk, ~50 subgraphs | ~1–3 s | `run_real_step.json` |

## Runtime note

The reference's `--num 20000` does **not** produce 20,000 subgraphs. On a real
chunk only ~0.2% of drawn nodes have out-degree ≥ `k1`, so the sampler returns
**~40–60 subgraphs per epoch**. This makes a full training run hours, not days,
on CPU. It is recorded as deviation V13 in `ROADMAP.md` §8.1.