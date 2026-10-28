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

## What is NOT done

- ☐ **CUDA build of torch.** The machine has a working GPU but this
  interpreter has `torch 2.3.0+cpu`, so every run so far is CPU. The lab GPU
  is available but has not been wired into an environment here.
- ☐ **Python 3.10 reference environment** (`negsc-ref`). Not created. Until it
  exists, the reference notebook runs on 3.12 with the `random.sample`
  workaround applied to both sides instead of neither.

### To create the reference environment

```bash
conda create -n negsc-ref python=3.10 -y
conda activate negsc-ref
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install pandas scikit-learn category_encoders networkx scipy matplotlib seaborn pytest
python -c "import torch, dgl; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), dgl.__version__)"
```

Python 3.10 is not cosmetic: on ≥3.11, `random.sample(set, k)` raises, and the
reference notebook passes a set. Pinning 3.10 lets the untouched notebook run,
which is what makes the parity gate a comparison rather than a comparison of
two patched implementations.

## Why pyarrow is absent

`pyarrow` 26 requires NumPy ≥ 2.0. This environment is pinned to 1.26 by
`torch 2.3.0`, and installing pyarrow leaves `sklearn` unable to import
(`sklearn.utils.fixes` imports pyarrow when present). It was installed and
removed during Phase 1. Consequence: **the dataset cache is CSV, not parquet**
(`make_cache.py`). No functional difference for this project.

## Verified capabilities

| Capability | Status | Evidence |
|---|---|---|
| End-to-end pipeline on synthetic data | ☑ | `pytest tests/ -q` → 12 passed |
| End-to-end pipeline on the real cache | ☑ | `outputs/run_real_step.json` |
| Dense and sparse adjacency numerically identical | ☑ | `tests/test_sparse_adj.py` (5 tests) |
| Same seed → same losses, in and across processes | ☑ | `tests/test_determinism.py` |
| Peak RAM during a 200k-row run | ~2.5 GB | measured, block-parallel |
| One contrastive step, real chunk, ~50 subgraphs | ~1–3 s | `run_real_step.json` |

## Runtime note

The reference's `--num 20000` does **not** produce 20,000 subgraphs. On a real
chunk only ~0.2% of drawn nodes have out-degree ≥ `k1`, so the sampler returns
**~40–60 subgraphs per epoch**. This makes a full training run hours, not days,
on CPU. It is recorded as deviation V13 in `ROADMAP.md` §8.1.