# Results

Every number here is traceable to a `run_id` and the git commit that produced it.
Nothing in this file is estimated or recalled.

**Status:** one full training run of variant A, single seed. This is a
reproduction of the reference implementation at full scale, not a benchmark and
not a variance estimate.

---

## 1. Run `full_v3` — first complete reproduction

| | |
|---|---|
| Date | 2026-11-02T11:53:02 |
| Git commit | `98edffe` |
| Variant | A (reference-faithful) |
| Command | `PYTHONHASHSEED=0 python main.py --run-id full_v3 --log-every 1` |
| Device | NVIDIA A100 80 GB PCIe, **MIG 1g.10gb slice** — `dgl_cuda: true` |
| Python / torch / dgl | 3.10.12 / 2.3.0+cu121 / 2.2.0+cu121 |
| Wall clock | **5,652 s (94 min)** |
| Artifacts | `outputs/run_full_v3.json`, `loss_full_v3.csv`, `confusion_matrix_full_v3.png`, `model_full_v3.pt`, `log_full_v3.pt` |

### Data and training scale

| | |
|---|---|
| Input | `data_raw/sample_3pct.csv`, 1,131,139 rows (sha256 `e2157235acae8f34…`) |
| Split | 70/30 stratified → **791,797 train / 339,342 test flows** |
| Graph chunks | **23** (22 × 35,000 rows + 1 × 21,797) |
| Nodes per chunk | 66,086 – 66,235 (last: 41,876) |
| Epochs | 300 per chunk → **6,900 training steps** |
| Subgraphs per step | mean **44.1**, min 10, max 74 |
| Subgraph pairs seen in total | **304,274** |

### Metrics

Evaluated on the test graph: 339,342 flows → **678,684 directed edges**.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.8908 | 0.6046 | 0.7203 | 2,428 |
| DDoS | 0.9604 | 0.9653 | 0.9628 | 328,938 |
| DoS | 0.9502 | 0.9390 | 0.9446 | 300,098 |
| Reconnaissance | 0.8652 | 0.9137 | 0.8888 | 47,172 |
| Theft | 0.0909 | 0.0833 | 0.0870 | 48 |
| **Weighted avg** | **0.9490** | **0.9487** | **0.9487** | 678,684 |
| **Macro avg** | **0.7515** | **0.7012** | **0.7207** | 678,684 |
| **Accuracy** | | | **0.9487** | |

---

## 2. Comparison with the paper

Paper Table 12, NF-BoT-IoT-v2, multiclass. Absolute difference is
ours − paper, in percentage points.

| Class | Paper Recall | Ours | Δ | Paper F1 | Ours | Δ |
|---|---|---|---|---|---|---|
| Benign | 61.40% | 60.46% | **−0.94** | 71.83% | 72.03% | **+0.20** |
| DDoS | 97.46% | 96.53% | **−0.93** | 96.58% | 96.28% | **−0.30** |
| DoS | 93.56% | 93.90% | **+0.34** | 94.71% | 94.46% | **−0.25** |
| Reconnaissance | 91.11% | 91.37% | **+0.26** | 89.26% | 88.88% | **−0.38** |
| Theft | 31.82% | 8.33% | **−23.49** | 24.35% | 8.70% | **−15.65** |
| **Weighted** | **95.16%** | **94.87%** | **−0.29** | **95.15%** | **94.87%** | **−0.28** |

**Four of five classes agree to within one point.** Weighted recall is 0.29
points below the reported figure, weighted F1 0.28 points below. No tuning was
applied to reach this — it is the default configuration (τ=0.5, lr=1e-3, wd=0,
k1=3, 300 epochs, seed 42) run once at full scale.

**Theft is the one real discrepancy:** 4 of 48 test edges correct, against an
expected ~15. With 48 samples this is a handful of flows, but at a true rate of
31.8% the probability of seeing 4 or fewer is small (~0.1%), so it is unlikely
to be chance alone. Candidates: seed variance on a class with 48 test examples,
or the embedding of very rare classes. Multi-seed runs will separate them.

### Correction to an earlier claim

An analysis performed before this run — based on the authors' shipped
checkpoints scoring 0.44–0.90 — concluded that the reported numbers were not
reachable. **That conclusion was wrong**, and the reason it was wrong is worth
recording: a saved checkpoint is bound to the feature-column order used when it
was trained, and that order is not recorded anywhere. The checkpoint result
proves *checkpoints are not portable*, not *the method does not reproduce*.
Training from scratch, the order is whatever your process produces and the
weights are learned against it.

What survives from that analysis: the released code does diverge from the paper
(V1–V13), the shipped checkpoints genuinely do not work, and the topology is
synthetic. What does not survive: "the reported numbers are unreproducible."

---

## 3. Finding: the contrastive loss does not converge

This is the most substantive observation from the run, and it is not visible in
the metrics.

| | |
|---|---|
| Loss over all 6,900 steps | mean **0.7842**, sd 0.0144, min 0.7126, max 0.8047 |
| Chunk 0, linear slope over 300 epochs | **−0.000012 / epoch** (−0.0037 over the whole chunk) |
| Chunk 0, 25-epoch rolling mean | 0.7824 → **0.7737**, minimum 0.7574 at epoch 227 |
| Chunk 22, slope | **+0.000058 / epoch** (slightly upward) |

The loss moves by less than the epoch-to-epoch noise over an entire chunk. It is
flat.

**Yet the downstream classifier reaches 94.87%.** Both facts are true, and
together they say something specific: either the encoder learns everything it is
going to learn within the first few epochs and the remaining 290 add nothing, or
the objective is close to saturated and the embeddings are not being refined by
training at all.

Two documented deviations predict exactly this:

- **V2** — the edge-embedding `MLPPredictor` is re-instantiated with fresh random
  weights on every step and never optimised. Half the reported loss (`0.5 ×
  edge_wd_loss`) is therefore computed on noise, which cannot decrease.
- **V13** — the sampler yields ~44 subgraphs per epoch, not 20,000. Over 300
  epochs a chunk sees ~13,000 subgraph comparisons in total; across all 23
  chunks, 304,274. That is a very small number of contrastive examples.

This is a Phase 3 / Phase 4 question, not a bug to fix in variant A. It also
raises a methodological point: if the downstream probe is doing most of the
work, the "self-supervised" claim needs qualifying, because the classifier is
trained on labelled edges.

---

## 4. Cost

| | |
|---|---|
| Total wall clock | 5,652 s (94 min) |
| Per chunk (300 epochs) | ≈ 245 s |
| Per training step | ≈ 0.82 s (includes ~44 subgraphs forward+backward) |
| Predict stage | < 2 min; peak GPU use ~0.3 GB of the 10.2 GB slice |
| Model size | encoder 343 KB, classifier 3 KB |

For reference, the paper reports 3,685 s on a full RTX 4090 (Table 14). Our
94 min on a one-seventh MIG slice is the same order once the slice is accounted
for.

---

## 5. Run history

Recorded because two of the three failures were harness bugs, not model
problems, and the fixes are part of the codebase now.

| Run | Outcome | Cause | Fix |
|---|---|---|---|
| `full_lab` | ✗ wrong scale | `--sample-frac` defaulted to 0.03, so the already-sampled cache was sampled again: 33,934 rows, **one** chunk | default is `1.0`; warns if applied to a cache (`e0b5d0b`) |
| `full_lab_v2` | ✗ crashed at predict | 23 chunks trained, then the predict stage ran out of GPU memory under autograd — ~2.1 GB of retained activations across three heads on a 680k-edge test graph, plus ~660 MB of graph tensors. The error surfaced as `NVML_SUCCESS == r INTERNAL ASSERT FAILED` because the allocator could not query device limits on the MIG slice. Training was **lost** — the checkpoint was written only after predict. | `embed_graphs` runs under `no_grad` and frees the train graph's features; checkpoint saved after every chunk; loss CSV written before predict |
| `full_v3` | ✓ complete | — | — |

---

## 6. What this run does not establish

Stated explicitly so nothing here is over-read:

1. **No variance estimate.** One seed. The 94.87% has no error bar, and given
   that Theft moved from 0% to 8.3% across runs, the spread on rare classes is
   not small. Three more seeds are the next step.
2. **Parity is not proven.** This run shows our pipeline produces a result close
   to the paper's. It does not show that our implementation computes the same
   thing as the released notebook, step for step. That is Phase 2, and it has an
   open issue: `sub_sam` samples from a sorted list while the reference samples
   from a set, so with the same RNG state the two choose different neighbours.
   Exact parity needs a `--legacy-sampling` switch used only in the Python 3.10
   reference environment.
3. **No firewall ablation.** This run used the reference's target encoder, which
   is fitted on `X_train, y_train`, so label information reaches the features.
   The 94.87% is therefore a *leaky* number by the project's own definition
   (ROADMAP Phase 6). The honest figure is not yet measured.
4. **Not the paper's protocol.** We do not know why Table 5 and Table 14
   disagree on test-set size. The agreement above is between our run and Table
   12, under our own reproducible protocol.
5. **Synthetic topology.** Every destination IP is randomised before the graph is
   built, so the model has never seen real network structure. This is inherited
   from the reference and applies to any comparison with the paper.

---

## 7. Next

| # | Step | Why it is next |
|---|---|---|
| 1 | 3 more seeds (`--seed 1 2 3`, same config) | turns 94.87 into a mean ± std |
| 2 | Phase 2 parity gate | separates "similar number" from "same computation" |
| 3 | Firewall ablation (L0 vs L1) | the 94.87 is currently a leaky number |
| 4 | Phase 3 deviation audit | the flat loss (§3) is the first item to explain |

---

## Appendix — reproducing this exact run

```bash
cd ~/negsc-fraud-detection
source ~/.venvs/negsc/bin/activate
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --run-id full_v3 --log-every 1
```

Requires `data_raw/sample_3pct.csv` with sha256
`e2157235acae8f34f35c7f81391dd143bf1fddeacb16c3c641eec2b9950ff0a9`, commit
`98edffe`, and the environment recorded in `ENV.md`.

Verify a rerun by comparing `outputs/loss_full_v3.csv` row for row.
