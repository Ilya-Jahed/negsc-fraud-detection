# NEGSC Baseline — Pre-Integration Roadmap

> **Purpose.** Everything that must be done *here* before anything is taken into the
> app. It is a work order, not a description of the code (for that see `README.md`).
>
> **Version 7 — 2026-11-02.** Variant A reproduced the paper: 94.87% weighted
> recall / 94.87% weighted F1 against the reported 95.16% / 95.15%, four of five
> classes within one point, on the first untuned full run (`RESULTS.md`). **This
> changes what the remaining phases are for.** They are no longer repairs needed
> to reach a number — the number arrived without them. They now answer a
> different question: *which parts of the released code produce that result, and
> which are accidental?* See §4.1.
>
> **Version 6 — 2026-11-02.** A full run of variant A completed on the lab
> server and landed within 0.29 points of the paper's weighted recall (94.87% vs
> 95.16%). See `RESULTS.md`. Two consequences: the earlier claim that the
> reported numbers were unreachable was **wrong** and is corrected there; and
> the next milestone is variance (3 seeds) before the parity gate, since a
> single run has no error bar.
>
> **Version 5 — 2026-10-28.** v4 reordered the project around paper fidelity.
> v5 records what Phase 0/1 actually produced, and **two measurements that change the
> plan**: the sampler yields ~50 subgraphs per epoch rather than 20,000 (V13), which
> makes training hours rather than days; and the dataset cache is CSV, not parquet,
> because pyarrow breaks this environment. See `ENV.md`.
>
> Contents: §0 how to read · §1 decisions · §2 state · §3 acceptance ·
> §4 **why this order** · §5–§14 phases · §15 milestones · §16 risks ·
> Appendices A–E.

---

## 0. How to read this document

| Marker | Meaning |
|---|---|
| ☐ / ☑ | not done / done, with the evidence recorded |
| 🔒 **GATE** | do not proceed until the decision is written into `DECISIONS.md` |
| ⚠ | will hurt later if ignored |
| 🧰 **(E)** | engineering change — no semantic effect, applies to A and B alike |
| 🔬 **(V)** | method change — what turns A into B |
| 📌 | a sentence you may use in the write-up once the box is ticked |

**How code comments refer to this document.** Source files cite the roadmap by
*phase name*, never by section number — `ROADMAP, Phase 1 -- sparse adjacency`,
not `§6.4`. Section numbers move whenever this document is reorganised (it has
been renumbered three times); phase names do not. Deviation IDs (`V1`–`V13`,
`B1`–`B5`) are stable item identifiers rather than positions, so they are cited
as-is.

**Phase names in use:** 0 Environment · 1 Engineering fixes · 2 Parity gate ·
3 Deviation audit · 4 Repairs (variant B) · 5 Verification · 6 Label firewall ·
7 Benchmark protocol · 8 Harness integration · 9 Packaging.

---

## Executive summary — the seven things that matter

1. **The port is faithful, and it reproduces the paper.** A full run of variant A
   scored **94.87% weighted recall** against the reported 95.16%, with four of five
   classes within one point (`RESULTS.md` §2). Appendix A audits every symbol against
   its source cell; the gap is concentrated in **Theft** (8.33% vs 31.82%, 4 of 48
   test edges), which needs seeds to interpret.
2. **It did not run at first.** Three blockers: `random.sample(set)` raises on Python
   ≥3.11, `category_encoders` was missing, and the 6 GB CSV does not fit in 16.9 GB
   of RAM next to pandas. All three are closed.
3. **Memory decided the schedule.** The reference builds a **dense 33k×33k adjacency
   per 35k-flow chunk = 4.3 GB**; the sparse replacement is proven equivalent. The
   predict stage then failed separately on a 10 GB MIG slice under autograd — fixed
   by `no_grad`, leaving >9 GB headroom.
4. **The released code computes a slightly different function from the one the paper
   describes** — thirteen documented places (V1–V13), two material. **But it is that
   code which produced the published numbers**, so the deviations cannot be what
   blocked reproduction. They are now a question about *what the paper actually
   measured*, not obstacles to clear.
5. **The earlier "unreachable" claim is retracted.** It rested on the authors'
   checkpoints scoring 0.44–0.90. That proves checkpoints are not portable across
   feature orderings, not that the method fails to reproduce. Corrected in
   `RESULTS.md` §2.
6. **The contrastive loss does not converge** — slope −1.2e-5/epoch over 300 epochs,
   against epoch noise of 1.4e-2 — yet the downstream probe reaches 94.87%. Explaining
   this is the most valuable open question in the project, because it bears directly on
   the word "self-supervised".
7. **The app is an offline research harness** (D8), so integration is a drop-in
   objective module, not a service (§13).

---

## 1. Decisions

### 1.1 Recorded

| ID | Decision | Your call | Consequence |
|---|---|---|---|
| **D1** | Which version ships? | **Both.** `variant-A` (untouched) proves fidelity **and already reproduces the paper**. `variant-B` (paper-faithful repairs) is still the intended app baseline — but it must now *earn* that place by matching or beating A, not by assumption | v7: A succeeded (§4.1); B is an ablation framework, and may score lower |
| **D4** | Compute | Lab GPU available | CUDA build installed in `negsc-ref` (torch 2.3.0+cu121, GPU visible). ⚠ **DGL publishes no Windows CUDA wheels**, so on native Windows this still runs on CPU. GPU runs need WSL2/Linux — see `ENV.md`. CPU cost is hours, not days, because of V13 |
| **D8** | App profile | Offline research harness: CSV in, artifacts out. No enforcement, no streaming, no latency budget, CPU at runtime, no export path | Integration contract in §13, not a serving contract |
| **D9** | Phase order | Paper fidelity first, app last | Applied in §4–§13 |
| **D10** | What the deviations mean now | Since A reproduces the paper, the deviations are part of how the published numbers were produced. **No repair may be justified by "it moves us toward 95.16%."** Each must be justified on correctness grounds, and its cost reported | Phase 3 documents mechanism; Phase 4 measures cost, not gain |

### 1.2 Still open

| ID | Decision | Needed by | Recommendation |
|---|---|---|---|
| D2 | Python strategy | Phase 0 | two envs: py3.10 reference, py3.12 delivery |
| D3 | Scope of the parity gate | Phase 2 | **1 seed, 3 chunks, 30 epochs** (~1 h GPU) |
| D5 | Data sampling | Phase 1 | cached sample, built once, hashed |
| D6 | Firewall definition | Phase 6 | §11.1 |
| D7 | Evaluation chunking | Phase 7 | the harness's own `limit`/`chunksize` |

---

## 2. Verified current state (2026-10-28)

### 2.1 Machine

| Item | Value | Consequence |
|---|---|---|
| GPU | RTX 3050 Laptop, **6 GB VRAM** | model fits; dense adjacency does not. ⚠ **DGL ships no Windows CUDA wheels**, so the GPU is unusable from native Windows — see `ENV.md`. GPU runs need WSL2/Linux |
| torch | `2.3.0+cpu` | CUDA build not installed |
| dgl | 2.2.1 | behaviour matches on every op exercised |
| Python | 3.12.3 | breaks `random.sample(set)` |
| RAM | 16.9 GB | full CSV + pandas ≈ 20 GB → does not fit |
| Disk | C: 12.5 GB, D: 57.7 GB free | artifacts on D: |

### 2.2 Data

| Item | Value |
|---|---|
| `data_raw/NF-BoT-IoT-v2.csv` | 6.05 GB, **37,763,496 data rows**, 45 columns |
| 3% per-class sample | ≈ **1,132,905 rows** |
| Chunks at `g_len=35000` | **33** → last dropped by the `num=20000` check → **32** |
| Reference sample (`test/`) | 113,291 rows (10% of the 3% sample) |

### 2.3 Measured cost

| Measurement | Value |
|---|---|
| One contrastive step, real chunk, CPU | **1–3 s** |
| Subgraphs per epoch (20,000 draws) | **~40–60** — see V13 |
| Full training, CPU, estimate | **hours**, not days |
| Full training, 4090 (paper) | 3,685 s |
| Published checkpoint accuracy | **0.44 – 0.90**, by feature order |
| **Full variant-A run, A100 MIG 1g.10gb** | **5,652 s (94 min)**, 23 chunks × 300 epochs, 678,684 test edges → **94.87% weighted recall** (`RESULTS.md`) |

⚠ An earlier estimate of 25–43 s/step and ~3.5 days was measured on a synthetic
graph where nearly every node had high degree, so `sub_sam` returned ~20,000
subgraphs. On real data it returns ~50 (V13), and the step costs 1–3 s.

### 2.4 The memory constraint

`main.py:146-150` allocates a dense `[N,N]` adjacency per chunk. Measured: a
35k-flow chunk has ≈**32,800 nodes** → **4.3 GB**, on CPU and again on the GPU.

| Target | Verdict |
|---|---|
| 4090 (24 GB, paper) | fine |
| RTX 3050 (6 GB) | **OOM** |
| CPU / 16.9 GB | fits, no headroom |

The app hits the same wall: **393 MB at 10k flows, 1.51 GB at 20k**. The O(E) gather
in §6.4 lifts both ceilings.

---

## 3. Acceptance contract — what "reproduces" means

| Comparison | Acceptance | Evidence |
|---|---|---|
| Untouched port vs source, same seed, same `PYTHONHASHSEED`, same data | **Exact** to 1e-6 | `PARITY.md` + two loss CSVs |
| After (E)-changes vs before | **Exact** | small-graph equality test |
| Variant B vs paper | **Not a parity claim** — report the gap and the ablations that explain it | `RESULTS.md` |
| Any method vs any method | ≥5 seeds, mean ± std, macro **and** weighted F1 | `results/runs.csv` |

⚠ A single run proves nothing. Unseeded runs of identical code differ by more than
the effects being measured.

---

## 4. Why this order — the answer to "is this better?"

Your proposal: **fix the implementation → characterise the gap → handle it → then the
app.** Yes, that is the right order for the goal you actually have (a paper-faithful
baseline for the app), for three reasons:

1. **The app needs B, not A.** Every day spent perfecting A's distribution statistics
   buys fidelity evidence the app does not consume. v3 spent ~4 days there; v4 spends
   one hour.
2. **The gap analysis is the deliverable, not a preliminary.** "Where does the code
   differ from the paper, and what does each difference cost?" is the intellectual
   contribution of this phase. It has to be a first-class document (`DEVIATIONS.md`),
   not a table buried in a roadmap.
3. **Repairs are only meaningful one at a time.** Building all of B and then measuring
   tells you *that* it changed, not *why*. The audit defines the units; each repair is
   then its own run.

**The one thing I would keep from the old order** is a **one-seed parity gate** in
Phase 2, costing about an hour. Without it, every later number carries an uncontrolled
confound: when B underperforms the paper, you cannot distinguish *"the paper's method
is weaker than advertised"* from *"my port has a bug I never found"* from *"my repair
is wrong"*. One hour buys the elimination of one of those three explanations. If you
disagree, drop the checkbox — nothing else depends on it.

### 4.1 What changes now that variant A reproduces the paper

This section is the most important edit in v7. Read it before planning any further
work.

**The result.** One untuned full run of variant A — the untouched port — reached
94.87% weighted recall against the reported 95.16%. Details in `RESULTS.md`.

**Why that is a surprise.** Phases 3–5 were designed on the premise that the released
code was broken relative to the paper, and that repairing the thirteen deviations
would be what closed the gap. The gap closed without any repair. So the premise was
wrong, and the phases mean something different now:

| Phase | Old purpose | Purpose now |
|---|---|---|
| 3 — deviation audit | find what stops us reaching the paper | **document what the paper's numbers actually came from** |
| 4 — repairs (variant B) | build the version that reaches the paper | **separate the deviations that matter from the ones that do not** |
| 5 — verification | prove the repairs worked | **measure what each repair costs or buys** |

**The inference to take seriously.** If the published numbers came from the released
code, deviations included, then those deviations are not bugs *relative to the
experienced result* — they are part of the mechanism that produced it. Some are
clearly unintended (a random projection inside a loss term cannot be a design
choice). Others may be doing real work:

- **V2** (untrained edge embedder) contributes a stochastic term to `L_edges`. It
  cannot be why the model works, but removing it changes the training signal.
- **V4** (averaging heads instead of concatenating) reduces the parameter count by a
  factor of three in the output projection.
- **V11** (τ hard-coded as 0.01/0.1 in the loss instead of 5) sharpens the
  contrastive objective by two orders of magnitude. This may be load-bearing.
- **V3** (a second GAT over the same graph instead of Eq. 5 interpolation) means the
  positive pair shares its input graph. That should make the task *easier*, not
  harder — consistent with a loss that never moves (§6 of `RESULTS.md`).

**Therefore variant B may score worse than variant A.** That is not a failure; it is
the expected outcome if the paper's description is aspirational and its numbers came
from the code. The project must be willing to report it.

**Consequences for the plan:**

1. **The metric is no longer "closeness to 95.16%".** A repair that drops F1 by two
   points but removes a random projection is still a correct repair, and must be
   reported as a cost, not suppressed as a regression.
2. **The firewall ablation moves up in priority.** The 94.87% is measured with a
   label-fitted target encoder, so it is a leaky number by this project's own
   definition. It is now the largest single threat to the validity of the headline
   result — larger than any deviation.
3. **The flat loss becomes the lead scientific question**, not a footnote. If the
   encoder barely trains and the probe does the work, "self-supervised" needs
   qualifying in every claim made downstream.
4. **The parity gate is unchanged in purpose but sharper in value.** It now
   distinguishes "we computed the same thing" from "we happened to land nearby".
5. **Do not tune toward the paper from here.** The run was untuned; keeping it that
   way is what makes the comparison honest.

---

## 5. Phase 0 — Environment (½ day)

```bash
# Reference env — Python 3.10 keeps random.sample(set) and hash order identical to
# the authors' setup, so the notebooks need only the one-line repair below.
conda create -n negsc-ref python=3.10 -y
conda activate negsc-ref
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install pandas scikit-learn category_encoders networkx scipy matplotlib seaborn pytest

# Delivery env — your port, modern Python
conda create -n negsc-dev python=3.12 -y
```

- ☐ Verify the GPU is used (DGL falls back silently on a wheel mismatch):

```bash
python -c "import torch, dgl; print('cuda', torch.cuda.is_available(), torch.version.cuda, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, '| dgl', dgl.__version__)"
```

- ☐ `pip freeze > requirements.lock.txt` in both; commit them.
- ☐ `ENV.md` with driver, CUDA, versions.

**Exit:** `cuda True`, RTX 3050 named, both envs.

---

## 6. Phase 1 — Engineering fixes: runnable, deterministic, memory-safe (2–3 days)

All (E). None changes what the model computes.

### 6.0 Status — implemented 2026-10-28

| Item | Artifact | State |
|---|---|---|
| B1 `random.sample` | `negsc.py` | ☑ sorted population |
| B2 dependency | `requirements.txt`, `requirements.lock.txt` | ☑ `category_encoders` pinned |
| B3 cache | `make_cache.py` → `data_raw/sample_3pct.csv` (1,131,139 rows, 181 MB) + `cache_manifest.json` | ☑ built and hashed |
| B4 reference repair | `reference/NEGSC_repaired.ipynb`, `reference/REPAIR_DIFF.md` | ☑ R1 + two configuration changes |
| B5 feature order | `data.feature_columns()` | ☑ deterministic by default, `--legacy-order` for A |
| Seeding | `seed.py` | ☑ plus a warning when `PYTHONHASHSEED` is unset |
| Sparse adjacency | `sparse_adj.py` | ☑ proven equal to the dense expression |
| CLI + manifests | `main.py` | ☑ per-run JSON + loss CSV |
| Tests | `tests/` | ☑ 12 passing |

Not done: the two environments in §5 (CUDA build, Python 3.10 reference). See
`ENV.md` — this machine still runs `torch 2.3.0+cpu`.

### 6.1 Blockers

- ☐ **B1** — `negsc.py:97`: `random.sample(sorted(eligible_neighbors), k - 1)`.
  Keep the original line in the py3.10 reference env; apply the fix in the delivery env.
- ☐ **B2** — pin `category_encoders`.
- ☐ **B3** — cached sample (§6.3).
- ☐ **B4** — `reference/NEGSC_repaired.ipynb`, cell 4 only:
  `data['label'] = le.fit_transform(data.label.values)`. Commit the one-line diff.
- ☐ **B5** — deterministic feature order (§6.5).

### 6.2 Seeding

```python
def set_seed(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
```

- ☐ Call first in every entrypoint. `build_ip_port_ids` is the largest variance source.
- ☐ `PYTHONHASHSEED` set by the launcher.

### 6.3 Dataset cache

- ☐ `make_cache.py`: stream, keep 3% per `Attack`, write `sample_3pct.csv` +
  `cache_manifest.json` (sha256, per-class counts).
- ☐ ⚠ A streaming sampler is **not** bit-identical to `groupby().sample(frac=.03,
  random_state=13)`. Both variants consume the same file, so parity stays meaningful.
- ☐ Verify against Table 5: 4,051 / 500,195 / 549,955 / 78,630 / 73.

### 6.4 Sparse adjacency (resolves §2.4)

```python
src, dst = g.edges()
adj_sp = sp.coo_matrix((np.ones(g.num_edges()), (src.numpy(), dst.numpy())),
                       shape=(g.num_nodes(), g.num_nodes()), dtype=np.float32).tocsr()

def block(node_ids):
    return torch.from_numpy(adj_sp[node_ids][:, node_ids].toarray())
```

- ☐ Replace the dense matrix in `main.py` and `_subgraph_adj`.
- ☐ **Equality test**: 2,000-node graph, 100 random id sets, `block(ids)` equals
  `dense[ids].t()[ids]`. Commit the test.
- ☐ Record peak RAM before/after.

### 6.5 Deterministic feature order

```python
LABEL_COL, ID_COLS = 'label', ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
cols_to_norm = [c for c in X_train.columns if c not in ID_COLS + [LABEL_COL, 'h']]
```

- ☐ Persist it into the bundle. `--legacy-order` keeps the original behaviour for A.
- ☐ On the same interpreter and `PYTHONHASHSEED`, `list(set(...))` yields the same
  order in both implementations — A needs no change here.

### 6.6 Smoke

- ☐ `PYTHONHASHSEED=0 python main.py --rows 5000 --epochs 1 --chunks 1`.
- ☐ `pytest`: graph build, finite loss, sparse-block equality, two-run identity.

**Exit:** smoke green; two seeded runs identical.

---

## 7. Phase 2 — The parity gate (½ day)

One seed, three chunks, thirty epochs, everything else at paper settings.

| Setting | Source (`NEGSC_repaired.ipynb`) | Port (`--variant A --legacy-order`) |
|---|---|---|
| Seed / `PYTHONHASHSEED` | 42 / 0 | 42 / 0 |
| Data | `sample_3pct.csv` | identical |
| `g_len`, `k1`, `num`, epochs | 35000, 3, 20000, 30 | identical |
| Device | cuda | cuda |

- ☐ Log `chunk, epoch, loss1, loss2, loss3, total` for both; diff to 1e-6.
- ☐ If they differ, bisect in this order: graph build → subgraph batch (dump
  `node_neighbor_cen`) → forward (`z`) → individual loss terms.
- ☐ `PARITY.md` with both curves and the diff.
- ☐ Tag the validated untouched port `variant-A`; it is the artifact of record and the
  control for every later ablation.

Known cosmetic differences that cannot affect numerics: `edata` key
`'edge_emb'` vs `'score'`; `expand` vs `repeat`; `train()` taking `optimizer`;
`sub_sam` copying lists instead of mutating the adjacency sets.

📌 *"Under a fixed seed the port reproduces the released implementation's training
trajectory exactly."*

---

## 8. Phase 3 — The deviation audit (3–5 days)

> **Reframed in v7.** This is no longer a search for what blocks reproduction — the
> reproduction already happened (§4.1). It is now the documentation of **what the
> paper's numbers actually came from**: which of the thirteen deviations the result
> depends on, and which are accidental. The output is the same document; the reason
> for writing it is different, and so is the standard of evidence.

**This is the deliverable you asked for: what the paper says vs what the code does.**
Output: `DEVIATIONS.md`, one entry per item, in this format:

```
### V7 — degree bound
Paper  §3.4.2, Table 4: d ∈ {3, 15}; nodes with degree < d are discarded.
Code   main.py:43 k1 = 3 only; no d=15 configuration exists.
Effect larger neighbourhoods → more context per subgraph; cost grows linearly in d.
Repair parameterise; run both values.
Verify results for d=3 must be identical to A (the default is unchanged).
```

### 8.1 The table to complete

| ID | Paper | Released code | Materiality |
|---|---|---|---|
| **V1** | Table 4 + text: lr = 0.001; τ and wd part of the configuration | Dead constants; `Model(...)` gets τ=0.5, `Adam()` gets lr=1e-3, wd=0 (`main.py:129-130`) | medium — the effective values are defaults, not choices |
| **V2** | Alg. 1 l.15: edge embedding = `CONCAT(z_v, z_u)` | a fresh random `MLPPredictor` **every step**, never optimised (`negsc.py:242`) | **high** — one of the two loss terms trains on noise |
| **V3** | Eq. 5: node interpolation through `(z_v ‖ z_vu ‖ z_u)` over the sampled subgraph | a second full GAT over the *same* graph, using **raw edge features**, not edge embeddings (`main.py:114`) | **high** — the positive pair is near-identical; little augmentation |
| **V4** | Fig. 4: attention heads are **concatenated** | averaged (`negat.py:89`) | medium |
| **V5** | node features `F_V`-dimensional | dim inferred from edge dim; only correct because this dataset has 39 | low |
| **V6** | Alg. 1 l.15 | covered by V2 | — |
| **V7** | d ∈ {3, 15} | `k1=3` only | medium |
| **V8** | binary **and** multiclass reported | multiclass only | medium (depends on the app) |
| **V9** | two-term loss | third node-WD term computed and discarded (`negsc.py:213-215`) | low — wasted compute, matches the formula |
| **V10** | Table 14: 2,113,290 test flows | ≈339,871 implied by Table 5 + the 3% sample | documentation |
| **V11** | Eq. 8–9: `exp(-WD/τ)` with the same τ as the cost matrix | hard-coded `0.01` (WD) and `0.1` (GWD) (`negsc.py:214, 222`) | medium — τ never reaches the loss |
| **V12** | Eq. 10: `L = L_edges + L_topology` | `0.5*loss3 + 0.5*loss2` (`negsc.py:231`) | low — a factor-2 rescale, equivalent to halving lr |
| **V13** | `N` sampled subgraphs per step, with `num` drawn centres | the degree filter (`len(neighs) >= k`) discards **>99.7%** of draws: measured **39–57 subgraphs** per epoch from 20,000 draws on real chunks | **high** — the contrastive batch is ~50, not 20,000; every per-step loss is computed on a tiny sample |

### 8.2 Ruled out — things that look different but are not

Recorded so the audit reads as a complete job rather than a list of complaints:

| Suspicion | Verdict |
|---|---|
| "Encoder runs before sampling, paper samples first" | Paper §3.4.2 describes encoder-then-sample; the code matches |
| "Two weight matrices where the paper has one" | Paper Eq. 1 uses `W^k`; Eq. 4 uses `𝒘^k₁` — two matrices are correct |
| "Softmax over in-edges, not neighbours" | Graph is `to_directed()`, so in-edges = neighbours |
| "Subgraph edges are not created" | `sub_adj` extracts the k×k block, matching "create edges between v and these d nodes" |
| "Negative sampling is wrong" | One permutation index, adjusted to avoid i — a valid M=2 draw |

### 8.3 Completion criteria

- ☐ Every V-item has paper location, code location, effect, repair, verification.
- ☐ The two **high** items (V2, V3) have a written design for their repair.
- ☐ The ruled-out table is filled from evidence, not memory.

---

## 9. Phase 4 — Repairs: building Variant B (1.5–2 weeks)

> **Reframed in v7.** Variant B is no longer "the version that reaches the paper" —
> variant A already does. B is now an **ablation framework**: apply one deviation fix
> at a time and measure what it costs. A repair that lowers F1 is a legitimate result
> and must be reported, not reverted. The question is "which parts of the released
> code carry the result?", not "how do we get closer to 95%?".
>
> ⚠ If B underperforms A, that is the expected outcome when the paper's description is
> aspirational and its numbers came from the code. Write it down either way.

### 9.1 Rules

1. **One V at a time.** Branch per item; the ablation grid in Phase 10 depends on it.
2. **Never repair V2 and V3 in the same commit** — they are the two that could
   plausibly explain everything, and you need them separated.
3. Every repair keeps A runnable (`--variant A`).

### 9.2 The two material repairs

**V2 — edge embedding (remove the noise term).**

- Default per the paper: `ẑ_e = concat(z_u, z_v)`, 78-dim, **no parameters**.
  `_edge_features`'s reshape changes from 39 to 78; `_wd` normalises, so dimension is
  free.
- Alternative for the ablation grid: a registered, trained `Linear` projecting to 39.
- Verify: `L_edges` is now a deterministic function of `z` for fixed inputs.

**V3 — the Eq. 5 generator.**

```python
class Generator(nn.Module):
    """z'_v = σ( Σ_u α_vu · W_g · (z_v ‖ z_vu ‖ z_u) ),  z_vu = cat(z_v, z_u)."""
    def __init__(self, feat, out_feat):
        super().__init__()
        self.W = nn.Linear(4 * feat, out_feat)
        self.a = nn.Parameter(torch.rand(2 * out_feat, 1))
```

- Messages are built from **edge embeddings**, not raw edge features.
- Topology is copied 1:1, as the paper requires.
- The generator must not read the original `edge_feats` at all; if it does, the
  augmentation is not an augmentation.

### 9.3 The rest

| ID | Repair |
|---|---|
| V1 | `Model(..., tau=cfg.tau)`; `Adam(lr=1e-3, weight_decay=1e-5)`; sweep τ ∈ {0.5, 5} |
| V4 | `out_feat = 39 // 3 = 13` per head, `torch.cat` → 39 |
| V5 | make `F_V = 39` explicit |
| V7 | `--k1 {3,15}` |
| V8 | `--task {binary,multiclass}` (`Label` vs `Attack`) |
| V9 | delete the dead computation |
| V11 | replace 0.01/0.1 with `cfg.tau` |
| V12 | decide sum vs mean and document it — do not leave it implicit |

**Exit:** `python main.py --variant B` trains end to end; A still runs unchanged.

---

## 10. Phase 5 — Verification and ablations (1 week)

- ☐ **Ablation grid**, each row one V applied to A: V1, V2, V3, V4, V7, V11.
  Three seeds each. Report Δ weighted F1, Δ macro F1, per-class recall.
- ☐ **V2 vs V3 interaction**: run them together last. If the combined effect is much
  larger than either alone, the paper's method depends on both.
- ☐ Optional full parity (A vs source, 5 seeds) only if a reviewer demands it.
- ☐ `RESULTS.md`: every number with its `run_id`.

---

## 11. Phase 6 — The label firewall (3–5 days)

### 11.1 Definition (D6)

> No information about a row's label reaches any feature, statistic, encoder or graph
> structure used to produce that row's prediction — at training and at evaluation.

Vectors: supervised feature encoding (**present**, `data.py:110`), label-derived
columns (none), test statistics in preprocessing (none), split leakage (none),
transductive batching (structural, D7).

### 11.2 Arms — and their equivalents in the app

| Arm | Encoding | App equivalent |
|---|---|---|
| **L0 — leaky** | `TargetEncoder.fit(X_train, y_train)` | `negsc_legacy/data.py:101-105` (deferral **C3**) |
| **L1 — firewall on** | one-hot / frequency, no labels | `NetflowPreprocessor` |
| **L2 — control** | an explicitly label-derived feature | — |

- ☐ ≥3 seeds; Δ = L0 − L1; run the pair **inside the harness** too, so C3 gets a number.
- ☐ Note that a three-column target encoder is a *mild* leak; L2 shows the mechanism at
  full strength and must be labelled as a control.

📌 *"The deferred leak C3 costs N points of weighted F1; L1 is the honest baseline."*

---

## 12. Phase 7 — Benchmark protocol (2–3 days)

- ☐ Freeze `BENCHMARK.md` first: stratified 70/30 `random_state=123`; ≥5 seeds;
  macro **and** weighted F1; per-class recall; confusion matrix; PR-AUC; cost.
- ☐ ⚠ Keep the paper's supervised binary probe (two-class CE) separate from the app's
  `OneClassHead` (anomaly detection, R@1%FPR). They are different measurements and
  must not share a table.
- ☐ `results/runs.csv`, one row per run.

---

## 13. Phase 8 — Integration with the harness (2–3 days)

Deliverable: `negsc_objective.py`, a drop-in replacement for `negsc_legacy`.

| Harness concept | Contract |
|---|---|
| `Objective.requires_labels` | `False` |
| `EdgeArrays.y_binary` / `y_multiclass` | never read |
| graph keys `h`, `orig_idx` | consume `h`; add no `label` key |
| multiclass head `(pred, proba[N,C])` | expose embeddings; the head classifies |
| `OneClassHead` | optional `score()` from the same embeddings |

**Clauses, each a test:**

1. Poisoned-label test — permute labels, assert identical output (catches content-level
   leaks that a key-name guard cannot).
2. No new edge keys.
3. Deterministic per seed and chunk.
4. Device-agnostic — GPU here, CPU there (`map_location='cpu'`).
5. Memory O(E) — shared sparse gather (§6.4).
6. Chunk parity with the harness's own `limit`/`chunksize`.

**Out of scope until you say otherwise** (D8): streaming, latency budgets, GPU at
runtime, ONNX/TorchScript. Each is a new phase, and each must be decided **before** the
FastAPI backend encodes the offline assumption.

**Blocking questions:** the `Objective` protocol signature; where `negsc_legacy` is
selected; whether your graph builder randomises destination IPs as the legacy path
does (if not, the comparison is invalid); whether `OneClassHead` consumes NEGSC
embeddings.

---

## 14. Phase 9 — Packaging (1–2 days)

- ☐ `README.md`: decisions, variant switch, firewall switch, one command per protocol.
- ☐ `RESULTS.md`, `LIMITATIONS.md`, `DEVIATIONS.md`.
- ☐ `bundle/`: `state_dict` only, `feature_columns.json`, `manifest.json`.
- ☐ Tag `v1.0-baseline`.

**`LIMITATIONS.md` must state:** the released code does not implement Eq. 5 (V3),
trains part of the loss on a random projection (V2), averages heads (V4); the paper's
artifacts are not reproducible (44–90%); the graph topology is synthetic because every
destination IP is randomised; the headline protocol is a supervised probe, so
"unsupervised" describes only the encoder; Table 5 vs Table 14 disagree on test size.

---

## 15. Milestones

| # | Milestone | Depends | Est. |
|---|---|---|---|
| M0 | Both envs green on GPU | D2 | ½ day |
| M1 | Engineering fixes + smoke + determinism | Phase 1 | ☑ done 2026-10-28 |
| M1.5 | Full variant-A run at scale (1 seed) | M1 | ☑ done 2026-11-02, `RESULTS.md` |
| M1.6 | Variance: 3 more seeds | M1.5 | **in progress — do this first** |
| M1.7 | Firewall ablation L0 vs L1 | M1.6 | promoted: the 94.87% is leaky |
| M2 | Parity gate exact | M1 | ½ day |
| M3 | `DEVIATIONS.md` complete | M2 | 3–5 days |
| M4 | Variant B training end to end | M3 | 1.5–2 weeks |
| M5 | Ablation grid | M4 | 1 week |
| M6 | Firewall ablation (repo + harness) | M5 | 3–5 days |
| M7 | Benchmark tables | M6 | 2–3 days |
| M8 | Objective module + golden test | M7 | 2–3 days |
| M9 | Harness integration | §13 answers, M8 | outside repo |

GPU time: ~3–5 h per full run; M5+M6 need ~15–20 runs → budget **3–4 days of
wall-clock GPU**, which is the real bottleneck, not the coding.

---

## 16. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Quoting the paper's 95% as the baseline | High | Fatal to credibility | always quote measured numbers with protocol |
| R2 | Skipping the parity gate, then blaming the paper for a port bug | Medium | Wasted months | the 1-hour gate in §7 |
| R3 | 6 GB VRAM OOM on the dense adjacency | **High** | Blocks all runs | §6.4 + equality test |
| R4 | Baseline weak → app looks better than it is | High | Compromises the result | ship B, and beat B |
| R5 | Firewall applied to one arm only | Medium | Comparison meaningless | single switch, golden test |
| R6 | Harness chunking ≠ evaluation chunking | Medium | Numbers not comparable | D7, golden test |
| R7 | Repairs bundled → unattributable results | Medium | The ablation grid is worthless | one V per branch (§9.1) |
| R8 | RAM forces silent downscaling | Medium | Not comparable to the paper | record every scale factor in `config.json` |
| R9 | `negsc_legacy` keeps the leak while the new featuriser does not | **High** | Leakage claim inverted | run L0/L1 inside the harness (§11.2) |
| R10 | API built on the offline assumption, then streaming turns out to be required | Medium | A whole phase of rework | §13 out-of-scope list decided first |

---

## 17. Experiment log schema

```csv
run_id,date,git_sha,variant,deviations_applied,python_env,device,seed,pyhashseed,data_sha256,rows,chunks,epochs,k1,num,tau,lr,wd,firewall,arm,task,weighted_recall,weighted_f1,macro_f1,accuracy,train_sec,notes
```

---

## 18. Command reference

```bash
conda activate negsc-dev
python -c "import torch; print(torch.cuda.is_available())"

python make_cache.py --src data_raw/NF-BoT-IoT-v2.csv --frac 0.03 --out data_raw/sample_3pct.csv
PYTHONHASHSEED=0 python main.py --rows 5000 --epochs 1 --chunks 1

# parity gate
PYTHONHASHSEED=0 python main.py --variant A --legacy-order --seed 42 --chunks 3 --epochs 30 --run-id A_parity

# baseline B, five seeds
for s in 0 1 2 3 4; do
  PYTHONHASHSEED=0 python main.py --variant B --firewall on --seed $s --run-id B_$s
done

# ablation: one deviation at a time
PYTHONHASHSEED=0 python main.py --variant A --apply V3 --seed $s --run-id abl_V3_$s
```

---

## Appendix A — Function-by-function equivalence audit

Verdicts are about behaviour, not style.

### A.1 Data path

| Port | Source | Comparison | Verdict |
|---|---|---|---|
| `load_raw` | cell 2 | `pd.read_csv`; `nrows` added for dev | equivalent |
| `sample_by_attack` | cell 3 | `groupby('Attack').sample(frac, random_state=13)` | equivalent |
| `build_ip_port_ids` | cell 3 | same order; random dst IP, `src:port` / `dst:port`, ports dropped | equivalent (unseeded in both) |
| `encode_labels` | cell 4 | source has a syntax error; port does `fit` + `transform` | equivalent to intent |
| `split_data` | cell 5 | same frame (label included) as X, same args | equivalent |
| `fit_target_encoder` | cell 6 | same columns, fit on train only | equivalent |
| `fit_scale_and_build_feature_vector` | cell 7 | same set, `set()` order nondeterministic in both | equivalent, ⚠ B5 |
| `scale_and_build_feature_vector` | cell 34 | `transform` not `fit` | equivalent |
| `build_train_graphs` | cells 16–17 | same slicing, `MultiGraph` → `to_directed`, same drop rule | equivalent |
| `build_full_train_graph` | cells 38–39 | same | equivalent |
| `build_test_graph` | cell 34 | same, `ndata['feature']` | equivalent |

### A.2 Model path

| Port | Source | Comparison | Verdict |
|---|---|---|---|
| `MLPPredictor` | cell 10 | same `Linear(2F→C)`; key renamed `score`→`edge_emb` | equivalent |
| `GATlayer` | cell 11 | `W`, `W_msg`, `a`, xavier, LeakyReLU, `edge_softmax`, weighted sum, ReLU | identical |
| `MultiHeadGATLayer` | cell 11 | concat then `mean(1)` | identical |
| `GAT` | cell 11 | hardcoded 39 | identical (V5) |
| `sub_sam` | cell 13 | same slot logic, same `[:-1]`; differs in not mutating `adj_lists` | equivalent (needs B1 on py≥3.11) |
| `Model.forward/embed/loss` | cell 14 | identical | identical |
| `Model.sub_loss_batch` | cell 14 | same tiling, shuffle, labels, `0.5/0.5` | identical |
| `_edge_features` | cell 14 | same per-step random `MLPPredictor`, same reshape 39 | identical (V2) |
| `_subgraph_adj` / `_subgraph_centers` | cell 14 | same gather and reshape | identical |
| `_wd` family | cell 14 | same cost, global min/max threshold, Sinkhorn β=0.5, 40 iters | identical |
| `_gwd` family | cell 14 | same GW iterations, `lamda=0.1`, plan reuse | identical |
| `train` | cell 15 | same order, `retain_graph=True` | identical |

### A.3 Predict path

| Port | Source | Comparison | Verdict |
|---|---|---|---|
| `embed_graphs` | cell 44 | same keys, train detached | identical |
| `train_classifier` | cell 44 | same head, lr 1e-3, 10k steps | identical |
| `predict` / `decode_labels` | cells 44–47 | argmax, inverse transform | identical |
| `evaluate` | cells 49–52 | same report; port defaults `normalize=True`, `test.ipynb` plots raw | equivalent for metrics |
| `run_inference.py` | `test.ipynb` | re-fits encoders in a new process | ⚠ superseded by the bundle |

**Bottom line:** the untouched port is a correct reproduction. Three caveats — B1,
the edge-key rename, inference-time re-fitting — none requiring a logic change.

---

## Appendix B — Evidence collected during analysis

| Observation | Value |
|---|---|
| Published checkpoint, natural column order | 0.444 |
| Published checkpoint, `PYTHONHASHSEED` 0–11 | 0.52, 0.52, 0.60, 0.60, 0.70, 0.78, 0.78, 0.83, 0.84, 0.88, 0.90, 0.90 |
| Random-IP seed change (order fixed) | 0.5997 → 0.6003 |
| Column-order change (IP seed fixed) | 0.52 → 0.90 |
| Nodes per 35k-flow chunk | ≈ **32,800** → 4.3 GB dense |
| One contrastive step, CPU | 25–43 s |

The checkpoint is bound to an unrecoverable feature ordering. That does not prove the
paper wrong; it proves the released artifacts are not reproducible — the claim you can
defend, using the authors' own files.

## Appendix C — Paper reference numbers (NF-BoT-IoT-v2)

| Task | Metric | Value |
|---|---|---|
| Binary | Acc / Precision / Recall / F1 | 99.64 / 99.29 / 99.64 / 99.46 % |
| Multiclass | weighted Recall / F1 | 95.16 / 95.15 % |
| Per-class Recall | Benign / DDoS / DoS / Recon / Theft | 61.40 / 97.46 / 93.56 / 91.11 / 31.82 % |
| Cost | train / size / test | 3,685 s / 334.5 KB / 0.26 µs·sample⁻¹ |
| Hyperparameters | K / heads / lr / d | 1 / 3 / 1e-3 / 3 |

## Appendix D — File map

| File | Role | Status |
|---|---|---|
| `data.py` | loading, sampling, encoding, graphs | complete; needs §6.3, §6.5 |
| `negat.py` | NEGAT encoder | complete; V4/V5 in Phase 4 |
| `negsc.py` | sampling, contrastive loss | complete; **B1**, V2/V3/V9/V11 |
| `predict.py` | embedding, probe, metrics | complete |
| `main.py` | training entrypoint | complete; needs variant switch, §6.4 |
| `run_inference.py` | checkpoint inference | ⚠ superseded by the bundle |
| `negsc_objective.py` | harness drop-in (§13) | to create |
| `DEVIATIONS.md` | Phase 3 deliverable | to create |
| `reference/NEGSC_repaired.ipynb` | source with B4 fixed | to create |
| `make_cache.py`, `seed.py`, `sparse_adj.py`, `bundle/`, `tests/` | harness | to create |

## Appendix E — What the app still owes this project

1. The `Objective` protocol signature.
2. Where `negsc_legacy` is selected.
3. Whether the graph builder randomises destination IPs.
4. Whether `OneClassHead` consumes NEGSC embeddings.