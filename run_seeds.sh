#!/usr/bin/env bash
# run_seeds.sh -- variance estimate for variant A (ROADMAP M1.6).
#
# `full_v3` (seed 42) reached 94.87% weighted recall. One run has no error bar.
# This runs three more seeds with everything else identical, then prints a
# summary so the result can be reported as mean ± std.
#
# WHAT THE SEED CHANGES
#   The reference seeds nothing, so all randomness is one stream. A different
#   --seed changes:
#     * destination-IP randomisation in build_ip_port_ids  -> graph topology
#     * subgraph sampling in sub_sam                       -> training examples
#     * node batching (torch.randint)                      -> training examples
#   The train/test split does NOT change (fixed random_state=123), and neither
#   does the feature-column order (fixed PYTHONHASHSEED).
#   So each seed is a different graph with a different training trajectory --
#   which is exactly the replication unit the paper's protocol implies.
#
# USAGE (server):
#   cd ~/negsc-fraud-detection
#   source ~/.venvs/negsc/bin/activate
#   bash run_seeds.sh              # seeds 1 2 3, sequential, ~4.7 h
#   bash run_seeds.sh 7 8 9        # any seeds you like
#
# Run it inside tmux, or with nohup:
#   nohup bash run_seeds.sh > outputs/seeds.out 2>&1 &
#   tail -f outputs/seeds.out

set -euo pipefail

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(1 2 3)
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p outputs

# Reproducibility: hash order fixed for all runs, cuBLAS deterministic.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

echo "=== variance runs: seeds ${SEEDS[*]} ==="
echo "start: $(date)"
echo

for SEED in "${SEEDS[@]}"; do
  RUN_ID="seed_${SEED}"
  if [ -f "outputs/run_${RUN_ID}.json" ]; then
    echo "[skip] ${RUN_ID} already has a manifest -- delete it to rerun"
    continue
  fi
  echo "--- ${RUN_ID} ($(date +%H:%M:%S)) ---"
  python main.py --seed "${SEED}" --run-id "${RUN_ID}" --log-every 25 \
    2>&1 | tee "outputs/${RUN_ID}.log"
  echo "--- ${RUN_ID} finished ($(date +%H:%M:%S)) ---"
  echo
done

echo "=== summary ==="
python - <<'PY'
import json, statistics, glob, os

runs = [("full_v3", "outputs/run_full_v3.json")]
for path in sorted(glob.glob("outputs/run_seed_*.json"),
                   key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0])):
    runs.append((os.path.basename(path)[4:-5], path))

rows, missing = [], []
for name, path in runs:
    if not os.path.exists(path):
        missing.append(name)
        continue
    m = json.load(open(path))
    met = m.get("metrics") or {}
    if not met:
        missing.append(name)
        continue
    rows.append((name, m["config"]["seed"], met))

if not rows:
    raise SystemExit("no completed runs found")

hdr = f"{'run':<10}{'seed':>6}{'acc':>9}{'w_recall':>10}{'w_f1':>9}{'macro_f1':>10}"
print(hdr)
print("-" * len(hdr))
for name, seed, met in rows:
    print(f"{name:<10}{seed:>6}{met['accuracy']:>9.4f}"
          f"{met['weighted_recall']:>10.4f}{met['weighted_f1']:>9.4f}"
          f"{met['macro_f1']:>10.4f}")

print()
for key in ("weighted_recall", "weighted_f1", "macro_f1", "accuracy"):
    vals = [m[key] for _, _, m in rows]
    if len(vals) > 1:
        sd = statistics.stdev(vals)
        print(f"{key:<16} mean {statistics.mean(vals):.4f}  "
              f"sd {sd:.4f}  min {min(vals):.4f}  max {max(vals):.4f}")
    else:
        print(f"{key:<16} {vals[0]:.4f}  (single run)")

print()
print("per-class recall (one row per class, one column per run):")
classes = list(rows[0][2]["per_class"])
print(f"{'class':<16}" + "".join(f"{n:>12}" for n, _, _ in rows))
for c in classes:
    print(f"{c:<16}" + "".join(f"{m['per_class'][c]['recall']:>12.4f}"
                               for _, _, m in rows))

if missing:
    print()
    print("NOT included (missing or incomplete):", ", ".join(missing))
PY

echo
echo "done: $(date)"
echo "Next: copy the numbers into RESULTS.md as mean ± std."
