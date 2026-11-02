# Running NEGSC on the lab server

A short runbook. Read it once, then you only need the **"Every time"** section.

---

## The 30-second version

```bash
ssh -p 2003 darchiny@192.168.1.12
cd ~/negsc-fraud-detection && git pull origin main
source ~/.venvs/negsc/bin/activate
nohup env PYTHONHASHSEED=0 python main.py --run-id <name> > outputs/<name>.log 2>&1 &
tail -f outputs/<name>.log
```

That is the whole loop. Everything below explains why.

---

## 1. The machine

| | |
|---|---|
| Address | `192.168.1.12`, port **2003** (not 22), user `darchiny` |
| OS | Ubuntu 22.04, a container (`a33550e15e96`) |
| GPU | **A100 80 GB, but you get a MIG slice**: ~9.5 GB, 14 of 108 SMs |
| Driver | 580.178.04, CUDA 13.0 (runs the `cu121` wheels fine) |
| CPU / RAM / disk | 84 cores / 314 GB / 61 GB free on `~` |
| Code | `~/negsc-fraud-detection` (cloned from GitHub) |
| Python env | `~/.venvs/negsc` (Python 3.10, torch 2.3.0+cu121, dgl 2.2.0+cu121) |
| Data | `~/negsc-fraud-detection/data_raw/sample_3pct.csv` (173 MB) |

**The MIG slice is the important number.** `nvidia-smi` says "A100 80GB", but the
usable memory is 9,728 MiB. Training fits. If the `[predict]` stage ever dies
with a CUDA out-of-memory error, that limit is why — not a code bug.

---

## 2. First-time setup (do this once)

You already did it. Keep it here so a new machine or a rebuilt container takes
20 minutes instead of a day.

### 2.1 Check the GPU is visible

```bash
nvidia-smi
```

Look for `MIG devices:` and note the MiB figure. Also check nobody else is
training:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

### 2.2 Get the code

```bash
cd ~ && git clone https://github.com/Ilya-Jahed/negsc-fraud-detection.git
cd negsc-fraud-detection
```

(`git` is installed. If it weren't: `sudo apt install git`.)

### 2.3 Copy the dataset (from **PowerShell on Windows**, not the server)

```powershell
scp -P 2003 "D:\AnomalyDetection_Lab\negsc-fraud-detection\data_raw\sample_3pct.csv" darchiny@192.168.1.12:~/negsc-fraud-detection/data_raw/
scp -P 2003 "D:\AnomalyDetection_Lab\negsc-fraud-detection\data_raw\cache_manifest.json" darchiny@192.168.1.12:~/negsc-fraud-detection/data_raw/
```

Verify on the server — it must match the manifest:

```bash
sha256sum ~/negsc-fraud-detection/data_raw/sample_3pct.csv
# expect: e2157235acae8f34f35c7f81391dd143bf1fddeacb16c3c641eec2b9950ff0a9
```

### 2.4 Install `uv` (no sudo needed)

```bash
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

`uv` downloads its own Python, so you do not need `python3-venv` or apt.

### 2.5 Build the environment

```bash
cd ~/negsc-fraud-detection && bash setup_server.sh
```

This installs everything into `~/.venvs/negsc` and must end with:

```
dgl graph on GPU: cuda:0
```

If it stops earlier, see §7.

### 2.6 Make the settings permanent

Two environment variables matter. `LD_LIBRARY_PATH` is already handled by
`setup_server.sh` (it patches the venv's `activate` script). Add the other two:

```bash
cat >> ~/.venvs/negsc/bin/activate <<'EOF'

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
EOF
```

- `CUBLAS_WORKSPACE_CONFIG` — without it, torch warns that cuBLAS matmuls are
  not deterministic, and two runs with the same seed can differ. This is what
  makes results reproducible.
- `PYTHONUNBUFFERED` — without it, `tail -f` on a redirected log shows nothing
  for long stretches, because Python buffers output.

---

## 3. Every time

### 3.1 Connect and update

```bash
ssh -p 2003 darchiny@192.168.1.12
cd ~/negsc-fraud-detection
git pull origin main          # only needed if code changed on the laptop
source ~/.venvs/negsc/bin/activate
```

You know the venv is active when your prompt starts with `(negsc)`.

### 3.2 Sanity check (10 seconds, saves hours)

```bash
python -c "import torch,dgl; print(torch.cuda.is_available(), dgl.__version__)"
```

Must print `True 2.2.0+cu121`. If it prints `False`, stop — see §7.

### 3.3 Run

Detached, so closing your laptop or losing Wi-Fi does not kill it:

```bash
nohup env PYTHONHASHSEED=0 python main.py --run-id full_v3 > outputs/full_v3.log 2>&1 &
```

`nohup` is used because **tmux is not installed** on this machine. If you can
`sudo apt install tmux`, prefer it — with tmux you can reattach and interact;
with `nohup` you can only read the log.

### 3.4 Watch

```bash
tail -f outputs/full_v3.log      # Ctrl-C is safe, the job keeps running
```

Check it is alive:

```bash
ps -p <PID>                      # PID is printed when you start it
```

---

## 4. Useful flags

| Flag | Use |
|---|---|
| `--rows 200000` | quick test on the first 200k rows |
| `--chunks 1 --epochs 2` | fastest valid run (~15 s) |
| `--skip-predict` | train only, no downstream classifier |
| `--run-id NAME` | names every output file; always set this |
| `--seed 42` | default; change for multi-seed experiments |

**Never pass `--sample-frac`.** The cache is already the 3% sample. Adding
`--sample-frac 0.03` would take 3% of 3% and train on 33,934 rows instead of
1,131,139. (This bug bit us once; the script now warns if you try.)

**Always set `PYTHONHASHSEED=0`.** Without it the feature-column order changes
between processes and no two runs are comparable.

---

## 5. Results

Everything lands in `outputs/`, suffixed by `--run-id`:

| File | What it is |
|---|---|
| `run_<id>.json` | config, seed, git SHA, environment, **all metrics** |
| `loss_<id>.csv` | chunk, epoch, subgraphs, loss |
| `confusion_matrix_<id>.png` | the plot |
| `model_<id>.pt`, `log_<id>.pt` | trained weights |
| `<id>.log` | full console transcript |

Quick look without copying anything:

```bash
python -c "
import json; m = json.load(open('outputs/run_full_v3.json'))
print(json.dumps(m['metrics'], indent=2)); print(m['env'])"
```

Confirm it used the GPU — look for `'device': 'cuda:0'` and `'dgl_cuda': True`.

Copy everything back (**PowerShell on Windows**):

```powershell
scp -P 2003 -r "darchiny@192.168.1.12:~/negsc-fraud-detection/outputs" "D:\AnomalyDetection_Lab\negsc-fraud-detection\outputs_lab"
```

---

## 6. Comparing against the reference

This is **Phase 2 of `ROADMAP.md`** — the parity gate. It answers: *does our
port reproduce the authors' code?*

Run both sides under identical conditions:

| | Reference | Port |
|---|---|---|
| Where | `reference/NEGSC_repaired.ipynb` (Jupyter on the server) | `main.py` |
| Seed / hashseed | 42 / 0 | 42 / 0 |
| `--chunks` / `--epochs` | 3 chunks / 30 epochs | 3 / 30 |
| Data | `data_raw/sample_3pct.csv` | same file |

```bash
# port side
nohup env PYTHONHASHSEED=0 python main.py --variant A --legacy-order \
  --chunks 3 --epochs 30 --skip-predict --run-id parity_port \
  > outputs/parity_port.log 2>&1 &
```

Then run the notebook with the same seed. The notebook prints `loss1`, `loss2`,
`loss3` each epoch but does not save them — collect them into
`outputs/loss_parity_reference.csv` with columns `chunk,epoch,loss` before you
compare.

Compare the two loss series:

```bash
python - <<'PY'
import csv
a = [float(r['loss']) for r in csv.DictReader(open('outputs/loss_parity_port.csv'))]
b = [float(r['loss']) for r in csv.DictReader(open('outputs/loss_parity_reference.csv'))]
print(len(a), len(b), max(abs(x - y) for x, y in zip(a, b)))
PY
```

Expected: the max difference is ~0 or ~1e-6. A large difference means the port
is not faithful, and **every later result is suspect** — fix that before
anything else.

If the two don't match, the bisection order in `ROADMAP.md` (Phase 2) isolates
it: graph construction → subgraph batch → forward pass → individual loss terms.

What this does *not* prove: that we match the paper. That is a separate
question, answered by the deviation audit (Phase 3) and the repairs (Phase 4).

---

## 7. When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot load Graphbolt C++ library` / `libnvrtc.so.12` | venv not activated, so `LD_LIBRARY_PATH` is unset | `source ~/.venvs/negsc/bin/activate` |
| `cannot import name 'DILL_AVAILABLE'` | torchdata 0.7.1 on Linux | `uv pip install "torchdata==0.8.0"` |
| `device=cpu` in the `[run]` line | GPU not visible to this process | `nvidia-smi`; check the venv is active |
| `subgraphs=0` every epoch | graph has no node with degree ≥ k1 | check `--rows`/data; on a tiny slice use `--min-nodes 50` |
| `sampled to N rows` in `[data]` | `--sample-frac` passed to the cache | remove it |
| Log file stays empty | Python buffering | `export PYTHONUNBUFFERED=1` |
| `cublas` determinism warning | missing env var | `export CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| CUDA out of memory in `[predict]` | 9.5 GB MIG slice, full-train graph | report it; the fix is a code change, not a flag |
| `git pull` says "not a git repository" | wrong folder | `cd ~/negsc-fraud-detection` first |

---

## 8. Things worth remembering

1. **`nohup` or `tmux`, never a bare session.** The first run died with
   `client_loop: send disconnect: Connection reset` and took the terminal
   output with it.
2. **The 23 chunks are sequential, not parallel.** The model trains on chunk 1,
   then 2, then 3… Each chunk is a separate graph.
3. **~50 subgraphs per epoch, not 20,000.** `--num 20000` draws 20,000 nodes,
   but only nodes with out-degree ≥ 3 become subgraphs — about 0.2% of them.
   This is deviation **V13** in the roadmap, and it is why a full run is hours
   rather than weeks.
4. **Weighted F1 hides the real story.** Look at macro F1 and per-class recall
   too. In the first (badly sampled) run, accuracy was 98.3% while Theft scored
   0.0 and Benign 0.139.
5. **Don't compare to the paper's 95.16% yet.** That number comes from a
   different protocol. Compare against your own measured baseline, run the same
   way, and record it in `RESULTS.md`.
