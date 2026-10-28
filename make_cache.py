"""
make_cache.py -- build the pre-sampled dataset every run consumes
(ROADMAP, Phase 1 -- dataset cache; blocker B3).

WHY
---
`data_raw/NF-BoT-IoT-v2.csv` is 6.05 GB / 37,763,496 rows. On this machine
(16.9 GB RAM) `pd.read_csv` on it cannot succeed alongside pandas' working
copy. The reference notebook reads the whole file, then samples 3% per class
with `groupby('Attack').sample(frac=0.03, random_state=13)`.

Rather than re-derive a sample in every run, we materialise it once. Two
consequences, both recorded in the manifest:

  1. **Not bit-identical to the source expression.** The source draws from a
     fully-loaded frame; we stream in chunks and keep each row with a
     per-class probability. The two samples are statistically equivalent
     (each row has the same 3% inclusion chance) but are not the same rows.
  2. **Every run must use this same file.** The parity gate (ROADMAP, Phase 2)
     compares two implementations, so both must see identical input; that is
     what makes the comparison meaningful even though the sample differs from
     the notebook's own draw.

WHY A PLAIN CSV
---------------
`pyarrow` requires NumPy >= 2.0 and this environment is pinned to 1.26 by
torch 2.3.0+cpu, so parquet is not available without an unrelated upgrade.
The cache is written as CSV: no new dependency, identical schema to the raw
file, and hashed as a whole so a run can prove which input it used.

USAGE
-----
    python make_cache.py                       # full 3% sample
    python make_cache.py --limit 2000000       # first 2M rows, for a dry run
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

DEFAULT_SRC = "data_raw/NF-BoT-IoT-v2.csv"
DEFAULT_OUT = "data_raw/sample_3pct.csv"
DEFAULT_MANIFEST = "data_raw/cache_manifest.json"

CHUNK_ROWS = 100_000


def count_per_class(src: str, limit: int = None) -> dict:
    """Pass 1: how many rows each `Attack` class has. One column only."""
    counts: dict = {}
    rows = 0
    for chunk in pd.read_csv(src, usecols=["Attack"], chunksize=CHUNK_ROWS,
                             nrows=limit):
        for cls, n in chunk["Attack"].value_counts().items():
            counts[cls] = counts.get(cls, 0) + int(n)
        rows += len(chunk)
    return counts, rows


def write_sample(src: str, out: str, keep_prob: dict, seed: int,
                 limit: int = None) -> dict:
    """Pass 2: stream again, keep each row with its class's probability."""
    rng = np.random.default_rng(seed)
    kept: dict = {}
    first = True
    rows = 0

    if os.path.exists(out):
        os.remove(out)

    for chunk in pd.read_csv(src, chunksize=CHUNK_ROWS, nrows=limit):
        probs = chunk["Attack"].map(keep_prob).to_numpy(dtype=np.float64)
        mask = rng.random(len(chunk)) < probs
        selected = chunk[mask]

        if len(selected):
            selected.to_csv(out, mode="a", header=first, index=False)
            first = False
            for cls, n in selected["Attack"].value_counts().items():
                kept[cls] = kept.get(cls, 0) + int(n)
        rows += len(chunk)
    return {"rows_read": rows, "rows_kept": sum(kept.values()),
            "per_class_kept": kept}


def sha256_file(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--frac", type=float, default=0.03,
                    help="fraction to keep per Attack class (paper: 0.03)")
    ap.add_argument("--seed", type=int, default=13,
                    help="sampling seed; 13 matches the notebook's random_state")
    ap.add_argument("--limit", type=int, default=None,
                    help="read only the first N rows (dry run)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[cache] pass 1: counting classes in {args.src} ...")
    counts, total_rows = count_per_class(args.src, limit=args.limit)
    print(f"[cache] {total_rows:,} rows, {len(counts)} classes")
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:<16} {n:>10,}")

    keep_prob = {cls: args.frac for cls in counts}
    print(f"[cache] pass 2: streaming and keeping ~{args.frac:.0%} per class ...")
    result = write_sample(args.src, args.out, keep_prob, args.seed,
                          limit=args.limit)

    digest = sha256_file(args.out)
    size = os.path.getsize(args.out)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {"path": args.src, "rows": total_rows,
                   "sha256": sha256_file(args.src)},
        "output": {"path": args.out, "rows": result["rows_kept"],
                   "bytes": size, "sha256": digest},
        "procedure": {
            "method": "streamed per-row inclusion, probability = frac",
            "frac": args.frac,
            "seed": args.seed,
            "chunk_rows": CHUNK_ROWS,
            "equivalent_to_source": False,
            "note": ("Statistically equivalent to "
                     "groupby('Attack').sample(frac=frac, random_state=13) "
                     "but not the same rows. All runs must consume this file."),
        },
        "class_counts_full": counts,
        "class_counts_sampled": result["per_class_kept"],
        "paper_table5_reference": {
            "Benign": 4051, "DoS": 500195, "DDoS": 549955,
            "Reconnaissance": 78630, "Theft": 73,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[cache] wrote {args.out}: {result['rows_kept']:,} rows, "
          f"{size/1e6:.1f} MB, sha256={digest[:16]}...")
    print(f"[cache] manifest -> {args.manifest}")
    print("[cache] sampled counts vs paper Table 5:")
    for cls, n in sorted(result["per_class_kept"].items(), key=lambda kv: -kv[1]):
        ref = manifest["paper_table5_reference"].get(cls)
        delta = f"{n - ref:+d}" if ref is not None else "n/a"
        print(f"    {cls:<16} {n:>10,}   paper {ref if ref is not None else '-':>8}   delta {delta}")


if __name__ == "__main__":
    main()
