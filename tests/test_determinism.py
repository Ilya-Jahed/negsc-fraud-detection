"""
Determinism tests (ROADMAP, Phase 1 -- seeding; exit criterion: "two seeded runs identical").

Two separate claims:

1. **In-process.** Given `set_seed(s)`, two runs of `main.main()` produce the
   same loss sequence. This is what fails whenever a code path draws from an
   unseeded RNG -- e.g. if `build_ip_port_ids` were called before `set_seed`,
   or if `sub_sam` used its own `random.Random()`.

2. **Cross-process.** Given the same seed *and* PYTHONHASHSEED, two separate
   interpreters produce the same loss. This is the claim that makes a
   checkpoint reusable and a result citable. It is the one the reference
   notebook fails, and the reason its own published checkpoints are unusable
   (see the roadmap appendix).

The second test costs a couple of interpreter startups; it is the price of
testing the thing that actually breaks.
"""

import csv
import os
import subprocess
import sys

import pytest

import main
from data import feature_columns
from seed import set_seed

ARGS = [
    "--sample-frac", "1.0",
    "--g-len", "1000",
    "--chunks", "1",
    "--epochs", "3",
    "--num", "200",
    "--min-nodes", "50",
    "--k1", "3",
    "--seed", "11",
    "--log-every", "1",
    "--skip-predict",
]


def _losses(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [float(r["loss"]) for r in csv.DictReader(f)]


def test_same_seed_same_loss(fixture_csv, tmp_path):
    """Same seed, same process -> identical loss sequence."""
    main.main(ARGS + ["--data", str(fixture_csv),
                      "--run-id", "det_a", "--out-dir", str(tmp_path)])
    main.main(ARGS + ["--data", str(fixture_csv),
                      "--run-id", "det_b", "--out-dir", str(tmp_path)])

    a = _losses(tmp_path / "loss_det_a.csv")
    b = _losses(tmp_path / "loss_det_b.csv")
    assert len(a) == len(b) == 3
    assert a == b, f"seeded runs diverged:\n  {a}\n  {b}"


def test_different_seed_different_loss(fixture_csv, tmp_path):
    """Seeding must actually control the run -- otherwise the first test is
    passing for the wrong reason (e.g. a constant loss)."""
    other = [x if x != "11" else "99" for x in ARGS]
    main.main(ARGS + ["--data", str(fixture_csv),
                      "--run-id", "det_seed11", "--out-dir", str(tmp_path)])
    main.main(other + ["--data", str(fixture_csv),
                       "--run-id", "det_seed99", "--out-dir", str(tmp_path)])

    a = _losses(tmp_path / "loss_det_seed11.csv")
    b = _losses(tmp_path / "loss_det_seed99.csv")
    assert a != b, "different seeds produced identical losses -- seed is not reaching the model"


def test_cross_process_reproducible(fixture_csv, tmp_path):
    """Same seed and PYTHONHASHSEED in two fresh interpreters -> same loss."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONHASHSEED="0")

    for run_id in ("xproc_a", "xproc_b"):
        cmd = [sys.executable, "main.py"] + ARGS + [
            "--data", str(fixture_csv),
            "--run-id", run_id,
            "--out-dir", str(tmp_path),
        ]
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"{run_id} failed:\n{proc.stdout}\n{proc.stderr}"

    a = _losses(tmp_path / "loss_xproc_a.csv")
    b = _losses(tmp_path / "loss_xproc_b.csv")
    assert a == b, f"cross-process runs diverged:\n  {a}\n  {b}"


def test_feature_order_is_deterministic():
    """`feature_columns` default order is stable and complete; the legacy
    option returns the same *set* of columns."""
    import pandas as pd

    frame = pd.DataFrame(
        {c: [0.0] for c in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "b", "a", "c", "label"]}
    )
    deterministic = feature_columns(frame)
    assert deterministic == ["b", "a", "c"], "must follow CSV order, not hash order"

    legacy = feature_columns(frame, legacy_order=True)
    assert set(legacy) == set(deterministic), "legacy must not add or drop columns"