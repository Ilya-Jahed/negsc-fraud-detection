"""
End-to-end smoke test (ROADMAP, Phase 1 -- smoke test; milestone M1).

Runs `main.main()` through every stage of the reference pipeline -- load, IP
randomisation, label encoding, split, target encoding, scaling, graph build,
NEGSC training, embedding, downstream classifier, metrics -- on a fixture
small enough to finish in seconds.

The point is not accuracy (the fixture is noise). The point is that no stage is
unreachable, and that the run leaves behind the artifacts later steps consume:
loss CSV, run manifest, checkpoints.
"""

import json

import pytest

import main
from conftest import ATTACKS, FEATURE_COLUMNS

BASE = [
    "--sample-frac", "1.0",
    "--g-len", "1000",
    "--chunks", "1",
    "--epochs", "2",
    "--num", "200",
    "--min-nodes", "50",
    "--k1", "3",
    "--seed", "7",
]


def test_end_to_end(fixture_csv, tmp_path):
    """The whole pipeline runs and leaves its artifacts behind."""
    run_id = "smoke"
    main.main(BASE + [
        "--data", str(fixture_csv),
        "--predict-steps", "5",
        "--run-id", run_id,
        "--out-dir", str(tmp_path),
    ])

    manifest_path = tmp_path / f"run_{run_id}.json"
    assert manifest_path.exists(), "run manifest not written"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["config"]["seed"] == 7
    assert manifest["config"]["k1"] == 3
    assert manifest["chunks"], "no graph chunk was built"
    assert manifest["classes"], "label encoder saw no classes"
    assert len(manifest["feature_columns"]) == len(FEATURE_COLUMNS)
    assert set(manifest["feature_columns"]) == set(FEATURE_COLUMNS)

    metrics = manifest["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "macro_f1" in metrics and "weighted_f1" in metrics
    assert set(metrics["per_class"]) == set(ATTACKS)

    assert (tmp_path / f"loss_{run_id}.csv").exists()
    assert (tmp_path / f"model_{run_id}.pt").exists()
    assert (tmp_path / f"log_{run_id}.pt").exists()
    assert (tmp_path / f"confusion_matrix_{run_id}.png").exists()


def test_rejects_empty_graph(fixture_csv, tmp_path):
    """The source drops the final chunk when it has fewer nodes than --num.
    If every chunk is dropped, main must stop loudly rather than train on
    nothing."""
    with pytest.raises(SystemExit):
        main.main(BASE + [
            "--data", str(fixture_csv),
            "--min-nodes", "100000",   # larger than the whole fixture
            "--skip-predict",
            "--run-id", "empty",
            "--out-dir", str(tmp_path),
        ])


def test_dense_and_sparse_adjacency_agree(fixture_csv, tmp_path):
    """`--dense-adj` (the reference's 4.3 GB path) and the default sparse path
    must train identically. Same seed, same data, only the adjacency differs."""
    import csv

    def run(tag, extra):
        main.main(BASE + ["--data", str(fixture_csv), "--log-every", "1",
                          "--skip-predict", "--run-id", tag,
                          "--out-dir", str(tmp_path)] + extra)
        with open(tmp_path / f"loss_{tag}.csv", newline="", encoding="utf-8") as f:
            return [float(r["loss"]) for r in csv.DictReader(f)]

    sparse = run("adj_sparse", [])
    dense = run("adj_dense", ["--dense-adj"])

    assert len(sparse) == len(dense) == 2
    for a, b in zip(sparse, dense):
        assert a == pytest.approx(b, rel=0, abs=1e-6), (sparse, dense)