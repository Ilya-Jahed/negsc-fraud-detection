"""
main.py -- end-to-end training entrypoint (mirrors NEGSC.ipynb top to bottom).

Phase 1 of the roadmap changes only the harness around the model:

  * a CLI, so smoke runs and the parity gate are a flag rather than an edit
  * `set_seed` at the top, and a warning when PYTHONHASHSEED is unpinned
  * `--legacy-order` to reproduce the source's hash-ordered feature columns
  * `--dense-adj` to reproduce the source's 4.3 GB dense adjacency
  * the O(E) sparse adjacency by default (sparse_adj.py)
  * per-epoch loss logging and a run manifest, so a number can be traced back
    to the exact configuration that produced it

The model, loss and optimiser are untouched: variant A still constructs
`negsc.Model(Encoder, gene)` and `Adam(model.parameters())` with no arguments,
so the dead `tau`/`learning_rate`/`weight_decay` constants below stay dead,
exactly as in the reference (deviation V1: the constants are never passed
in). Variant B, which wires them up, is
Phase 4 and is not implemented here.
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict

# Headless: predict.evaluate() calls plt.show(); without this it blocks on a
# machine with no display.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import data
import negat
import negsc
import predict
from device import describe as describe_device, get_device
from seed import describe, set_seed, warn_if_unpinned
from sparse_adj import SparseAdjacency

DEVICE = get_device()

# Reference defaults (NEGSC.ipynb cells 9, 12, 19, 29). Kept at the top so a
# run manifest can record what was actually used.
DEFAULTS = dict(
    # The cache from make_cache.py is ALREADY sampled 3% per class; there is
    # deliberately no sampling default here (see --sample-frac).
    data="data_raw/sample_3pct.csv",
    test_size=0.3,
    split_random_state=123,
    g_len=35000,
    num=20000,
    k1=3,
    epochs=300,
    seed=42,
    # Never passed to the model in variant A -- deviation V1. See the module
    # docstring.
    tau=5,
    learning_rate=0.0001,
    weight_decay=0.00001,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="NEGSC training pipeline (variant A: reference-faithful).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", default=DEFAULTS["data"],
                    help="pre-sampled cache (make_cache.py) or a raw csv")
    ap.add_argument("--rows", type=int, default=None,
                    help="read only the first N rows of --data (smoke runs)")
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="per-Attack-class downsampling fraction. Defaults to "
                         "1.0: the cache from make_cache.py is already sampled "
                         "3%% per class. Set this only when --data is a raw csv.")
    ap.add_argument("--test-size", type=float, default=DEFAULTS["test_size"])
    ap.add_argument("--g-len", type=int, default=DEFAULTS["g_len"],
                    help="rows per training graph chunk")
    ap.add_argument("--chunks", type=int, default=None,
                    help="limit the number of chunks built (parity gate, smoke)")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--num", type=int, default=DEFAULTS["num"],
                    help="nodes sampled per epoch")
    ap.add_argument("--min-nodes", type=int, default=None,
                    help="min node count for keeping the final chunk. The "
                         "source reuses --num here; that coupling is kept as "
                         "the default but can be broken for small fixtures")
    ap.add_argument("--k1", type=int, default=DEFAULTS["k1"],
                    help="subgraph size: 1 centre + k1-1 neighbours")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--variant", choices=["A"], default="A",
                    help="A = reference-faithful. B (paper-faithful) is Phase 4.")
    ap.add_argument("--legacy-order", action="store_true",
                    help="use the source's set()-ordered feature columns "
                         "(requires PYTHONHASHSEED to be pinned for parity)")
    ap.add_argument("--dense-adj", action="store_true",
                    help="use the source's dense [N,N] adjacency (4.3 GB per "
                         "chunk). Default is the O(E) sparse path.")
    ap.add_argument("--skip-predict", action="store_true",
                    help="train only; skip the downstream classifier")
    ap.add_argument("--predict-steps", type=int, default=10000,
                    help="steps for the downstream classifier (source: 10000)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--log-every", type=int, default=0,
                    help="write a loss row every N epochs (0 = last epoch only)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def build_adjacency(g, use_dense: bool):
    """Return the adjacency in the form `negsc.Model._subgraph_adj` expects."""
    if use_dense:
        src, dst = g.edges()
        dense = sp.coo_matrix(
            (np.ones(g.num_edges()), (src.numpy(), dst.numpy())),
            shape=(g.num_nodes(), g.num_nodes()), dtype=np.float32,
        ).toarray()
        return torch.from_numpy(dense).to(DEVICE)
    return SparseAdjacency.from_dgl(g)


def adjacency_lists(g) -> dict:
    adj_lists = defaultdict(set)
    u, v = g.edges()
    for a, b in zip(u.tolist(), v.tolist()):
        adj_lists[a].add(b)
    return adj_lists


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()

    set_seed(args.seed)
    warn_if_unpinned()

    # Never re-sample by default. `--data` is normally the cache built by
    # make_cache.py, which already holds the 3% per-class sample; applying 3%
    # again would leave 0.09% of the dataset (33,934 rows instead of
    # 1,131,139). Sampling is opt-in via --sample-frac, and only meaningful
    # when --data points at a raw, unsampled CSV.
    sample_frac = 1.0 if args.sample_frac is None else args.sample_frac

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[run] id={run_id} variant={args.variant} device={DEVICE} "
          f"seed={args.seed} legacy_order={args.legacy_order} "
          f"dense_adj={args.dense_adj}")

    # -- data ------------------------------------------------------------
    print(f"[data] loading {args.data} (rows={args.rows or 'all'}) ...")
    raw = data.load_cache(args.data, nrows=args.rows)
    print(f"[data] loaded {len(raw):,} rows")

    if sample_frac < 1.0:
        if "sample" in os.path.basename(args.data).lower():
            print(f"[data] WARNING: --sample-frac {sample_frac} applied to "
                  f"{args.data}, which looks like an already-sampled cache. "
                  f"Row count {len(raw):,} -> ~{int(len(raw) * sample_frac):,}. "
                  f"Omit --sample-frac to use the cache as-is.")
        raw = data.sample_by_attack(raw, frac=sample_frac,
                                    random_state=DEFAULTS["split_random_state"])
        print(f"[data] sampled to {len(raw):,} rows (frac={sample_frac})")

    raw = data.build_ip_port_ids(raw)
    raw, label_encoder = data.encode_labels(raw)
    print(f"[data] classes: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = data.split_data(
        raw, test_size=args.test_size, random_state=DEFAULTS["split_random_state"])
    print(f"[data] split: train={len(X_train):,} test={len(X_test):,}")

    target_encoder = data.fit_target_encoder(X_train, y_train)
    X_train = data.transform_with_encoder(X_train, target_encoder)
    X_train, scaler, cols_to_norm = data.fit_scale_and_build_feature_vector(
        X_train, legacy_order=args.legacy_order)
    print(f"[data] feature columns: {len(cols_to_norm)} "
          f"({'legacy set() order' if args.legacy_order else 'deterministic'})")

    graph = data.build_train_graphs(X_train, g_len=args.g_len, num=args.num,
                                    max_chunks=args.chunks,
                                    min_nodes=args.min_nodes)
    if not graph:
        raise SystemExit(
            "[data] no training chunks survived. The last chunk is dropped when "
            "it has fewer than --num nodes (source behaviour); either raise "
            "--rows or lower --num.")
    for i, g in enumerate(graph):
        print(f"  chunk[{i}] nodes={g.num_nodes():,} edges={g.num_edges():,} "
              f"edge_feat_dim={g.edata['h'].shape[1]}")

    # -- model -----------------------------------------------------------
    n_dim = e_dim = out_dim = graph[0].edata["h"].shape[1]
    Encoder = negat.GAT(n_dim, e_dim, out_dim, num_heads=3).to(DEVICE)
    gene = negat.GAT(n_dim, e_dim, out_dim, num_heads=3)

    # Variant A: no tau / lr / weight_decay passed -- the reference's dead
    # constants stay dead -- deviation V1.
    model = negsc.Model(Encoder, gene).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    print(f"[negsc] model tau={model.tau} "
          f"lr={optimizer.defaults['lr']} "
          f"wd={optimizer.defaults['weight_decay']} (reference defaults)")

    # -- train -----------------------------------------------------------
    loss_log = []
    for gi, g in enumerate(graph):
        print(f"[negsc] chunk {gi + 1}/{len(graph)}: "
              f"nodes={g.num_nodes():,} edges={g.num_edges():,}")
        g.ndata["h"] = torch.ones(g.num_nodes(), g.edata["h"].shape[1])
        adj = build_adjacency(g, use_dense=args.dense_adj)
        adj_lists = adjacency_lists(g)
        g = g.to(DEVICE)
        node_feats = g.ndata["h"]
        edge_feats = g.edata["h"]

        for epoch in range(1, args.epochs + 1):
            nodes_batch = torch.randint(0, g.num_nodes(), (args.num,))
            subgraphs = negsc.sub_sam(nodes_batch, adj_lists, args.k1)
            loss = negsc.train(model, g, node_feats, edge_feats, adj,
                               subgraphs, optimizer)
            last = epoch == args.epochs
            if last or (args.log_every and epoch % args.log_every == 0):
                loss_log.append(dict(run_id=run_id, chunk=gi, epoch=epoch,
                                     subgraphs=len(subgraphs), loss=loss))
                print(f"    epoch {epoch:>4}/{args.epochs}  "
                      f"subgraphs={len(subgraphs):,}  loss={loss:.6f}")

        del adj, g
        gc.collect()
        if DEVICE.type == "cuda":
            for _ in range(5):
                torch.cuda.empty_cache()

    # -- predict ---------------------------------------------------------
    metrics = {}
    if not args.skip_predict:
        print("[predict] preparing test graph and full training graph ...")
        X_test = data.transform_with_encoder(X_test, target_encoder)
        X_test = data.scale_and_build_feature_vector(X_test, scaler, cols_to_norm)
        G_test = data.build_test_graph(X_test).to(DEVICE)
        full_train_graph = data.build_full_train_graph(X_train).to(DEVICE)

        train_embs, test_embs, train_lbls, test_lbls = predict.embed_graphs(
            model, full_train_graph, G_test)

        n_classes = len(label_encoder.classes_)
        log = predict.train_classifier(
            train_embs, train_lbls, in_features=n_dim, n_classes=n_classes,
            g_for_predictor=full_train_graph, num_steps=args.predict_steps)

        preds = predict.predict(log, G_test, test_embs)
        test_lbls_decoded, preds_decoded = predict.decode_labels(
            label_encoder, test_lbls, preds)

        cm_path = os.path.join(args.out_dir, f"confusion_matrix_{run_id}.png")
        metrics = predict.evaluate(test_lbls_decoded, preds_decoded,
                                   save_path=cm_path)

        torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{run_id}.pt"))
        torch.save(log.state_dict(), os.path.join(args.out_dir, f"log_{run_id}.pt"))

    # -- manifest --------------------------------------------------------
    elapsed = round(time.time() - t_start, 1)
    manifest = dict(
        run_id=run_id,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=git_sha(),
        variant=args.variant,
        argv=sys.argv[1:] if argv is None else list(argv),
        config=dict(
            data=args.data, rows=args.rows, sample_frac=sample_frac,
            test_size=args.test_size, g_len=args.g_len, epochs=args.epochs,
            num=args.num, k1=args.k1, seed=args.seed,
            legacy_order=args.legacy_order, dense_adj=args.dense_adj,
            tau=model.tau, lr=optimizer.defaults["lr"],
            weight_decay=optimizer.defaults["weight_decay"],
        ),
        feature_columns=cols_to_norm,
        classes=list(label_encoder.classes_),
        env=dict(python=platform.python_version(), **describe(),
                 **describe_device()),
        chunks=[dict(nodes=g.num_nodes(), edges=g.num_edges()) for g in graph],
        metrics=metrics,
        elapsed_sec=elapsed,
    )
    manifest_path = os.path.join(args.out_dir, f"run_{run_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if loss_log:
        import csv
        log_path = os.path.join(args.out_dir, f"loss_{run_id}.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(loss_log[0]))
            w.writeheader()
            w.writerows(loss_log)

    print(f"[run] done in {elapsed}s -> {manifest_path}")


if __name__ == "__main__":
    main()
