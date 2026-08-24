"""
main.py -- end-to-end training entrypoint (mirrors NEGSC.ipynb top to bottom).

This file is being built up phase by phase alongside the modules. Right now
only the "Data Flow" / "Graph Representation" phase (data.py) is wired in.
Later phases (NEGAT encoder, NEGSC contrastive training, Predict) are marked
as TODO and will be filled in as negat.py / negsc.py / predict.py land.
"""

import gc
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import data
import negat
import negsc
import predict

# ---------------------------------------------------------------------------
# Config (kept inline per project decision -- no separate config.py)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data_raw/NF-BoT-IoT-v2.csv"

# Set to an int (e.g. 200_000) for fast local dev/smoke-testing on a subset
# of the raw csv. Leave as None to use the full dataset (matches the paper).
NROWS_LIMIT = None

SAMPLE_FRAC = 0.03
SAMPLE_RANDOM_STATE = 13

SPLIT_TEST_SIZE = 0.3
SPLIT_RANDOM_STATE = 123

learning_rate = 0.0001
num = 20000          # nodes sampled per training epoch (sub_sam batch size)
k1 = 3                # subgraph size (center + k1-1 neighbors)
tau = 5                # temperature for WD/GWD contrastive losses
num_epochs = 300        # epochs per graph chunk in the NEGSC training loop
weight_decay = 0.00001
g_len = 35000             # rows per training graph chunk
num_heads = 3               # attention heads in NEGAT


def main():
    # -----------------------------------------------------------------
    # Phase 2: Data Flow / Graph Representation  (data.py)
    # -----------------------------------------------------------------
    print(f"[data] loading raw csv from {DATA_PATH} "
          f"(nrows={NROWS_LIMIT or 'all'})...")
    raw = data.load_raw(DATA_PATH, nrows=NROWS_LIMIT)
    print(f"[data] loaded {len(raw)} rows")

    raw = data.sample_by_attack(raw, frac=SAMPLE_FRAC,
                                 random_state=SAMPLE_RANDOM_STATE)
    print(f"[data] sampled down to {len(raw)} rows "
          f"(frac={SAMPLE_FRAC} per attack class)")

    raw = data.build_ip_port_ids(raw)
    print("[data] built ip:port node ids")

    raw, label_encoder = data.encode_labels(raw)
    print(f"[data] encoded labels: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = data.split_data(
        raw, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    print(f"[data] split: X_train={len(X_train)} rows, X_test={len(X_test)} rows")

    target_encoder = data.fit_target_encoder(X_train, y_train)
    X_train = data.transform_with_encoder(X_train, target_encoder)
    print("[data] target-encoded categorical columns")

    X_train, scaler, cols_to_norm = data.fit_scale_and_build_feature_vector(X_train)
    print("[data] scaled features and built 'h' feature vectors")

    print("[data] building training graphs (this can take a while on large "
          "chunks)...")
    graph = data.build_train_graphs(X_train, g_len=g_len, num=num)

    # X_test is encoded/scaled later in the notebook (right before the
    # downstream classifier stage), kept here for reference:
    # X_test = data.transform_with_encoder(X_test, target_encoder)
    # X_test = data.scale_and_build_feature_vector(X_test, scaler, cols_to_norm)
    # G_test = data.build_test_graph(X_test)

    print(f"[data] built {len(graph)} training graph chunks on {DEVICE}")
    for i, g in enumerate(graph):
        print(f"  graph[{i}]: nodes={g.num_nodes()} edges={g.num_edges()} "
              f"edge_feat_dim={g.edata['h'].shape[1]}")

    # -----------------------------------------------------------------
    # Phase 3: NEGAT Encoder  (negat.py)
    # -----------------------------------------------------------------
    # Source (NEGSC.ipynb cell 28) reads these dims off the last built
    # training graph chunk; since all chunks share the same edge feature
    # width, using graph[0] here is equivalent.
    n_dim = e_dim = out_dim = graph[0].edata['h'].shape[1]
    activation = F.relu

    Encoder = negat.GAT(n_dim, e_dim, out_dim, num_heads).to(DEVICE)
    print(f"[negat] built Encoder (NEGAT GAT) with n_dim={n_dim}, "
          f"e_dim={e_dim}, num_heads={num_heads}")

    # -----------------------------------------------------------------
    # Phase 4: NEGSC contrastive training  (negsc.py)
    # -----------------------------------------------------------------
    gene = negat.GAT(n_dim, e_dim, out_dim, num_heads)

    # !!! FIDELITY WARNING !!!
    # The source notebook (NEGSC.ipynb cell 29) constructs Model and the
    # optimizer WITHOUT passing tau, lr, or weight_decay, even though those
    # are defined above (tau=5, learning_rate=0.0001, weight_decay=0.00001).
    # This means the source actually trains with Model's default tau=0.5
    # and Adam's default lr=0.001 / weight_decay=0 -- the config values
    # above are effectively DEAD in the original code. This is very likely
    # an oversight in the source, but per project decision we replicate it
    # exactly rather than silently "fixing" it. If you want the config
    # values to actually take effect, change the two lines below to:
    #   model = negsc.Model(Encoder, gene, tau=tau).to(DEVICE)
    #   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    #                                 weight_decay=weight_decay)
    model = negsc.Model(Encoder, gene).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    print(f"[negsc] built Model (tau={model.tau}) and optimizer "
          f"(lr={optimizer.defaults['lr']}, "
          f"weight_decay={optimizer.defaults['weight_decay']}) "
          f"-- NOTE: these ignore the tau/learning_rate/weight_decay "
          f"config values above, matching the source notebook exactly.")

    # Source cell 30 also defines node_neighbor={}, best=1e9, best_t=0,
    # bestacc=0 here -- none of these are referenced anywhere in the
    # visible training loop, so they are vestigial/unused and omitted.

    for gi, g in enumerate(graph):
        print(f"[negsc] training on graph {gi + 1}/{len(graph)} "
              f"(nodes={g.num_nodes()}, edges={g.num_edges()})...")

        g.ndata['h'] = torch.ones(g.num_nodes(), g.edata['h'].shape[1])
        adj = sp.coo_matrix(
            (np.ones(g.num_edges()), (g.edges()[0], g.edges()[1])),
            shape=(g.num_nodes(), g.num_nodes()), dtype=np.float32
        ).toarray()
        adj = torch.from_numpy(adj).to(DEVICE)
        adj_lists = defaultdict(set)
        g1 = g
        for x in range(g1.num_edges()):
            adj_lists[g1.edges()[0][x].item()].add(g1.edges()[1][x].item())
        g = g.to(DEVICE)
        node_feats = g.ndata['h']
        edge_feats = g.edata['h']

        for epoch in range(1, num_epochs + 1):
            nodes_batch = torch.randint(0, g.num_nodes(), (num,))
            node_neighbor_cen = negsc.sub_sam(nodes_batch, adj_lists, k1)
            loss = negsc.train(model, g, node_feats, edge_feats, adj,
                                node_neighbor_cen, optimizer)

        print(f"[negsc] graph {gi + 1}/{len(graph)} done, final loss={loss:.4f}")

        del adj, g
        gc.collect()
        if DEVICE.type == "cuda":
            for _ in range(5):
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Phase 5: Predict  (predict.py)
    # -----------------------------------------------------------------
    print("[predict] preparing test graph and full training graph...")
    X_test = data.transform_with_encoder(X_test, target_encoder)
    X_test = data.scale_and_build_feature_vector(X_test, scaler, cols_to_norm)
    G_test = data.build_test_graph(X_test)
    G_test = G_test.to(DEVICE)

    # Source rebuilds a fresh, unchunked graph over the entire X_train here
    # (NEGSC.ipynb cells 38-41) -- this is a different graph object from
    # any of the per-chunk training graphs used in Phase 4.
    full_train_graph = data.build_full_train_graph(X_train)
    full_train_graph = full_train_graph.to(DEVICE)

    print("[predict] embedding train/test graphs with trained NEGSC encoder...")
    train_embs, test_embs, train_lbls, test_lbls = predict.embed_graphs(
        model, full_train_graph, G_test
    )

    n_classes = len(label_encoder.classes_)
    print(f"[predict] training downstream classifier for {n_classes} classes "
          f"(10000 steps, this will take a while)...")
    log = predict.train_classifier(
        train_embs, train_lbls, in_features=n_dim, n_classes=n_classes,
        g_for_predictor=full_train_graph, num_steps=10000
    )

    preds = predict.predict(log, G_test, test_embs)
    test_lbls_decoded, preds_decoded = predict.decode_labels(
        label_encoder, test_lbls, preds
    )

    print("[predict] evaluation:")
    predict.evaluate(test_lbls_decoded, preds_decoded,
                      save_path="outputs/confusion_matrix.png")

    torch.save(model, "checkpoints/model.pth")
    torch.save(log, "checkpoints/log.pth")
    print("[predict] saved checkpoints/model.pth and checkpoints/log.pth")


if __name__ == "__main__":
    main()