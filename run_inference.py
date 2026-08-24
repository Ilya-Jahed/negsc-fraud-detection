"""
run_inference.py -- inference entrypoint (mirrors test.ipynb).

Loads a saved NEGSC model + downstream classifier checkpoint and runs
inference on a held-out test split, without any training.

Fidelity notes vs. NEGSC.ipynb / main.py:
  - test.ipynb does NOT downsample with groupby(Attack).sample(frac=...);
    it loads the csv as-is. This script matches that (no sample_by_attack
    call).
  - test.ipynb does NOT save/load the fitted LabelEncoder / TargetEncoder /
    StandardScaler from the training run. It re-fits fresh instances on
    X_train (using the same split random_state), assuming this reproduces
    equivalent encodings. This is fragile (relies on determinism across
    runs/environments) but is exactly what the source does -- replicated
    here rather than "fixed" by saving/loading the original fitted
    encoders, per project fidelity rules.
  - No full_train_graph / train_embs are built here -- the downstream
    classifier ("log") is already trained and loaded from checkpoint, so
    only the test graph needs to be embedded.
"""

import torch

import data
import negat  # noqa: F401  (needed so torch.load can unpickle negat.GAT)
import negsc  # noqa: F401  (needed so torch.load can unpickle negsc.Model / MLPPredictor)
import predict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data_raw/NF-BoT-IoT-v2.csv"

# Set to an int for fast local dev/smoke-testing on a subset of the raw csv.
# Leave as None to use the full dataset (matches test.ipynb).
NROWS_LIMIT = None

SPLIT_TEST_SIZE = 0.3
SPLIT_RANDOM_STATE = 123

MODEL_CKPT_PATH = "checkpoints/model.pth"
LOG_CKPT_PATH = "checkpoints/log.pth"


def main():
    # -----------------------------------------------------------------
    # Data (test.ipynb cells 2-9) -- no sample_by_attack call
    # -----------------------------------------------------------------
    print(f"[data] loading raw csv from {DATA_PATH} "
          f"(nrows={NROWS_LIMIT or 'all'})...")
    raw = data.load_raw(DATA_PATH, nrows=NROWS_LIMIT)
    print(f"[data] loaded {len(raw)} rows")

    raw = data.build_ip_port_ids(raw)
    print("[data] built ip:port node ids")

    raw, label_encoder = data.encode_labels(raw)
    print(f"[data] encoded labels: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = data.split_data(
        raw, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    print(f"[data] split: X_train={len(X_train)} rows, X_test={len(X_test)} rows")

    # !!! FIDELITY WARNING !!!
    # target_encoder and scaler are re-fit fresh here on X_train, not
    # loaded from the original training run. This matches test.ipynb
    # exactly, but means results depend on this split reproducing the
    # same X_train as whatever produced the loaded checkpoints.
    target_encoder = data.fit_target_encoder(X_train, y_train)
    X_train = data.transform_with_encoder(X_train, target_encoder)
    X_train, scaler, cols_to_norm = data.fit_scale_and_build_feature_vector(X_train)
    print("[data] re-fit target encoder and scaler on X_train "
          "(not loaded from a saved artifact, matching test.ipynb)")

    X_test = data.transform_with_encoder(X_test, target_encoder)
    X_test = data.scale_and_build_feature_vector(X_test, scaler, cols_to_norm)
    G_test = data.build_test_graph(X_test)
    G_test = G_test.to(DEVICE)
    print(f"[data] built test graph: nodes={G_test.num_nodes()} "
          f"edges={G_test.num_edges()}")

    # -----------------------------------------------------------------
    # Load checkpoints (test.ipynb cell: torch.load(...))
    # -----------------------------------------------------------------
    print(f"[inference] loading checkpoints from {MODEL_CKPT_PATH} "
          f"and {LOG_CKPT_PATH}...")
    model = torch.load(MODEL_CKPT_PATH, map_location=DEVICE, weights_only=False)
    log = torch.load(LOG_CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    log.eval()

    # -----------------------------------------------------------------
    # Embed + predict (test.ipynb cells: embed, argmax, decode)
    # -----------------------------------------------------------------
    print("[inference] embedding test graph...")
    test_embs = model.embed(G_test, G_test.ndata['feature'], G_test.edata['h'])
    test_lbls = G_test.edata['label']

    preds = predict.predict(log, G_test, test_embs)
    test_lbls_decoded, preds_decoded = predict.decode_labels(
        label_encoder, test_lbls, preds
    )

    print("[inference] evaluation:")
    predict.evaluate(test_lbls_decoded, preds_decoded,
                      save_path="outputs/confusion_matrix_inference.png")


if __name__ == "__main__":
    main()