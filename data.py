"""
data.py -- "Data Flow" / "Graph Representation" panel.

Ports the preprocessing + graph-construction cells from NEGSC.ipynb and
test.ipynb into reusable functions. Logic is kept identical to the source
notebooks; only the wrapping into functions is new.
"""

import math
import random
import socket
import struct

import networkx as nx
import pandas as pd
import torch as th
from dgl import from_networkx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce


# ---------------------------------------------------------------------------
# Loading + basic preprocessing (NEGSC.ipynb cells 2-4, 8)
# ---------------------------------------------------------------------------

def load_raw(path: str, nrows: int = None) -> pd.DataFrame:
    """
    Read the raw NF-BoT-IoT-v2 csv.

    `nrows`, if given, reads only the first N rows -- useful for fast local
    dev/smoke-testing on a huge file. Leave as None for the full dataset
    (matches the source notebook exactly).

    NOTE: the full file is 6.05 GB / 37,763,496 rows and does not fit in
    16.9 GB of RAM alongside pandas' working copy. Use `load_cache()` on the
    pre-sampled file for anything larger than a smoke test (ROADMAP, Phase 1 -- dataset cache).
    """
    return pd.read_csv(path, nrows=nrows)


def load_cache(path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load the pre-sampled cache produced by `make_cache.py`.

    The cache is a plain CSV, identical in schema to the raw file. It exists
    because the raw file cannot be read on this machine, and because every run
    -- reference notebook included -- must consume byte-identical input for
    the parity gate (ROADMAP, Phase 2) to mean anything.

    Returns the frame with the `Attack`/`Label` columns intact; sampling has
    already happened upstream.
    """
    return pd.read_csv(path, nrows=nrows)


def sample_by_attack(data: pd.DataFrame, frac: float = 0.03,
                      random_state: int = 13) -> pd.DataFrame:
    """Stratified downsampling per Attack class, as in NEGSC.ipynb cell 3."""
    return data.groupby(by='Attack').sample(frac=frac, random_state=random_state)


def build_ip_port_ids(data: pd.DataFrame) -> pd.DataFrame:
    """
    Randomize destination IPs, and fold src/dst port into the IP string as
    'ip:port', then drop the separate port columns. Mirrors NEGSC.ipynb
    cell 4.
    """
    data = data.copy()
    data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(
        lambda x: socket.inet_ntoa(
            struct.pack('>I', random.randint(0xac100001, 0xac1f0001))
        )
    )
    data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
    data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
    data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
    data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)
    data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
    data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']
    data.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)
    return data


def encode_labels(data: pd.DataFrame):
    """
    Drop the binary 'Label' column, rename 'Attack' -> 'label', and
    label-encode it. Mirrors NEGSC.ipynb cell 8 (label part only; scaling
    happens later in encode_and_scale_train).

    Returns (data, label_encoder).
    """
    data = data.copy()
    data.drop(columns=['Label'], inplace=True)
    data.rename(columns={"Attack": "label"}, inplace=True)
    le = LabelEncoder()
    le.fit(data.label.values)
    data['label'] = le.transform(data['label'])
    label = data.label
    data.drop(columns=['label'], inplace=True)
    data = pd.concat([data, label], axis=1)
    return data, le


def split_data(data: pd.DataFrame, test_size: float = 0.3,
                random_state: int = 123):
    """
    train_test_split stratified on label. Mirrors NEGSC.ipynb cell 9.
    Note: source passes the full `data` frame as X (label column included);
    kept as-is for fidelity.
    """
    label = data.label
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=test_size, random_state=random_state,
        stratify=label
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Target encoding + scaling (NEGSC.ipynb cells 10-11, test.ipynb cell 16-17)
# ---------------------------------------------------------------------------

TARGET_ENCODE_COLS = ['TCP_FLAGS', 'L7_PROTO', 'PROTOCOL']


def fit_target_encoder(X_train: pd.DataFrame, y_train: pd.Series) -> ce.TargetEncoder:
    """Fit category_encoders.TargetEncoder on train only. NEGSC.ipynb cell 10."""
    encoder = ce.TargetEncoder(cols=TARGET_ENCODE_COLS)
    encoder.fit(X_train, y_train)
    return encoder


def transform_with_encoder(X: pd.DataFrame, encoder: ce.TargetEncoder) -> pd.DataFrame:
    return encoder.transform(X)


def feature_columns(X_train: pd.DataFrame, legacy_order: bool = False) -> list:
    """
    The ordered list of columns that become the edge feature vector 'h'.

    ROADMAP, Phase 1 -- feature-column order; blocker B5. The source does:

        list(set(list(X_train.iloc[:, 2:].columns)) - set(['label']))

    A `set` of strings iterates in an order derived from their hashes, which
    CPython randomises per process unless PYTHONHASHSEED is fixed. The column
    order is *not* cosmetic: each feature dimension gets its own weight in the
    encoder, so a different order is a different model. Measured on the
    authors' own checkpoint, the same weights score between 0.44 and 0.90
    accuracy depending only on this order (see the evidence table in the roadmap appendix).

    Default: CSV column order after the two IP:port id columns -- the same set
    of columns, deterministically ordered.

    `legacy_order=True` reproduces the source expression exactly, for the
    parity gate and for variant A.
    """
    if legacy_order:
        return list(set(list(X_train.iloc[:, 2:].columns)) - set(['label']))
    return [c for c in X_train.columns[2:] if c != 'label']


def fit_scale_and_build_feature_vector(X_train: pd.DataFrame,
                                       legacy_order: bool = False):
    """
    Fit a StandardScaler on all feature columns (excluding the first two
    IP:port id columns and 'label'), then build the 'h' feature-vector
    column. Mirrors NEGSC.ipynb cell 11.

    Returns (X_train, scaler, cols_to_norm). `cols_to_norm` is part of the
    model: persist it with the checkpoint and reuse it at inference.
    """
    X_train = X_train.copy()
    cols_to_norm = feature_columns(X_train, legacy_order=legacy_order)
    scaler = StandardScaler()
    X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
    X_train['h'] = X_train[cols_to_norm].values.tolist()
    return X_train, scaler, cols_to_norm


def scale_and_build_feature_vector(X: pd.DataFrame, scaler: StandardScaler,
                                     cols_to_norm) -> pd.DataFrame:
    """
    Transform (not fit) an already-encoder-transformed frame with a fitted
    scaler, and build the 'h' column. Mirrors NEGSC.ipynb cell 34 /
    test.ipynb cell 17 (X_test path).
    """
    X = X.copy()
    X[cols_to_norm] = scaler.transform(X[cols_to_norm])
    X['h'] = X[cols_to_norm].values.tolist()
    return X


# ---------------------------------------------------------------------------
# Graph construction (NEGSC.ipynb cells 22, 38; test.ipynb cell 17)
# ---------------------------------------------------------------------------

def build_train_graphs(X_train: pd.DataFrame, g_len: int, num: int,
                       max_chunks: int = None, min_nodes: int = None):
    """
    Split X_train into chunks of g_len rows, build a directed multigraph
    per chunk with edge attrs ['h', 'label'], convert to DGL. Drop the last
    graph if it has fewer than `num` nodes. Mirrors NEGSC.ipynb cells
    13/22/26.

    `max_chunks` limits how many chunks are built, for the parity gate and
    smoke runs. The reference builds all of them; leave it None for that.

    Note the chunking is positional (`X_train[i*g_len:(i+1)*g_len]`) on a
    frame whose index was shuffled by train_test_split. That is the source
    behaviour and is kept: it means the chunks are not contiguous row ranges
    of the original file, but arbitrary samples. It does not affect
    correctness -- each chunk is still a valid graph -- only the composition
    of each chunk.
    """
    g_num = math.ceil(len(X_train) / g_len)
    if max_chunks is not None:
        g_num = min(g_num, max_chunks)
    graph = []
    for i in range(g_num):
        chunk = X_train[i * g_len: (i + 1) * g_len]
        G = nx.from_pandas_edgelist(
            chunk, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'],
            create_using=nx.MultiGraph()
        )
        G = G.to_directed()
        G = from_networkx(G, edge_attrs=['h', 'label'])
        graph.append(G)

    threshold = num if min_nodes is None else min_nodes
    if graph and graph[-1].num_nodes() < threshold:
        graph = graph[:-1]
    return graph


def build_full_train_graph(X_train: pd.DataFrame):
    """
    Build a single directed multigraph over the *entire* X_train (used
    later for the final embedding step). Mirrors NEGSC.ipynb cell 38.
    """
    g = nx.from_pandas_edgelist(
        X_train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'],
        create_using=nx.MultiGraph()
    )
    g = g.to_directed()
    g = from_networkx(g, edge_attrs=['h', 'label'])
    g.ndata['h'] = th.ones(g.num_nodes(), g.edata['h'].shape[1])
    return g


def build_test_graph(X_test: pd.DataFrame):
    """
    Build the test directed multigraph with edge attrs ['h', 'label'], and
    initialize node feature 'feature' as ones. Mirrors NEGSC.ipynb cell 34 /
    test.ipynb cell 17.
    """
    G_test = nx.from_pandas_edgelist(
        X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'],
        create_using=nx.MultiGraph()
    )
    G_test = G_test.to_directed()
    G_test = from_networkx(G_test, edge_attrs=['h', 'label'])
    G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G_test.edata['h'].shape[1])
    return G_test