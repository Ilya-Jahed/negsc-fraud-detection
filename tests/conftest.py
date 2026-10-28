"""Shared pytest configuration and fixtures.

The fixture CSV reproduces the 45-column reference schema. It deliberately
uses a handful of repeated source IPs: `main()` randomises every destination
IP before building the graph (source behaviour), so with unique source IPs the
graph degenerates into a matching and `sub_sam` finds no centre node with
degree >= k1. Repeated sources keep the fixture graph sampleable.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FEATURE_COLUMNS = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

ATTACKS = ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"]
CATEGORICAL = ("PROTOCOL", "L7_PROTO", "TCP_FLAGS")


def make_fixture(path, n_rows=600, n_ips=10, n_ports=6, seed=0):
    """Write a 45-column CSV with the reference schema.

    Source identity is `IP:port` after `build_ip_port_ids`, so a fixture with
    random source ports produces one distinct source node per flow and every
    node ends up with degree 1 -- `sub_sam` then finds no centre with
    degree >= k1 and returns nothing. Real NetFlow has servers reusing the
    same (ip, port), so the fixture does too: `n_ips * n_ports` combinations
    are cycled, giving each source ~n_rows / (n_ips*n_ports) edges.
    """
    rng = np.random.default_rng(seed)
    n_combos = n_ips * n_ports
    src_ips = [f"192.168.10.{i % n_ips}" for i in range(n_combos)]
    src_ports = [1024 + 17 * (i // n_ips) for i in range(n_combos)]

    frame = pd.DataFrame({
        "IPV4_SRC_ADDR": [src_ips[i % n_combos] for i in range(n_rows)],
        "L4_SRC_PORT": [src_ports[i % n_combos] for i in range(n_rows)],
        "IPV4_DST_ADDR": [f"10.0.{i % 7}.{i % 53}" for i in range(n_rows)],
        "L4_DST_PORT": rng.integers(1, 1024, n_rows),
    })
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL:
            frame[col] = rng.integers(0, 5, n_rows)
        else:
            frame[col] = rng.normal(100, 25, n_rows)
    frame["Attack"] = [ATTACKS[i % len(ATTACKS)] for i in range(n_rows)]
    frame["Label"] = (frame["Attack"] != "Benign").astype(int)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def fixture_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "fixture.csv"
    return make_fixture(path)