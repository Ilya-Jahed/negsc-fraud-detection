"""
Equality proof for the sparse adjacency (ROADMAP, Phase 1).

The claim this file has to establish: `SparseAdjacency.block(ids)` returns
exactly what the reference's dense expression returns, on graphs that exercise
the two easy-to-miss details -- the transpose and parallel-edge multiplicities.

If this test passes, replacing the 4.3 GB dense matrix changes nothing about
what the model computes. If it fails, the GWD cost is being fed a different
matrix and every downstream number is invalid.
"""

import random

import dgl
import pytest
import torch

from sparse_adj import SparseAdjacency, reference_block, reference_dense_adjacency


def _graph(src, dst, num_nodes):
    return dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)


def test_matches_reference_dense():
    """Block extraction equals `adj[ids].t()[ids]` on a graph with parallel
    edges, self-loops and asymmetric connectivity."""
    src = [0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 0]
    dst = [1, 1, 2, 0, 3, 0, 1, 2, 4, 0, 1, 0]  # 0->1 twice, 2->2 self-loop
    g = _graph(src, dst, num_nodes=5)

    dense = reference_dense_adjacency(g)
    adj = SparseAdjacency.from_dgl(g)

    rng = random.Random(0)
    for _ in range(200):
        k = rng.choice([1, 2, 3, 4])
        ids = [rng.randrange(5) for _ in range(k)]
        expected = reference_block(dense, ids)
        actual = adj.block(ids)
        assert torch.equal(actual, expected), (
            f"ids={ids}\nexpected={expected}\nactual={actual}"
        )


def test_parallel_edges_are_counts_not_flags():
    """coo_matrix sums duplicate coordinates, so 0->1 twice means adj[0][1]==2."""
    g = _graph([0, 0], [1, 1], num_nodes=2)
    adj = SparseAdjacency.from_dgl(g)

    assert adj.counts[(0, 1)] == 2
    block = adj.block([0, 1])
    # transposed: edge 0->1 appears at [row=1][col=0]
    assert block[1, 0].item() == 2.0
    assert block[0, 1].item() == 0.0


def test_transpose_is_preserved():
    """`adj[ids].t()[ids]` transposes; a directed graph must show it."""
    g = _graph([0], [1], num_nodes=2)  # one edge, 0 -> 1, no reverse
    adj = SparseAdjacency.from_dgl(g)

    block = adj.block([0, 1])
    assert block[1, 0].item() == 1.0, "edge 0->1 should sit at [1][0] after .t()"
    assert block[0, 1].item() == 0.0

    dense = reference_dense_adjacency(g)
    assert torch.equal(block, reference_block(dense, [0, 1]))


def test_batch_matches_per_block():
    g = _graph([0, 1, 2, 3], [1, 2, 3, 0], num_nodes=4)
    adj = SparseAdjacency.from_dgl(g)
    subgraphs = [[0, 1], [1, 2], [2, 3], [3, 0]]

    batched = adj.batch(subgraphs)
    assert batched.shape == (4, 2, 2)
    for i, ids in enumerate(subgraphs):
        assert torch.equal(batched[i], adj.block(ids))


def test_sub_sam_shaped_subgraphs():
    """Subgraphs from the real sampler have the centre last and length k."""
    from negsc import sub_sam

    src = [0, 0, 0, 1, 1, 2, 2, 3, 3]
    dst = [1, 2, 3, 0, 2, 0, 3, 0, 1]
    g = _graph(src, dst, num_nodes=4)
    dense = reference_dense_adjacency(g)
    adj = SparseAdjacency.from_dgl(g)

    from collections import defaultdict

    adj_lists = defaultdict(set)
    u, v = g.edges()
    for a, b in zip(u.tolist(), v.tolist()):
        adj_lists[a].add(b)

    torch.manual_seed(0)
    random.seed(0)
    subgraphs = sub_sam(torch.arange(4), adj_lists, k=3)
    assert subgraphs, "sampler produced no subgraphs"

    for ids in subgraphs:
        assert len(ids) == 3
        assert torch.equal(adj.block(ids), reference_block(dense, ids))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
