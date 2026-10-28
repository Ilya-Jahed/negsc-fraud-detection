"""
sparse_adj.py -- O(E) replacement for the reference's dense [N, N] adjacency
(ROADMAP, Phase 1 -- sparse adjacency; see also the memory
constraint in the roadmap's state section).

WHY THIS EXISTS
---------------
`main.py` in the reference builds, once per training chunk:

    adj = sp.coo_matrix((np.ones(g.num_edges()), (g.edges()[0], g.edges()[1])),
                        shape=(g.num_nodes(), g.num_nodes()), dtype=np.float32).toarray()

A 35k-flow chunk has ~32,800 nodes, so that dense matrix is 4.3 GB -- resident
on CPU and again on the GPU. It fits the paper's 24 GB card and OOMs a 6 GB one.

The matrix is read in exactly one place:

    sub_adj:  subg1_adj[i] = adj[sub_g1[i]].t()[sub_g1[i]]

i.e. a k x k block per subgraph, where k = k1 = 3. Nothing needs the other
~1.07e9 entries. This module computes those blocks directly.

EXACTNESS -- TWO DETAILS THAT ARE EASY TO GET WRONG
---------------------------------------------------
1. **The transpose.** The reference reads `adj[ids].t()[ids]`, which is
   `(adj[ids][:, ids]).T`, not `adj[ids][:, ids]`. For an undirected graph the
   two coincide; the graph here is `to_directed()`, so they do not. We
   reproduce the transpose exactly.

2. **Parallel edges sum.** `sp.coo_matrix` with duplicate (i, j) coordinates
   *adds* them, so the adjacency is a **multiplicity count**, not a 0/1
   indicator. A MultiGraph with two flows from u to v yields adj[u][v] == 2.0.
   A binary indicator would silently change `exp(-adj / tau)` in the GWD cost.
   We keep the counts.

Both details are pinned by `tests/test_sparse_adj.py::test_matches_reference_dense`,
which compares against the reference's own expression on a graph with parallel
edges and self-loops.
"""

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import torch


class SparseAdjacency:
    """Multiplicity-weighted adjacency with O(1) k x k block extraction.

    Args:
        counts: {(u, v): multiplicity} for every directed edge in the graph.
        num_nodes: node count, for sanity checks only.
    """

    def __init__(self, counts: Counter, num_nodes: int):
        self.counts = counts
        self.num_nodes = num_nodes

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_edges(cls, src: torch.Tensor, dst: torch.Tensor,
                   num_nodes: int) -> "SparseAdjacency":
        """Build from a DGL graph's `edges()` pair, preserving multiplicities."""
        counts = Counter(zip(src.tolist(), dst.tolist()))
        return cls(counts, num_nodes)

    @classmethod
    def from_dgl(cls, g) -> "SparseAdjacency":
        src, dst = g.edges()
        return cls.from_edges(src, dst, g.num_nodes())

    # ------------------------------------------------------------------
    # The one operation the model needs
    # ------------------------------------------------------------------

    def block(self, node_ids: Sequence[int]) -> torch.Tensor:
        """Return the k x k adjacency block for one subgraph.

        Matches the reference expression `adj[node_ids].t()[node_ids]`
        exactly, including the transpose (see module docstring).

        Args:
            node_ids: the subgraph's node list, centre last (as produced by
                `negsc.sub_sam`).

        Returns:
            float32 tensor of shape (k, k), where entry [i][j] is the number
            of edges from node_ids[j] to node_ids[i] -- transposed on purpose.
        """
        k = len(node_ids)
        out = np.zeros((k, k), dtype=np.float32)
        counts = self.counts
        for i, u in enumerate(node_ids):
            for j, v in enumerate(node_ids):
                # transposed relative to (u -> v): we want [u][v] at [j][i]
                c = counts.get((u, v))
                if c:
                    out[j, i] = c
        return torch.from_numpy(out)

    def batch(self, subgraphs: Iterable[Sequence[int]]) -> torch.Tensor:
        """Stack `block()` over a batch: (B, k, k), matching `sub_adj`."""
        return torch.stack([self.block(ids) for ids in subgraphs])

    # ------------------------------------------------------------------
    # Test support only
    # ------------------------------------------------------------------

    def dense(self, device="cpu") -> torch.Tensor:
        """Materialise the full [N, N] matrix. NEVER call this in training --
        it is the allocation this module exists to avoid. Used by the equality
        test against the reference expression."""
        dense = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        for (u, v), c in self.counts.items():
            dense[u, v] = float(c)
        return dense.to(device)


def reference_dense_adjacency(g) -> torch.Tensor:
    """The reference's own construction, verbatim, for equality testing.

    This is the 4.3 GB allocation on a real chunk. Only ever call it on a
    small graph.
    """
    import scipy.sparse as sp

    src, dst = g.edges()
    adj = sp.coo_matrix(
        (np.ones(g.num_edges()), (src.numpy(), dst.numpy())),
        shape=(g.num_nodes(), g.num_nodes()),
        dtype=np.float32,
    ).toarray()
    return torch.from_numpy(adj)


def reference_block(dense: torch.Tensor, node_ids: Sequence[int]) -> torch.Tensor:
    """The reference's block expression, verbatim: `adj[ids].t()[ids]`."""
    return dense[node_ids].t()[node_ids]
