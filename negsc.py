"""
negsc.py -- NEGSC panel: subgraph sampling and contrastive loss (WD + GWD).

Source cells (NEGSC.ipynb, 0-indexed):
  Cell 10 -> MLPPredictor
  Cell 12 -> Genetation  (dead code in notebook -- omitted here, see note)
  Cell 13 -> sub_sam
  Cell 14 -> Model  (WD / GWD contrastive loss)
  Cell 15 -> train()

The Genetation class (cell 12) is defined in the notebook but never
instantiated in the real training pipeline -- cell 19 builds `gene` as a
second GAT instance instead.  It is omitted to keep the file readable.

All .cuda() calls are written as .to(DEVICE) so the code runs on both
CPU and GPU without any changes, matching the convention in main.py.
"""

import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defined at module level to match the original notebook scope (cell 20).
# Used inside Model.sub_loss_batch.
b_xent = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------------------
# MLPPredictor  (notebook cell 10)
# Produces a per-edge embedding by concatenating the src and dst node
# embeddings and passing them through a linear layer.
# Used during contrastive training (_edge_features) to embed edges, and
# in the predict stage as the downstream edge classifier.
# ---------------------------------------------------------------------------

class MLPPredictor(nn.Module):
    def __init__(self, in_features: int, out_classes: int):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        edge_emb = self.W(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'edge_emb': edge_emb}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['edge_emb']


# ---------------------------------------------------------------------------
# sub_sam  (notebook cell 13)
# "Sample Node / Sample Subgraphs" panel.
#
# For each node in `nodes`, if it has >= k neighbors, randomly sample k-1
# of them and build a subgraph entry: [neighbor_1, ..., neighbor_{k-1}, center].
# Nodes with fewer than k neighbors are skipped.
#
# NOTE: the double-write of node_neighbor_cen (once inside the if-block,
# once after num_nei += 1) and the trailing [:-1] trim are present verbatim
# in the source notebook.  Do not "fix" them without re-validating results.
# ---------------------------------------------------------------------------

def sub_sam(nodes, adj_lists: dict, k: int) -> list:
    """
    Build ego-subgraphs of size k for a batch of nodes.

    Args:
        nodes:      1-D tensor of node indices.
        adj_lists:  dict mapping node_id -> set of neighbor node_ids.
        k:          subgraph size (1 center + k-1 sampled neighbors).

    Returns:
        List of subgraph node lists.  Each inner list has length k, with
        the center node placed last.
    """
    n = nodes.shape[0]
    subgraphs = [[] for _ in range(n)]
    subgraph_count = 0

    for node in nodes:
        node_id = int(node)
        neighbor_set = adj_lists[node_id]
        subgraph_nodes = adj_lists[node_id]  # fallback if node has too few neighbors

        if len(neighbor_set) >= k:
            # Remove the center itself, sample k-1 random neighbors
            eligible_neighbors = neighbor_set - {node_id}
            subgraph_nodes = random.sample(eligible_neighbors, k - 1)

            # Write neighbors-only first (matches notebook structure)
            subgraphs[subgraph_count] = list(subgraph_nodes)

            # Append center to complete the subgraph entry
            subgraph_nodes = list(subgraph_nodes) + [node_id]
            subgraphs[subgraph_count] = subgraph_nodes
            subgraph_count += 1

        # Overwrite next slot -- verbatim from the source notebook
        subgraphs[subgraph_count] = list(subgraph_nodes)

    # Drop unfilled slots and the trailing duplicate produced by the loop
    subgraphs = [sg for sg in subgraphs if sg]
    subgraphs = subgraphs[:-1]
    return subgraphs


# ---------------------------------------------------------------------------
# Model  (notebook cell 14)
# Wraps the Encoder (NEGAT GAT) and the generative network (also a GAT).
# Implements: WD contrastive loss on node embeds, GWD on graph structure,
# and WD contrastive loss on edge features.
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    NEGSC contrastive model.

    Args:
        Encoder:  negat.GAT used as the main encoder.
        gene:     negat.GAT used as the generative (augmented) network.
        tau:      temperature for the cost matrix (default 0.5, matching
                  the notebook default -- see fidelity note in main.py).
    """

    def __init__(self, Encoder, gene, tau: float = 0.5):
        super().__init__()
        self.encoder = Encoder
        self.ge = gene
        self.tau = tau

    # ------------------------------------------------------------------
    # Forward / embed
    # ------------------------------------------------------------------

    def forward(self, graph, node_feats, edge_feats):
        """Run encoder and generative network; return both outputs."""
        enc_node_emb, enc_graph = self.encoder(graph, node_feats, edge_feats)
        gen_node_emb, gen_graph = self.ge(graph, enc_node_emb, edge_feats)
        return enc_node_emb, gen_node_emb, enc_graph, gen_graph

    def embed(self, graph, node_feats, edge_feats):
        """Return encoder node embeddings only (used at inference time)."""
        enc_node_emb, _ = self.encoder(graph, node_feats, edge_feats)
        return enc_node_emb

    # ------------------------------------------------------------------
    # Loss entry point
    # ------------------------------------------------------------------

    def loss(self, enc_node_emb, gen_node_emb, adj, subgraphs, enc_graph, gen_graph):
        return self.sub_loss_batch(enc_node_emb, gen_node_emb, adj, subgraphs, enc_graph, gen_graph)

    # ------------------------------------------------------------------
    # Contrastive loss  (notebook cell 14 -- sub_loss_batch)
    #
    # Three pairs are formed by tiling:
    #   Positive         : (subgraph z,  subgraph z_g)        -- same subgraph
    #   Negative (enc)   : (subgraph z,  shuffled z)          -- mismatched encoder
    #   Negative (gen)   : (subgraph z,  shuffled z_g)        -- mismatched generator
    # ------------------------------------------------------------------

    def sub_loss_batch(self, enc_node_emb, gen_node_emb, adj, subgraphs, enc_graph, gen_graph):
        enc_subgraph_emb, gen_subgraph_emb = self._subgraph_centers(
            enc_node_emb, gen_node_emb, subgraphs
        )

        # Build a shuffle index so that index i maps to a different subgraph.
        # (Logic preserved verbatim from notebook.)
        shuffle_idx = torch.randint(0, len(subgraphs) - 1, (len(subgraphs),))
        if shuffle_idx[0] == 0:
            shuffle_idx[0] = 1
        for i in range(1, len(shuffle_idx)):
            if shuffle_idx[i] == i:
                shuffle_idx[i] -= 1

        # Negative (shuffled) node embeddings
        enc_subgraph_neg = enc_subgraph_emb[shuffle_idx]   # mismatched encoder
        gen_subgraph_neg = gen_subgraph_emb[shuffle_idx]   # mismatched generator

        # Tile into [positive | neg-enc | neg-gen]
        node_anchor = torch.cat((enc_subgraph_emb, enc_subgraph_emb, enc_subgraph_emb), dim=0)
        node_paired = torch.cat((gen_subgraph_emb, enc_subgraph_neg, gen_subgraph_neg), dim=0)

        # Edge-level inputs
        enc_edge_emb, gen_edge_emb = self._edge_features(
            enc_graph, gen_graph, subgraphs, enc_node_emb, gen_node_emb
        )
        enc_edge_neg = enc_edge_emb[shuffle_idx]
        gen_edge_neg = gen_edge_emb[shuffle_idx]
        edge_anchor = torch.cat((enc_edge_emb, enc_edge_emb, enc_edge_emb), dim=0).requires_grad_(True)
        edge_paired = torch.cat((gen_edge_emb, enc_edge_neg, gen_edge_neg), dim=0).requires_grad_(True)

        # Adjacency (tiled to match the three-way cat)
        subgraph_adj = self._subgraph_adj(adj, subgraphs)
        tiled_adj = torch.cat((subgraph_adj, subgraph_adj, subgraph_adj), dim=0)

        # Labels: 1 for the positive pair, 0 for the two negative pairs
        n_sub  = len(subgraphs)
        n_edge = len(gen_edge_emb)
        node_labels = torch.cat([torch.ones(n_sub),  torch.zeros(n_sub  * 2)]).to(DEVICE)
        edge_labels = torch.cat([torch.ones(n_edge), torch.zeros(n_edge * 2)]).to(DEVICE)

        # --- WD loss on node embeddings ---
        node_wd, ot_plan = self._wd(node_anchor, node_paired, self.tau)
        node_wd_loss = b_xent(torch.squeeze(torch.exp(-node_wd / 0.01)), node_labels)
        logger.debug("node_wd_loss: %.6f", node_wd_loss.item())

        # --- GWD loss (graph structure) ---
        graph_gwd = self._gwd(
            node_anchor.transpose(2, 1), node_paired.transpose(2, 1),
            ot_plan, tiled_adj, self.tau,
        )
        graph_gwd_loss = b_xent(torch.squeeze(torch.exp(-graph_gwd / 0.1)), node_labels)
        logger.debug("graph_gwd_loss: %.6f", graph_gwd_loss.item())

        # --- WD loss on edge embeddings ---
        edge_wd, _ = self._wd(edge_anchor, edge_paired, self.tau)
        edge_wd_loss = b_xent(torch.squeeze(torch.exp(-edge_wd / 0.01)), edge_labels)
        logger.debug("edge_wd_loss: %.6f", edge_wd_loss.item())

        # Combined loss (matches notebook: 0.5 * loss3 + 0.5 * loss2)
        return 0.5 * edge_wd_loss + 0.5 * graph_gwd_loss

    # ------------------------------------------------------------------
    # Helper: edge features per subgraph  (notebook: edges_f)
    #
    # NOTE: MLPPredictor is instantiated fresh each call with random weights,
    # matching the source notebook exactly (cell 14).  The edge scores are
    # therefore random and untrained -- this is how the paper implemented it.
    # ------------------------------------------------------------------

    def _edge_features(self, enc_graph, gen_graph, subgraphs, enc_node_emb, gen_node_emb):
        edge_embedder = MLPPredictor(enc_graph.edata['h'].shape[1], 39).to(DEVICE)
        enc_edge_emb = edge_embedder(enc_graph, enc_node_emb)
        gen_edge_emb = edge_embedder(gen_graph, gen_node_emb)

        # Pre-allocate with empties (matches notebook) then append real entries
        enc_edge_feats = [[] for _ in range(len(subgraphs))]
        gen_edge_feats = [[] for _ in range(len(subgraphs))]

        for subgraph in subgraphs:
            center    = subgraph[-1]
            neighbors = subgraph[:-1]

            for neighbor in neighbors:
                edge_ids = enc_graph.edge_ids(center, neighbor, return_uv=True)

                enc_feat = torch.Tensor(enc_edge_emb[edge_ids[2]]).float().tolist()
                gen_feat = torch.Tensor(gen_edge_emb[edge_ids[2]]).float().tolist()

                enc_edge_feats.append(enc_feat)
                gen_edge_feats.append(gen_feat)

                # If two parallel edges were returned, keep only the first
                if len(enc_edge_feats[-1]) == 2:
                    enc_edge_feats[-1] = [enc_edge_feats[-1][0]]
                    gen_edge_feats[-1] = [gen_edge_feats[-1][0]]

        # Drop the pre-allocated empty slots
        enc_edge_feats = [e for e in enc_edge_feats if e]
        gen_edge_feats = [e for e in gen_edge_feats if e]

        enc_edge_feats = torch.Tensor(enc_edge_feats).reshape(len(subgraphs), -1, 39)
        gen_edge_feats = torch.Tensor(gen_edge_feats).reshape(len(subgraphs), -1, 39)
        return enc_edge_feats, gen_edge_feats

    # ------------------------------------------------------------------
    # Helper: k×k adjacency block for each subgraph  (notebook: sub_adj)
    # ------------------------------------------------------------------

    def _subgraph_adj(self, adj, subgraphs) -> torch.Tensor:
        subgraph_size = len(subgraphs[0])
        subgraph_adj = torch.zeros(len(subgraphs), subgraph_size, subgraph_size)
        for i, node_ids in enumerate(subgraphs):
            subgraph_adj[i] = adj[node_ids].t()[node_ids]
        return subgraph_adj

    # ------------------------------------------------------------------
    # Helper: gather + reshape node embeddings  (notebook: subg_centor)
    # ------------------------------------------------------------------

    def _subgraph_centers(self, enc_node_emb, gen_node_emb, subgraphs):
        """Index enc/gen embeddings by subgraph nodes and reshape to (B, k, D)."""
        all_nodes = [node for sg in subgraphs for node in sg]
        enc_emb = enc_node_emb[all_nodes].reshape(len(subgraphs), len(subgraphs[0]), -1)
        gen_emb = gen_node_emb[all_nodes].reshape(len(subgraphs), len(subgraphs[0]), -1)
        return enc_emb, gen_emb

    # ------------------------------------------------------------------
    # Wasserstein Distance  (notebook: wd, OT_distance_batch, OT_batch,
    #                        cost_matrix_batch, batch_trace)
    # ------------------------------------------------------------------

    def _wd(self, x, y, tau):
        """Batched Wasserstein distance between x and y."""
        cost_matrix = self._cost_matrix_batch(
            x.transpose(2, 1), y.transpose(2, 1), tau
        ).transpose(1, 2)

        # Soft-threshold to zero out small costs
        beta = 0.1
        cost_min, cost_max = cost_matrix.min(), cost_matrix.max()
        thresholded_cost = F.relu(cost_matrix - (cost_min + beta * (cost_max - cost_min)))

        return self._ot_distance_batch(thresholded_cost, x.size(0), x.size(1), y.size(1), iterations=40)

    def _ot_distance_batch(self, cost, bs, n, m, iterations=50):
        cost = cost.float().to(DEVICE)
        transport_plan = self._ot_batch(cost, bs, n, m, iteration=iterations)
        distance = self._batch_trace(torch.bmm(cost.transpose(1, 2), transport_plan), m, bs)
        return distance, transport_plan

    def _ot_batch(self, cost, bs, n, m, beta=0.5, iteration=50):
        """Sinkhorn iterations for batched optimal transport."""
        sigma = torch.ones(bs, int(m), 1).to(DEVICE) / float(m)
        transport_plan = torch.ones(bs, n, m).to(DEVICE)
        kernel = torch.exp(-cost / beta).float().to(DEVICE)

        for _ in range(iteration):
            scaled_kernel = kernel * transport_plan
            # Single normalisation step (matches notebook: `for k in range(1)`)
            delta = 1.0 / (n * torch.bmm(scaled_kernel, sigma))
            a     = torch.bmm(scaled_kernel.transpose(1, 2), delta)
            sigma = 1.0 / (float(m) * a)
            transport_plan = delta * scaled_kernel * sigma.transpose(2, 1)
        return transport_plan

    def _cost_matrix_batch(self, x, y, tau=0.5):
        """Batched cosine-distance cost matrix."""
        bs, D = x.size(0), x.size(1)
        assert x.size(1) == y.size(1)
        x = x.contiguous().view(bs, D, -1)
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cosine_sim = torch.bmm(x.transpose(1, 2), y)
        return torch.exp(-cosine_sim / tau).transpose(2, 1)

    def _batch_trace(self, matrix, n, bs):
        identity = torch.eye(n).to(DEVICE).unsqueeze(0).expand(bs, -1, -1)
        return torch.sum(torch.sum(identity * matrix, dim=-1), dim=-1).unsqueeze(1)

    # ------------------------------------------------------------------
    # Gromov-Wasserstein Distance  (notebook: gwd, GW_distance,
    #                               GW_batch, cos_batch)
    # ------------------------------------------------------------------

    def _gwd(self, X, Y, ot_plan, tiled_adj, tau, lamda=1e-1, iteration=5, ot_iteration=20):
        """Batched Gromov-Wasserstein distance."""
        bs   = X.size(0)
        m, n = X.size(2), Y.size(2)
        p = (torch.ones(bs, m, 1) / m).to(DEVICE)
        q = (torch.ones(bs, n, 1) / n).to(DEVICE)
        return self._gw_distance(X, Y, p, q, ot_plan, tiled_adj, tau,
                                 lamda=lamda, iteration=iteration, ot_iteration=ot_iteration)

    def _gw_distance(self, X, Y, p, q, ot_plan, tiled_adj, tau,
                     lamda=0.5, iteration=5, ot_iteration=20):
        # Source cost matrix: soft-thresholded adjacency
        adj_cost = torch.exp(-tiled_adj / tau).to(DEVICE)
        beta = 0.1
        lo, hi = adj_cost.min(), adj_cost.max()
        source_cost = F.relu((adj_cost - (lo + beta * (hi - lo))).transpose(2, 1))

        # Target cost matrix: Y self-similarity
        target_cost = self._cos_batch(Y, Y, tau).float().to(DEVICE)

        bs = source_cost.size(0)
        n, m = source_cost.size(2), target_cost.size(2)
        transport_plan, cost_tensor = self._gw_batch(
            source_cost, target_cost, bs, n, m, p, q,
            beta=lamda, iteration=iteration, ot_iteration=ot_iteration
        )
        return self._batch_trace(torch.bmm(cost_tensor.transpose(1, 2), ot_plan), m, bs)

    def _gw_batch(self, source_cost, target_cost, bs, n, m, p, q,
                  beta=0.5, iteration=5, ot_iteration=20):
        ones_m = torch.ones(bs, m, 1).float().to(DEVICE)
        ones_n = torch.ones(bs, n, 1).float().to(DEVICE)

        cost_tensor = (
            torch.bmm(torch.bmm(source_cost ** 2, p), ones_m.transpose(1, 2))
            + torch.bmm(ones_n, torch.bmm(q.transpose(1, 2), target_cost.transpose(1, 2) ** 2))
        )
        transport_plan = torch.bmm(p, q.transpose(2, 1))

        for _ in range(iteration):
            updated_cost = cost_tensor - 2 * torch.bmm(
                torch.bmm(source_cost, transport_plan), target_cost.transpose(1, 2)
            )
            transport_plan = self._ot_batch(updated_cost, bs, n, m, beta=beta, iteration=ot_iteration)

        final_cost = cost_tensor - 2 * torch.bmm(
            torch.bmm(source_cost, transport_plan), target_cost.transpose(1, 2)
        )
        return transport_plan.detach(), final_cost

    def _cos_batch(self, x, y, tau):
        """Soft-thresholded batched cosine similarity matrix (used for target_cost)."""
        bs, D = x.size(0), x.size(1)
        assert x.size(1) == y.size(1)
        x = x.contiguous().view(bs, D, -1)
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cosine_sim = torch.exp(-torch.bmm(x.transpose(1, 2), y) / tau).transpose(1, 2)

        beta = 0.1
        lo, hi = cosine_sim.min(), cosine_sim.max()
        return F.relu((cosine_sim - (lo + beta * (hi - lo))).transpose(2, 1))


# ---------------------------------------------------------------------------
# train()  (notebook cell 15)
# ---------------------------------------------------------------------------

def train(model: Model, g, node_feats, edge_feats, adj,
          node_neighbor_cen, optimizer) -> float:
    """
    Single training step.

    Args:
        model:             NEGSC Model instance.
        g:                 DGL graph for this chunk (on DEVICE).
        node_feats:        Node feature tensor.
        edge_feats:        Edge feature tensor.
        adj:               Dense adjacency matrix (torch.Tensor on DEVICE).
        node_neighbor_cen: Subgraph list from sub_sam().
        optimizer:         torch optimizer.

    Returns:
        Scalar loss value for this step.
    """
    model.train()
    optimizer.zero_grad()
    enc_node_emb, gen_node_emb, enc_graph, gen_graph = model(g, node_feats, edge_feats)
    loss = model.loss(enc_node_emb, gen_node_emb, adj, node_neighbor_cen, enc_graph, gen_graph)
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()