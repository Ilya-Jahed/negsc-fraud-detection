"""
negat.py -- "NEGAT / Encoder" panel.

Ports GATlayer, MultiHeadGATLayer, and GAT from NEGSC.ipynb cell 17
unchanged. This same GAT class is instantiated twice in the pipeline: once
as the "Encoder" and once as the "gene" (generative) network shown as
"Generative (NEGAT)" in the diagram -- both are the same architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from dgl.nn.functional import edge_softmax


class GATlayer(nn.Module):
    def __init__(self, n_feat, e_feat, out_feat, num_heads):
        super(GATlayer, self).__init__()
        self.n_feat = n_feat
        self.e_feat = e_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        # Projects [src_node || dst_node || edge] -> out_feat.
        # Produces the *message content* sent along each edge (used in message_func).
        self.W_msg = nn.Linear(2 * n_feat + e_feat, out_feat)
        # Projects [src_node || dst_node || edge] -> 2*out_feat.
        # Feeds into the attention score computation (edge_attention), not the
        # message content itself -- a separate learned projection from W_msg.
        self.W = nn.Linear(2 * n_feat + e_feat, 2 * out_feat)
        # Attention vector: dotted with the W-projected edge features to produce
        # a single unnormalized attention logit per edge (standard GAT-style
        # attention, Velickovic et al.), extended here to include edge features.
        self.a = nn.Parameter(torch.rand(size=(2 * out_feat, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init; gain=sqrt(2) is the recommended gain for ReLU/LeakyReLU.
        gain = math.sqrt(2)
        init.xavier_normal_(self.W.weight, gain=gain)
        init.xavier_normal_(self.a, gain=gain)

    def edge_attention(self, edges):
        # Raw (unnormalized) attention logit e_uv for edge (u -> v):
        # concat [src, dst, edge] -> project with W -> dot with a -> LeakyReLU.
        feat_cat = torch.cat([edges.src['h'], edges.dst['h'], edges.data['h']], dim=1)
        w_feat_cat = self.W(feat_cat)
        return {'e': F.leaky_relu(torch.matmul(w_feat_cat, self.a))}

    def message_func(self, edges):
        # Per-edge message content ('h', via W_msg) and its already-normalized
        # attention weight ('x', filled in by edge_softmax in forward()).
        return {
            'h': self.W_msg(torch.cat([edges.src['h'], edges.dst['h'], edges.data['h']], dim=1)),
            'x': edges.data['x'],
        }

    def reduce_func(self, nodes):
        # Attention-weighted sum of incoming messages: for each destination
        # node, sum (attention_weight * message) over all incoming edges.
        h = (nodes.mailbox['x'] * nodes.mailbox['h']).sum(1)
        return {'h': h}

    def forward(self, g, n_feat, e_feat):
        with g.local_scope():
            g.ndata['h'] = n_feat
            g.edata['h'] = e_feat
            g.apply_edges(self.edge_attention)          # compute raw 'e' per edge
            attention = edge_softmax(g, g.edata['e'])     # normalize per destination node
            g.edata['x'] = attention
            g.update_all(self.message_func, self.reduce_func)  # weighted aggregation
            g.ndata['h'] = F.relu(g.ndata['h'])
            feat = g.ndata['h']
            return feat


class MultiHeadGATLayer(nn.Module):
    def __init__(self, n_feat, e_feat, out_feat, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATlayer(n_feat, e_feat, out_feat, num_heads))

    def forward(self, g, h, e_feat):
        out_feat = [attn_head(g, h, e_feat) for attn_head in self.heads]
        out_feat = torch.cat(out_feat, dim=1).reshape(g.num_nodes(), len(self.heads), -1)
        return out_feat.mean(1)


class GAT(nn.Module):
    """
    NEGAT encoder. Note: the hidden/output width of layer1 is hardcoded to
    39 in the source notebook regardless of the `out_dim` argument passed
    in -- kept as-is for fidelity.
    """

    def __init__(self, in_dim, e_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, e_dim, 39, num_heads)

    def forward(self, g, h, e_feat):
        h = self.layer1(g, h, e_feat)
        g.ndata['h'] = h
        return h, g