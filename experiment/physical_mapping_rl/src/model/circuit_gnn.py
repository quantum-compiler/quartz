from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


def circuit_message_func(edges):
    """incorporate edges' features by cat"""
    return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}


class CircuitGraphConv(nn.Module):
    def __init__(self, in_feat: int, inter_dim: int, out_feat: int):
        super(CircuitGraphConv, self).__init__()
        self.linear2 = nn.Linear(in_feat + inter_dim, out_feat)
        self.linear1 = nn.Linear(in_feat + 3, inter_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)

    def reduce_func(self, nodes):
        # nodes.mailbox['m']: (num_nodes, num_neighbors, msg_dim)
        tmp = self.linear1(nodes.mailbox['m'])
        tmp = F.leaky_relu(tmp)
        """aggregate neighbors' features"""
        # (num_nodes, num_neighbors, msg_dim) -> (num_nodes, msg_dim)
        h = torch.mean(tmp, dim=1)  # average on dim of num_neighbors
        # h = torch.max(tmp, dim=1).values
        return {'h_N': h}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        g.ndata['h'] = h
        # g.edata['w'] = w #self.embed(torch.unsqueeze(w,1))
        g.update_all(circuit_message_func, self.reduce_func)
        h_N = g.ndata['h_N']  # (num_nodes, inter_dim)
        """combine node's feature with its neighbors' to get its feature at the next layer"""
        h_total = torch.cat([h, h_N], dim=1)
        h_linear = self.linear2(h_total)
        h_relu = F.relu(h_linear)
        # h_norm = torch.unsqueeze(torch.linalg.norm(h_relu, dim=1), dim=1)
        # h_normed = torch.divide(h_relu, h_norm)
        # return h_normed
        return h_relu


class CircuitGNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_gate_types,
                 gate_type_embed_dim,
                 h_feats,
                 inter_dim
                 ) -> None:
        """
        output_dim = h_feats
        """
        super(CircuitGNN, self).__init__()
        self.embedding = nn.Embedding(num_gate_types, gate_type_embed_dim)
        self.conv_0 = CircuitGraphConv(gate_type_embed_dim, inter_dim, h_feats)
        convs: List[nn.Module] = []
        for _ in range(num_layers - 1):
            convs.append(CircuitGraphConv(h_feats, inter_dim, h_feats))
        self.convs: nn.Module = nn.ModuleList(convs)

    def forward(self, g: dgl.DGLGraph):
        # print(g.ndata['gate_type'])
        # print(self.embedding)
        g.ndata['h'] = self.embedding(g.ndata['is_input'])
        w = torch.cat([torch.unsqueeze(g.edata['logical_idx'], 1),
                       torch.unsqueeze(g.edata['physical_idx'], 1),
                       torch.unsqueeze(g.edata['reversed'], 1)], dim=1)
        g.edata['w'] = w
        h = self.conv_0(g, g.ndata['h'])
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
        return h
