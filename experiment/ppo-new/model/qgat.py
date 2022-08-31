import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim + edge_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        """
        Concat source and target node representations and edge features
        to compute attention coefficients for each edge.
        """
        z2 = torch.cat([edges.src['z'], edges.data['w'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a, negative_slope=0.2)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        # equation (1)
        z: torch.Tensor = self.fc(h)
        z = F.leaky_relu(z, negative_slope=0.2)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        num_heads: int,
        merge: str = 'cat',
    ):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, edge_dim))
        self.merge = merge

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        head_outs: list[torch.Tensor] = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class QGAT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_gate_types: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        add_self_loop: bool = True,
    ):
        super(QGAT, self).__init__()
        self.add_self_loop: bool = add_self_loop
        self.embedding = nn.Embedding(num_gate_types, in_dim)
        self.convs = nn.ModuleList()

        self.convs.append(MultiHeadGATLayer(in_dim, hidden_dim, 3, num_heads))
        for i in range(num_layers - 2):
            self.convs.append(
                MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, 3, num_heads)
            )
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.convs.append(MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 3, 1))

    def forward(self, g: dgl.DGLGraph):
        graph: dgl.DGLGraph
        if self.add_self_loop:
            graph = dgl.add_self_loop(g)
        else:
            graph = g
        h: torch.Tensor = self.embedding(graph.ndata['gate_type'])
        w = torch.cat(
            [
                torch.unsqueeze(graph.edata['src_idx'], 1),
                torch.unsqueeze(graph.edata['dst_idx'], 1),
                torch.unsqueeze(graph.edata['reversed'], 1),
            ],
            dim=1,
        )
        graph.edata['w'] = w
        graph.ndata['z'] = h

        for i in range(len(self.convs)):
            h = self.convs[i](graph, h)
        return h
