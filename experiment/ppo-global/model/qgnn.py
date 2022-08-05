from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class QConv(nn.Module):
    def __init__(self, in_feat, inter_dim, out_feat):
        super(QConv, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_feat + 3, inter_dim),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(
            nn.Linear(in_feat + inter_dim, out_feat, bias=False), nn.ReLU())
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def message_func(self, edges):
        return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}

    def reduce_func(self, nodes):
        tmp = self.linear1(nodes.mailbox['m'])
        h = torch.sum(tmp, dim=1)
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        h_linear = self.linear2(h_total)
        h = F.normalize(h_linear, p=2, dim=-1)
        return h


class QGNN(nn.Module):
    def __init__(self, num_layers, num_gate_types, gate_type_embed_dim,
                 h_feats, inter_dim) -> None:
        """
        output_dim = h_feats
        """
        super(QGNN, self).__init__()
        self.embedding = nn.Embedding(num_gate_types, gate_type_embed_dim)
        convs_: List[nn.Module] = []
        conv_0: nn.Module = QConv(gate_type_embed_dim, inter_dim, h_feats)
        convs_.append(conv_0)
        for _ in range(num_layers - 1):
            convs_.append(QConv(h_feats, inter_dim, h_feats))
        self.convs: nn.ModuleList = nn.ModuleList(convs_)

    def forward(self, g: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns h for all nodes and a global READOUT
        The global READOUT is batch-aware
        '''
        num_node_list: list[int] = g.batch_num_nodes().tolist()
        num_graphs = len(num_node_list)
        g.ndata['h'] = self.embedding(g.ndata['gate_type'])
        w = torch.cat([
            torch.unsqueeze(g.edata['src_idx'], 1),
            torch.unsqueeze(g.edata['dst_idx'], 1),
            torch.unsqueeze(g.edata['reversed'], 1)
        ],
                      dim=1)
        g.edata['w'] = w
        h: torch.Tensor = g.ndata['h']
        readouts: List[torch.Tensor] = []
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
            h_per_graph = torch.split(h, num_node_list, dim=0)
            readout_per_graph: List[torch.Tensor] = []
            for h_ in h_per_graph:
                readout_per_graph.append(torch.sum(h_, dim=0))
            readouts.append(torch.stack(readout_per_graph, dim=0))
        return h, torch.cat(readouts, dim=1)
