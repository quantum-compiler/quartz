from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class QConv(nn.Module):
    def __init__(self,
                 in_feat,
                 inter_dim,
                 out_feat,
                 aggregator='sum',
                 normalize=False):
        super(QConv, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_feat + 3, inter_dim),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(
            nn.Linear(in_feat + inter_dim, out_feat, bias=False), nn.ReLU())
        self.apply(self._init_weights)
        self.aggregator: str = aggregator
        self.normalize: bool = normalize

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
        if self.aggregator == 'sum':
            h = torch.sum(tmp, dim=1)
        elif self.aggregator == 'mean':
            h = torch.mean(tmp, dim=1)
        elif self.aggregator == 'max':
            h = torch.max(tmp, dim=1)[0]
        else:
            raise NotImplementedError
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        h = self.linear2(h_total)
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
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

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        g.ndata['h'] = self.embedding(g.ndata['gate_type'])
        w = torch.cat([
            torch.unsqueeze(g.edata['src_idx'], 1),
            torch.unsqueeze(g.edata['dst_idx'], 1),
            torch.unsqueeze(g.edata['reversed'], 1)
        ],
                      dim=1)
        g.edata['w'] = w
        h: torch.Tensor = g.ndata['h']
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
        return h