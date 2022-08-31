from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class QConv(nn.Module):
    def __init__(
        self,
        in_feat: int,
        inter_dim: int,
        out_feat: int,
        aggregator_type: str = 'sum',
        normalize: bool = False,
    ):
        super(QConv, self).__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(in_feat + 3, inter_dim),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_feat + inter_dim, out_feat, bias=False),
            nn.ReLU(),
        )
        self.apply(self._init_weights)
        self.aggregator_type: str = aggregator_type
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
        # NOTE: "Pooling aggregator" of GraphSAGE is defined as a Linear and an activation
        tmp = self.aggregator(nodes.mailbox['m'])
        if self.aggregator_type == 'sum':
            h = torch.sum(tmp, dim=1)
        elif self.aggregator_type == 'mean':
            h = torch.mean(tmp, dim=1)
        elif self.aggregator_type == 'max':
            h = torch.max(tmp, dim=1).values
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
    def __init__(
        self,
        num_layers: int,
        num_gate_types: int,
        gate_type_embed_dim: int,
        h_feats: int,
        inter_dim: int,
    ) -> None:
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
        w = torch.cat(
            [
                torch.unsqueeze(g.edata['src_idx'], 1),
                torch.unsqueeze(g.edata['dst_idx'], 1),
                torch.unsqueeze(g.edata['reversed'], 1),
            ],
            dim=1,
        )
        g.edata['w'] = w
        h: torch.Tensor = g.ndata['h']
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
        return h
