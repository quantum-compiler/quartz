import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class QConv(nn.Module):
    def __init__(self, in_feat, inter_dim, out_feat):
        super(QConv, self).__init__()
        self.linear2 = nn.Linear(in_feat + inter_dim, out_feat)
        self.linear1 = nn.Linear(in_feat + 3, inter_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)

    def message_func(self, edges):
        # print(f'node h {edges.src["h"].shape}')
        # print(f'node w {edges.data["w"].shape}')
        return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}

    def reduce_func(self, nodes):
        # print(f'node m {nodes.mailbox["m"].shape}')
        tmp = self.linear1(nodes.mailbox['m'])
        tmp = F.leaky_relu(tmp)
        h = torch.mean(tmp, dim=1)
        # h = torch.max(tmp, dim=1).values
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        # g.edata['w'] = w #self.embed(torch.unsqueeze(w,1))
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        h_linear = self.linear2(h_total)
        h_relu = F.relu(h_linear)
        # h_norm = torch.unsqueeze(torch.linalg.norm(h_relu, dim=1), dim=1)
        # h_normed = torch.divide(h_relu, h_norm)
        # return h_normed
        return h_relu


class QGNN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, inter_dim) -> None:
        super(QGNN, self).__init__()
        self.embedding = nn.Embedding(in_feats, in_feats)
        self.conv_0 = QConv(in_feats, inter_dim, h_feats)
        self.convs = []
        for _ in range(num_layers - 1):
            self.convs.append(QConv(h_feats, inter_dim, h_feats))
        self.convs = nn.ModuleList(self.convs)

    def forward(self, g):
        # print(g.ndata['gate_type'])
        # print(self.embedding)
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
        h = self.conv_0(g, g.ndata['h'])
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
        return h


# class QGNN(nn.Module):
#     def __init__(self, in_feats, h_feats, inter_dim):
#         super(QGNN, self).__init__()
#         self.conv1 = QConv(in_feats, inter_dim, h_feats)
#         self.conv2 = QConv(h_feats, inter_dim, h_feats)
#         self.conv3 = QConv(h_feats, inter_dim, h_feats)
#         self.conv4 = QConv(h_feats, inter_dim, h_feats)
#         self.conv5 = QConv(h_feats, inter_dim, h_feats)
#         # self.linear1 = nn.Linear(h_feats, 32)
#         # self.linear2 = nn.Linear(32, num_classes)
#         # gain = nn.init.calculate_gain('relu')
#         # nn.init.xavier_normal_(self.linear1.weight, gain=gain)
#         # nn.init.xavier_normal_(self.linear2.weight, gain=gain)
#         self.embedding = nn.Embedding(in_feats, in_feats)

#     def forward(self, g):
#         #print(g.ndata['gate_type'])
#         #print(self.embedding)
#         g.ndata['h'] = self.embedding(g.ndata['gate_type'])
#         w = torch.cat([
#             torch.unsqueeze(g.edata['src_idx'], 1),
#             torch.unsqueeze(g.edata['dst_idx'], 1),
#             torch.unsqueeze(g.edata['reversed'], 1)
#         ],
#                       dim=1)
#         g.edata['w'] = w
#         h = self.conv1(g, g.ndata['h'])
#         h = self.conv2(g, h)
#         h = self.conv3(g, h)
#         h = self.conv4(g, h)
#         h = self.conv5(g, h)
#         # h = self.linear1(h)
#         # h = F.relu(h)
#         # h = self.linear2(h)
#         return h
