from typing import Callable, Tuple, List, Any
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
        #print(f'node h {edges.src["h"].shape}')
        #print(f'node w {edges.data["w"].shape}')
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
        #g.edata['w'] = w #self.embed(torch.unsqueeze(w,1))
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
        convs: List[nn.Module] = []
        for _ in range(num_layers - 1):
            convs.append(QConv(h_feats, inter_dim, h_feats))
        self.convs: nn.Module = nn.ModuleList(convs)

    def forward(self, g):
        #print(g.ndata['gate_type'])
        #print(self.embedding)
        g.ndata['h'] = self.embedding(g.ndata['gate_type'])
        w = torch.cat([
            torch.unsqueeze(g.edata['src_idx'], 1),
            torch.unsqueeze(g.edata['dst_idx'], 1),
            torch.unsqueeze(g.edata['reversed'], 1)
        ],
                      dim=1)
        g.edata['w'] = w
        h = self.conv_0(g, g.ndata['h'])
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
        return h


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_gate_type: int,
        graph_embed_size: int,
        actor_hidden_size: int,
        critic_hidden_size: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.critic_embedding = QGNN(6, num_gate_type, graph_embed_size, graph_embed_size)
        self.actor_embedding = QGNN(6, num_gate_type, graph_embed_size, graph_embed_size)
        self.actor = nn.Sequential(
            nn.Linear(graph_embed_size, actor_hidden_size),
            nn.ReLU(),
            nn.Linear(actor_hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(graph_embed_size, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor, callee: str) -> torch.Tensor:
        if callee == self.critic_embedding_name():
            return self.critic_embedding(x)
        elif callee == self.actor_embedding_name():
            return self.actor_embedding(x)
        elif callee == self.actor_name():
            return self.actor(x)
        elif callee == self.critic_name():
            return self.critic(x)
        else:
            raise NotImplementedError(f'Unexpected callee name: {callee}')
    
    @staticmethod
    def critic_embedding_name() -> str:
        return 'critic_embedding'
    
    @staticmethod
    def actor_embedding_name() -> str:
        return 'critic_embedding'
    
    @staticmethod
    def actor_name() -> str:
        return 'actor'
    
    @staticmethod
    def critic_name() -> str:
        return 'critic'
    