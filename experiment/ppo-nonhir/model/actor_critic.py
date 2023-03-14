from __future__ import annotations

from typing import Any, Callable, List, Tuple, cast

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basis import *
from model.qgin import *
from model.qgnn import *
from torch.nn.parallel import DistributedDataParallel as DDP


class NodeGraphAttn(nn.Module):
    def __init__(
        self,
        node_embed_dim: int,
        graph_embed_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.node_linear = nn.Sequential(
            nn.Linear(node_embed_dim, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
        )
        self.graph_linear = nn.Sequential(
            nn.Linear(graph_embed_dim, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, g_feats: torch.Tensor, n_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g_feats: (B, graph_embed_dim)
            n_feats: (B, num_nodes, node_embed_dim)
        Return: (B, num_nodes)
        """
        g_feats = self.graph_linear(g_feats)
        n_feats = self.node_linear(n_feats)
        scores = torch.bmm(n_feats, g_feats.unsqueeze(-1)).squeeze(-1)
        return scores


class NonHirActorCritic(nn.Module):
    def __init__(
        self,
        gnn_type: str = 'QGIN',
        num_gate_types: int = 20,
        gate_type_embed_dim: int = 16,
        gnn_num_layers: int = 6,
        gnn_hidden_dim: int = 32,
        gnn_output_dim: int = 32,
        gin_num_mlp_layers: int = 2,
        gin_learn_eps: bool = False,
        gin_neighbor_pooling_type: str = 'sum',  # sum', 'mean', 'max'
        gin_graph_pooling_type: str = None,  # 'sum', 'mean', 'max'
        actor_hidden_size: int = 32,
        critic_hidden_size: int = 32,
        action_dim: int = 32,
        device: torch.device = 'cpu',
        gnn: nn.Module = None,
        actor: nn.Module = None,
        critic: nn.Module = None,
        attn: nn.Module = None,
    ) -> None:
        """
        Args:
            num_gate_types: serve as the input dim of GNN (imput dim of feature of each node)

        """
        super().__init__()
        """init the network with existed modules"""
        if gnn is not None:
            assert actor is not None
            assert critic is not None
            self.gnn = gnn
            self.actor = actor
            self.critic = critic
            self.attn = attn
            self.device = device
            return

        self.gnn_num_layers = gnn_num_layers
        self.device = device
        if gnn_type.lower() == 'QGNN'.lower():
            self.gnn = QGNN(
                gnn_num_layers,
                num_gate_types,
                gate_type_embed_dim,
                gnn_hidden_dim,
                gnn_hidden_dim,
            )
            gnn_output_dim = gnn_hidden_dim
        elif gnn_type.lower() == 'QGIN'.lower():
            self.gnn = QGIN(
                num_layers=gnn_num_layers,
                num_mlp_layers=gin_num_mlp_layers,
                num_gate_types=num_gate_types,
                gate_type_embed_dim=gate_type_embed_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                learn_eps=gin_learn_eps,
                neighbor_pooling_type=gin_neighbor_pooling_type,
                graph_pooling_type=gin_graph_pooling_type,
            )
        else:
            raise NotImplementedError(f'Unknown GNN type {gnn_type}.')

        self.actor = MLP(2, gnn_output_dim, actor_hidden_size, action_dim)
        self.critic = MLP(2, gnn_output_dim, critic_hidden_size, 1)
        self.attn = NodeGraphAttn(gnn_output_dim, gnn_output_dim, gnn_output_dim)

    def ddp_model(self) -> NonHirActorCritic:
        """make ddp verison instances for each sub-model"""
        _ddp_model = NonHirActorCritic(
            device=self.device,
            gnn=DDP(self.gnn, device_ids=[self.device]),
            actor=DDP(self.actor, device_ids=[self.device]),
            critic=DDP(self.critic, device_ids=[self.device]),
            attn=DDP(self.attn, device_ids=[self.device]),
        )
        return _ddp_model

    def forward(self, x: Any, callee: str) -> torch.Tensor:
        if callee == self.gnn_name():
            """
            Get tensor representations of nodes in graph(s)
            Return: torch.Tensor (num_nodes, gnn_output_dim)
            """
            x = cast(dgl.DGLGraph, x)
            return self.gnn(x)
        elif callee == self.actor_name():
            """
            Evaluate actions for a node
            Args:
                x: (B, gnn_output_dim)
            Return: torch.Tensor (B, action_dim)
            """
            x = cast(torch.Tensor, x)
            return self.actor(x)
        elif callee == self.critic_name():
            """
            Evaluate nodes in graphs
            Args:
                x: (B, gnn_output_dim)
            Return: torch.Tensor (B, 1)
            """
            x = cast(torch.Tensor, x)
            return self.critic(x)
        elif callee == self.attn_name():
            """
            Evaluate nodes in graphs
            Args:
                x: (g_feats, n_feats)
                    g_feats: (B, graph_embed_dim)
                    n_feats: (B, num_nodes, node_embed_dim)
            Return: torch.Tensor (B, num_nodes)
            """
            x = cast(Tuple[torch.Tensor, torch.Tensor], x)
            return self.attn(x[0], x[1])
        else:
            raise NotImplementedError(f'Unexpected callee name: {callee}')

    @staticmethod
    def gnn_name() -> str:
        return 'gnn'

    @staticmethod
    def actor_name() -> str:
        return 'actor'

    @staticmethod
    def critic_name() -> str:
        return 'critic'

    @staticmethod
    def attn_name() -> str:
        return 'attn'
