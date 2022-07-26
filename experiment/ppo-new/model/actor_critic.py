from __future__ import annotations
from typing import Callable, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.qgnn import *
from model.qgin import *

class ActorCritic(nn.Module):
    def __init__(
        self,
        gnn_type: str,
        num_gate_types: int,
        gate_type_embed_dim: int,
        gnn_num_layers: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        gin_num_mlp_layers: int,
        gin_learn_eps: bool,
        gin_neighbor_pooling_type: str, # sum', 'mean', 'max'
        actor_hidden_size: int,
        critic_hidden_size: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            num_gate_types: serve as the input dim of GNN (imput dim of feature of each node)
            
        """
        super().__init__()
        if gnn_type.lower() == 'QGNN'.lower():
            self.gnn = QGNN(
                gnn_num_layers, num_gate_types, gate_type_embed_dim,
                gnn_hidden_dim, gnn_hidden_dim
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
            )
        else:
            raise NotImplementedError(f'Unknown GNN type {gnn_type}.')

        self.actor = nn.Sequential(
            nn.Linear(gnn_output_dim, actor_hidden_size),
            nn.ReLU(),
            nn.Linear(actor_hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(gnn_output_dim, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor | dgl.DGLGraph, callee: str) -> torch.Tensor:
        if callee == self.gnn_name():
            """
            Get tensor representations of nodes in graph(s)
            Return: torch.Tensor (num_nodes, gnn_output_dim)
            """
            return self.gnn(x)
        elif callee == self.actor_name():
            """
            Evaluate actions for a node
            Args:
                x: (B, gnn_output_dim)
            Return: torch.Tensor (B, action_dim)
            """
            return self.actor(x)
        elif callee == self.critic_name():
            """
            Evaluate nodes in graphs
            Args:
                x: (B, gnn_output_dim)
            Return: torch.Tensor (B, 1)
            """
            return self.critic(x)
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