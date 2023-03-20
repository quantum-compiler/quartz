from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basis import *
from model.qgin import *
from model.qgnn import *
from torch.nn.parallel import DistributedDataParallel as DDP


class ActorCritic(nn.Module):
    def __init__(
        self,
        gnn_type: str = "QGIN",
        num_gate_types: int = 20,
        gate_type_embed_dim: int = 16,
        gnn_num_layers: int = 6,
        gnn_hidden_dim: int = 32,
        gnn_output_dim: int = 32,
        gin_num_mlp_layers: int = 2,
        gin_learn_eps: bool = False,
        gin_neighbor_pooling_type: str = "sum",  # sum', 'mean', 'max'
        gin_graph_pooling_type: str = None,  # 'sum', 'mean', 'max'
        actor_hidden_size: int = 32,
        critic_hidden_size: int = 32,
        action_dim: int = 32,
        device: torch.device = "cpu",
        gnn: nn.Module = None,
        actor_gate: nn.Module = None,
        actor_xfer: nn.Module = None,
        critic: nn.Module = None,
    ) -> None:
        """
        Args:
            num_gate_types: serve as the input dim of GNN (imput dim of feature of each node)

        """
        super().__init__()
        """init the network with existed modules"""
        if gnn is not None:
            assert actor_gate is not None
            assert actor_xfer is not None
            assert critic is not None
            self.gnn = gnn
            self.actor_gate = actor_gate
            self.actor_xfer = actor_xfer
            self.critic = critic
            self.device = device
            return

        self.gnn_num_layers = gnn_num_layers
        self.device = device
        if gnn_type.lower() == "QGNN".lower():
            self.gnn = QGNN(
                gnn_num_layers,
                num_gate_types,
                gate_type_embed_dim,
                gnn_hidden_dim,
                gnn_hidden_dim,
            )
            gnn_output_dim = gnn_hidden_dim
        elif gnn_type.lower() == "QGIN".lower():
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
            raise NotImplementedError(f"Unknown GNN type {gnn_type}.")

        self.actor_gate = nn.MultiheadAttention(
            embed_dim=gnn_output_dim,
            num_heads=1,
            kdim=gnn_output_dim * gnn_num_layers,
            vdim=1,
        )
        self.actor_xfer = MLP(
            2, gnn_output_dim * gnn_num_layers, actor_hidden_size, action_dim
        )
        self.critic = MLP(2, gnn_output_dim, critic_hidden_size, 1)

    def ddp_model(self) -> ActorCritic:
        """make ddp verison instances for each sub-model"""
        _ddp_model = ActorCritic(
            device=self.device,
            gnn=DDP(self.gnn, device_ids=[self.device]),
            actor_gate=DDP(self.actor_gate, device_ids=[self.device]),
            actor_xfer=DDP(self.actor_xfer, device_ids=[self.device]),
            critic=DDP(self.critic, device_ids=[self.device]),
        )
        return _ddp_model

    def forward(
        self,
        x: torch.Tensor | dgl.DGLGraph,
        callee: str,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if callee == self.gnn_name():
            """
            Get tensor representations of nodes in graph(s)
            Return: torch.Tensor (num_nodes, gnn_output_dim)
            """
            return self.gnn(x)
        elif callee == self.actor_gate_name():
            """
            Get the attention scores of nodes in graph(s)
            Args:
                x: torch.Tensor (num_nodes, gnn_output_dim)
                k: torch.Tensor (num_nodes, gnn_output_dim * gnn_num_layers)
                v: torch.Tensor (num_nodes, 1)
            Return: torch.Tensor (num_nodes, 1)
            """
            return self.actor_gate(x, k)[0]
        elif callee == self.actor_xfer_name():
            """
            Get the transformation logits
            Return: torch.Tensor (num_nodes, action_dim)
            """
            return self.actor_xfer(x)
        elif callee == self.critic_name():
            """
            Evaluate nodes in graphs
            Args:
                x: (B, gnn_output_dim)
            Return: torch.Tensor (B, 1)
            """
            return self.critic(x)
        else:
            raise NotImplementedError(f"Unexpected callee name: {callee}")

    @staticmethod
    def gnn_name() -> str:
        return "gnn"

    @staticmethod
    def actor_gate_name() -> str:
        return "actor_gate"

    @staticmethod
    def actor_xfer_name() -> str:
        return "actor_xfer"

    @staticmethod
    def critic_name() -> str:
        return "critic"
