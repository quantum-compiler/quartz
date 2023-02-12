# this file is under mypy's checking
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import threading
from typing import Any, Dict, List, Tuple

import dgl  # type: ignore
import qtz
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ds import *
from icecream import ic  # type: ignore
from IPython import embed  # type: ignore
from model.actor_critic import ActorCritic
from omegaconf.dictconfig import DictConfig
from torch.distributions import Categorical
from torch.futures import Future
from utils import *

# import quartz # type: ignore


class Searcher:
    def __init__(
        self,
        searcher_id: int,
        num_searchers: int,
        device: torch.device,
        batch_inference: bool,
        invalid_reward: float,
        limit_total_gate_count: bool,
        cost_type: CostType,
        ac_net: ActorCritic,
        input_graphs: List[Dict[str, str]],
        softmax_temp_en: bool,
        hit_rate: float,
        dyn_eps_len: bool,
        max_eps_len: int,
        min_eps_len: int,
        subgraph_opt: bool,
        output_full_seq: bool,
        output_dir: str,
    ) -> None:
        self.id = searcher_id
        self.num_searchers = num_searchers
        self.device = device
        self.output_full_seq = output_full_seq
        self.output_dir = output_dir

        self.dyn_eps_len = dyn_eps_len
        self.max_eps_len = max_eps_len
        self.min_eps_len = min_eps_len
        self.cost_type = cost_type
        self.invalid_reward = invalid_reward
        self.limit_total_gate_count = limit_total_gate_count
        self.subgraph_opt = subgraph_opt

        """networks related"""
        self.ac_net = ac_net  # NOTE: just a ref
        self.softmax_temp_en = softmax_temp_en
        self.hit_rate = hit_rate

        self.graph_buffers: List[GraphBuffer] = [
            GraphBuffer(
                input_graph['name'],
                input_graph['qasm'],
                self.cost_type,
                self.device,
            )
            for input_graph in input_graphs
        ]
        self.init_buffer_turn: int = 0

    def search(self):
        pass
