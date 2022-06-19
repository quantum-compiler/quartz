from __future__ import annotations
from dataclasses import dataclass, fields
import os
import random
from typing import Callable, Iterable, Iterator, Tuple, List, Any
import warnings
from collections import deque, namedtuple
from functools import partial
import threading
import time
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.futures import Future
import dgl # type: ignore
import numpy as np
import quartz # type: ignore

import hydra
import wandb

from IPython import embed # type: ignore

def seed_all(seed: int) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class Action:
    node: int
    xfer: int
    
    def to_tensor(self) -> torch.LongTensor:
        return torch.LongTensor([self.node, self.xfer])

@dataclass
class ActionTmp:
    node: int
    xfer_dist: torch.Tensor
@dataclass
class SerializableExperience:
    state: str
    action: Action
    reward: float
    next_state: str
    game_over: bool
    next_nodes: List[int]
    xfer_mask: torch.BoolTensor
    xfer_logprob: float
    
    def __iter__(self) -> Iterator:
        return iter([
            getattr(self, field.name)
            for field in fields(self)
        ])

@dataclass
class BSerializableExperience:
    state: Iterable[str]
    action: Iterable[Action]
    reward: Iterable[float]
    next_state: Iterable[str]
    game_over: Iterable[bool]
    next_nodes: Iterable[List[int]]
    xfer_mask: Iterable[torch.BoolTensor]
    xfer_logprob: Iterable[float]
    
@dataclass
class BatchedExperience:
    state: dgl.graph
    action: torch.LongTensor # (B, 2)
    reward: torch.Tensor # (B,)
    next_state: dgl.graph
    game_over: torch.BoolTensor # (B,)
    next_nodes: List[torch.LongTensor]
    xfer_mask: torch.BoolTensor # (B,)
    xfer_logprob: torch.Tensor # (B,)
    
    @staticmethod
    def new_empty() -> BatchedExperience:
        return BatchedExperience(*[None]*8) # type: ignore

    def __add__(self, other) -> BatchedExperience:
        res = BatchedExperience.new_empty()
        res.state = dgl.batch([self.state, other.state])
        res.next_state = dgl.batch([self.next_state, other.next_state])
        res.action = torch.cat([self.action, other.action]) # type: ignore
        res.reward = torch.cat([self.reward, other.reward])
        res.game_over = torch.cat([self.game_over, other.game_over]) # type: ignore
        res.next_nodes = self.next_nodes + other.next_nodes
        res.xfer_mask = torch.cat([self.xfer_mask, other.xfer_mask]) # type: ignore
        res.xfer_logprob = torch.cat([self.xfer_logprob, other.xfer_logprob])
        return res
    
    def __len__(self) -> int:
        return len(self.next_nodes)

@dataclass
class QuartzInitArgs:
    gate_set: List[str]
    ecc_file_path: str
    no_increase: bool
    include_nop: bool
    

def get_agent_name(agent_id: int) -> str:
    return f'agent_{agent_id}'

def get_obs_name(agent_id: int, obs_id: int) -> str:
    return f'obs_{agent_id}_{obs_id}'

def get_quartz_context(init_args: QuartzInitArgs) -> Tuple[quartz.QuartzContext, quartz.PyQASMParser]:
    quartz_context = quartz.QuartzContext(
        gate_set=init_args.gate_set,
        filename=init_args.ecc_file_path,
        no_increase=init_args.no_increase,
        include_nop=init_args.include_nop,
    )
    quartz_parser = quartz.PyQASMParser(context=quartz_context)
    return quartz_context, quartz_parser

def masked_softmax(logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    logits[~mask] -= 1e10
    return F.softmax(logits, dim=-1)
