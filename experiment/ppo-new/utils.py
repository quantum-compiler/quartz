from __future__ import annotations

import copy
import datetime
import itertools
import os
import random
import sys
import threading
import time
import warnings
from collections import deque, namedtuple
from dataclasses import dataclass, fields
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Tuple

import dgl  # type: ignore
import numpy as np
import psutil  # type: ignore
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from ds import *
from IPython import embed  # type: ignore
from torch.distributions import Categorical
from torch.futures import Future
from torch.nn.parallel import DistributedDataParallel as DDP

import quartz  # type: ignore


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
class QuartzInitArgs:
    gate_set: List[str]
    ecc_file_path: str
    no_increase: bool
    include_nop: bool


class CostType(Enum):
    gate_count = 0

    cx_count = 1
    cx_gate = 2

    depth = 3
    depth_gc = 4
    depth_2_gc = 5

    @staticmethod
    def from_str(s: str) -> CostType:
        if s == 'gate_count':
            return CostType.gate_count
        elif s == 'cx_count':
            return CostType.cx_count
        elif s == 'cx_gate':
            return CostType.cx_gate
        elif s == 'depth':
            return CostType.depth
        elif s == 'depth_gc':
            return CostType.depth_gc
        elif s == 'depth_2_gc':
            return CostType.depth_2_gc
        else:
            raise NotImplementedError(f'Unexpected input to CostType {s}')


def get_cost(graph: quartz.PyGraph, tp: CostType) -> int:
    if tp is CostType.gate_count:
        return graph.gate_count
    elif tp is CostType.cx_count:
        return graph.cx_count
    elif tp is CostType.cx_gate:
        return 2 * graph.cx_count + graph.gate_count
    elif tp is CostType.depth:
        return graph.depth
    elif tp is CostType.depth_gc:
        return graph.depth + graph.gate_count
    elif tp is CostType.depth_2_gc:
        return 2 * graph.depth + graph.gate_count
    else:
        raise NotImplementedError(f'Unexpected CostType {tp} ({tp.__class__()})')


def get_agent_name(agent_id: int) -> str:
    return f'agent_{agent_id}'


def get_obs_name(agent_id: int, obs_id: int) -> str:
    return f'obs_{agent_id}_{obs_id}'


def get_quartz_context(
    init_args: QuartzInitArgs,
) -> Tuple[quartz.QuartzContext, quartz.PyQASMParser]:
    quartz_context = quartz.QuartzContext(
        gate_set=init_args.gate_set,
        filename=init_args.ecc_file_path,
        no_increase=init_args.no_increase,
        include_nop=init_args.include_nop,
    )
    quartz_parser = quartz.PyQASMParser(context=quartz_context)
    return quartz_context, quartz_parser


def masked_softmax(logits: torch.Tensor, valid_mask: torch.BoolTensor) -> torch.Tensor:
    masked_logits = logits.clone()
    masked_logits[~valid_mask] -= 1e10
    return F.softmax(masked_logits, dim=-1)


def split_reduce_mean(x: torch.Tensor, sizes: torch.LongTensor) -> torch.Tensor:
    """Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html#torch.Tensor.index_reduce_"""
    ind = torch.arange(len(sizes), device=sizes.device).repeat_interleave(sizes)
    reduced = torch.zeros((len(sizes), *x.shape[1:]), dtype=x.dtype, device=x.device)
    reduced.index_add_(0, ind, x)
    return reduced / sizes.unsqueeze(-1)


def errprint(s: str, file=sys.stderr, end='\n') -> None:
    print(s, file=file, end=end)


def printfl(s: str, end='\n') -> None:
    print(s, flush=True, end=end)


def logprintfl(s: str, end='\n') -> None:
    print(f'[{datetime.datetime.now()}] {s}', flush=True, end=end)


def get_time_ns() -> int:
    return time.time_ns()


def dur_ms(t1: int, t2: int) -> float:
    return abs(t2 - t1) / 1e6


def sec_to_hms(sec: float) -> str:
    return str(datetime.timedelta(seconds=sec))


def hms_to_sec(hms: str) -> float:
    return sum(x * float(t) for x, t in zip([3600, 60, 1], hms.split(':')))


def shuffle_lists(*ls):
    zip_ls = list(zip(*ls))
    random.shuffle(zip_ls)
    shuf_lists = map(list, zip(*zip_ls))
    return shuf_lists


def vmem_used_perct() -> float:
    return psutil.virtual_memory().percent


def cur_proc_vmem_perct() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_percent()


def pop_dict_first(d: Dict[Any, Any]) -> Any:
    first_key = next(iter(d))
    return d.pop(first_key)
