from __future__ import annotations
from dataclasses import dataclass, fields
import sys
import psutil
import random
from typing import Tuple, List, Any
import warnings
from collections import deque, namedtuple
from functools import partial
import threading
import time
import datetime
import copy
import itertools
from enum import Enum

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

from ds import *

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
class QuartzInitArgs:
    gate_set: List[str]
    ecc_file_path: str
    no_increase: bool
    include_nop: bool 

class CostType(Enum):
    gate_count = 0
    cx_count = 1
    gate_cx = 2
    
    @staticmethod
    def from_str(s: str) -> CostType:
        if s == 'gate_count':
            return CostType.gate_count
        elif s == 'cx_count':
            return CostType.cx_count
        elif s == 'gate_cx':
            return CostType.gate_cx
        else:
            raise NotImplementedError(f'Unexpected input to CostType {s}')

def get_cost(graph: quartz.PyGraph, tp: CostType) -> int:
    if tp is CostType.gate_count:
        return graph.gate_count
    elif tp is CostType.cx_count:
        return graph.cx_count
    elif tp is CostType.gate_cx:
        return graph.gate_count + 2 * graph.cx_count
    else:
        raise NotImplementedError(f'Unexpected CostType {tp} ({tp.__class__()})')

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

def errprint(s: str, file = sys.stderr) -> None:
    print(s, file=file)

def printfl(s: str) -> None:
    print(s, flush=True)
    
def logprintfl(s: str) -> None:
    print(f'[{datetime.datetime.now()}] {s}', flush=True)

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
