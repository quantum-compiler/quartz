from __future__ import annotations

import copy
import datetime
import itertools
import random
import sys
import threading
import time
import warnings
from collections import deque, namedtuple
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import dgl  # type: ignore
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
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


def masked_softmax(logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    logits[~mask] -= 1e10
    return F.softmax(logits, dim=-1)


def errprint(s: str, file=sys.stderr) -> None:
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
