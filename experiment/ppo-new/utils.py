from __future__ import annotations
import os
import random
from typing import Callable, Tuple, List, Any
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

Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'game_over'],
)

SerializableExperience = namedtuple(
    'SerializableExperience',
    ['state', 'action', 'reward', 'next_state', 'game_over'],
)

QuartzInitArgs = namedtuple(
    'QuartzInitArgs',
    ['gate_set', 'ecc_file_path', 'no_increase', 'include_nop'],
)

Action = namedtuple(
    'Action',
    ['node', 'xfer'],
)

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
