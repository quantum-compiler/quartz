"""data structures"""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Callable, Iterable, Iterator, Optional, Tuple, List, Any, Sequence

import torch
import dgl # type: ignore
import quartz # type: ignore
from IPython import embed # type: ignore

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
class Experience:
    state: quartz.PyGraph
    action: Action
    reward: float
    next_state: quartz.PyGraph
    game_over: bool
    next_nodes: List[int]
    xfer_mask: torch.BoolTensor
    xfer_logprob: float
    info: Any
    
    def __iter__(self) -> Iterator:
        return iter([
            getattr(self, field.name)
            for field in fields(self)
        ])
    
    @staticmethod
    def new_empty() -> Experience:
        return Experience(*[None]*len(fields(Experience))) # type:ignore
    
    def __str__(self) -> str:
        s = f'{self.state} (gate_count = {self.state.gate_count}) ' \
            f'{self.next_state} (gate_count = {self.next_state.gate_count}) \n' \
            f'{self.action}  reward = {self.reward}  game_over = {self.game_over}  ' \
            f'next_nodes = {self.next_nodes}  \n' \
            f'xfer_mask = {self.xfer_mask}  xfer_logprob = {self.xfer_logprob}  info = {self.info}'
        return s

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
    info: Any
    
    def __iter__(self) -> Iterator:
        return iter([
            getattr(self, field.name)
            for field in fields(self)
        ])
    
    @staticmethod
    def new_empty() -> SerializableExperience:
        return SerializableExperience(*[None]*len(fields(SerializableExperience))) # type:ignore

@dataclass
class ExperienceList:
    state: List[str | dgl.graph]
    action: List[Action]
    reward: List[float]
    next_state: List[str | dgl.graph]
    game_over: List[bool]
    next_nodes: List[List[int]]
    xfer_mask: List[torch.BoolTensor]
    xfer_logprob: List[float]
    info: List[Any]
    
    def __len__(self) -> int:
        return len(self.state)
    
    def __add__(self, other) -> ExperienceList:
        ret = ExperienceList.new_empty()
        for field in fields(self):
            setattr(ret, field.name, getattr(self, field.name) + getattr(other, field.name))
        return ret

    def __iadd__(self, other) -> ExperienceList:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name).__iadd__(getattr(other, field.name)))
        return self
    
    @staticmethod
    def new_empty() -> ExperienceList:
        return ExperienceList(*[None]*len(fields(ExperienceList))) # type: ignore

    def get_batch(
        self,
        start_pos: int = 0,
        batch_size: int = 1,
        device: torch.device = torch.device('cpu'),
    ) -> BatchedExperience:
        exps = BatchedExperience.new_empty()
        if start_pos < len(self):
            sc = slice(start_pos, start_pos + batch_size)
            exps.state = dgl.batch(self.state[sc]).to(device)
            exps.next_state = dgl.batch(self.next_state[sc]).to(device)
            exps.action = torch.stack([a.to_tensor() for a in self.action[sc]]).to(device) # type: ignore
            exps.reward = torch.Tensor(self.reward[sc]).to(device)
            exps.game_over = torch.BoolTensor(self.game_over[sc]).to(device) # type: ignore
            exps.next_nodes = [ torch.LongTensor(ns).to(device) for ns in self.next_nodes[sc] ] # type: ignore
            exps.xfer_mask = torch.stack(self.xfer_mask[sc]).to(device) # type: ignore
            exps.xfer_logprob = torch.Tensor(self.xfer_logprob[sc]).to(device)
        
        return exps

class ExperienceListIterator:

    def __init__(self, src: ExperienceList, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        self.src = src
        self.batch_size = batch_size
        self.device = device
        self.start_pos = 0
    
    def __iter__(self):
        return self

    def __next__(self) -> BatchedExperience:
        if self.start_pos < len(self.src):
            ret = self.src.get_batch(self.start_pos, self.batch_size, self.device)
            self.start_pos += self.batch_size
            return ret
        else:
            raise StopIteration

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
        return BatchedExperience(*[[]]*len(fields(BatchedExperience))) # type: ignore

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
