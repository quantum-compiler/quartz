# this file is under mypy's checking
"""data structures"""
from __future__ import annotations

import gc
import heapq
import itertools
import math
import random
import sys
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterator, List, Set, Tuple, TypeVar, cast

import dgl  # type: ignore
import qtz
import torch
from IPython import embed  # type: ignore
from sortedcontainers import SortedDict  # type: ignore
from utils import *

import quartz  # type: ignore


@dataclass
class Action:
    node: int
    xfer: int

    def to_tensor(self) -> torch.LongTensor:
        return torch.LongTensor([self.node, self.xfer])


@dataclass
class ActionTmp:
    node: int
    node_value: float
    xfer_dist: torch.Tensor


@dataclass
class Experience:
    state: quartz.PyGraph
    action: Action
    reward: float
    next_state: quartz.PyGraph
    game_over: bool
    node_value: float
    next_nodes: List[int]
    xfer_mask: torch.BoolTensor
    xfer_logprob: float
    info: Any

    def __iter__(self) -> Iterator:
        return iter([getattr(self, field.name) for field in fields(self)])

    @staticmethod
    def new_empty() -> Experience:
        return Experience(*[None] * len(fields(Experience)))  # type:ignore

    def __str__(self) -> str:
        s = (
            f'{self.state} (gate_count = {self.state.gate_count}) '
            f'{self.next_state} (gate_count = {self.next_state.gate_count}) \n'
            f'{self.action}  reward = {self.reward}  game_over = {self.game_over}  '
            f'next_nodes = {self.next_nodes}  \n'
            f'xfer_mask = {self.xfer_mask}  xfer_logprob = {self.xfer_logprob}  info = {self.info}'
        )
        return s


@dataclass
class SerializableExperience:
    state: str
    action: Action
    reward: float
    next_state: str
    game_over: bool
    node_value: float
    next_nodes: List[int]
    xfer_mask: torch.BoolTensor
    xfer_logprob: float
    info: Any

    def __iter__(self) -> Iterator:
        return iter([getattr(self, field.name) for field in fields(self)])

    @staticmethod
    def new_empty() -> SerializableExperience:
        return SerializableExperience(
            *[None] * len(fields(SerializableExperience))
        )  # type:ignore


@dataclass
class ExperienceList:
    state: List[str | dgl.DGLGraph]
    action: List[Action]
    reward: List[float]
    next_state: List[str | dgl.DGLGraph]
    game_over: List[bool]
    node_value: List[float]
    next_nodes: List[List[int]]
    xfer_mask: List[torch.BoolTensor]
    xfer_logprob: List[float]
    info: List[Any]

    def __len__(self) -> int:
        return len(self.state)

    def __iter__(self) -> Iterator:
        return iter([getattr(self, field.name) for field in fields(self)])

    def items(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}.items()

    def __add__(self, other) -> ExperienceList:
        ret = ExperienceList.new_empty()
        for field in fields(self):
            setattr(
                ret, field.name, getattr(self, field.name) + getattr(other, field.name)
            )
        return ret

    def __iadd__(self, other) -> ExperienceList:
        for field in fields(self):
            setattr(
                self,
                field.name,
                getattr(self, field.name).__iadd__(getattr(other, field.name)),
            )
        return self

    @staticmethod
    def new_empty() -> ExperienceList:
        return ExperienceList(*[[] for _ in range(len(fields(ExperienceList)))])  # type: ignore

    def sanity_check(self) -> None:
        for name, field in self.items():
            assert len(field) == len(
                self
            ), f'{len(name)} = len({name}) != len(self) = {len(self)})'

    def shuffle(self) -> None:
        (
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.game_over,
            self.node_value,
            self.next_nodes,
            self.xfer_mask,
            self.xfer_logprob,
            self.info,
        ) = shuffle_lists(
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.game_over,
            self.node_value,
            self.next_nodes,
            self.xfer_mask,
            self.xfer_logprob,
            self.info,
        )

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
            exps.action = torch.stack([a.to_tensor() for a in self.action[sc]]).to(device)  # type: ignore
            exps.reward = torch.Tensor(self.reward[sc]).to(device)
            exps.game_over = torch.BoolTensor(self.game_over[sc]).to(device)  # type: ignore
            exps.node_value = torch.Tensor(self.node_value[sc]).to(device)
            exps.next_nodes = [torch.LongTensor(ns).to(device) for ns in self.next_nodes[sc]]  # type: ignore
            exps.xfer_mask = torch.stack(self.xfer_mask[sc]).to(device)  # type: ignore
            exps.xfer_logprob = torch.Tensor(self.xfer_logprob[sc]).to(device)

        return exps


@dataclass
class TrainExpList(ExperienceList):
    target_values: List[float]
    advantages: List[float]

    @staticmethod
    def new_empty() -> TrainExpList:
        return TrainExpList(*[None] * len(fields(TrainExpList)))  # type: ignore

    def shuffle(self) -> None:
        (
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.game_over,
            self.node_value,
            self.next_nodes,
            self.xfer_mask,
            self.xfer_logprob,
            self.info,
            self.target_values,
            self.advantages,
        ) = shuffle_lists(
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.game_over,
            self.node_value,
            self.next_nodes,
            self.xfer_mask,
            self.xfer_logprob,
            self.info,
            self.target_values,
            self.advantages,
        )

    def get_batch(
        self,
        start_pos: int = 0,
        batch_size: int = 1,
        device: torch.device = torch.device('cpu'),
    ) -> TrainBatchExp:
        exps = TrainBatchExp.new_empty()
        if start_pos < len(self):
            sc = slice(start_pos, start_pos + batch_size)
            exps.state = dgl.batch(self.state[sc]).to(device)
            exps.next_state = dgl.batch(self.next_state[sc]).to(device)
            exps.action = torch.stack([a.to_tensor() for a in self.action[sc]]).to(device)  # type: ignore
            exps.reward = torch.Tensor(self.reward[sc]).to(device)
            exps.game_over = torch.BoolTensor(self.game_over[sc]).to(device)  # type: ignore
            exps.node_value = torch.Tensor(self.node_value[sc]).to(device)
            exps.next_nodes = [torch.LongTensor(ns).to(device) for ns in self.next_nodes[sc]]  # type: ignore
            exps.xfer_mask = torch.stack(self.xfer_mask[sc]).to(device)  # type: ignore
            exps.xfer_logprob = torch.Tensor(self.xfer_logprob[sc]).to(device)
            exps.target_values = torch.Tensor(self.target_values[sc]).to(device)
            exps.advantages = torch.Tensor(self.advantages[sc]).to(device)
        return exps


@dataclass
class BatchedExperience:
    state: dgl.DGLGraph
    action: torch.LongTensor  # (B, 2)
    reward: torch.Tensor  # (B,)
    next_state: dgl.DGLGraph
    game_over: torch.BoolTensor  # (B,)
    node_value: torch.Tensor  # (B,)
    next_nodes: List[torch.LongTensor]
    xfer_mask: torch.BoolTensor  # (B,)
    xfer_logprob: torch.Tensor  # (B,)

    @staticmethod
    def new_empty() -> BatchedExperience:
        return BatchedExperience(*[None for _ in range(len(fields(BatchedExperience)))])  # type: ignore

    def __add__(self, other: BatchedExperience) -> BatchedExperience:
        res = BatchedExperience.new_empty()
        res.state = dgl.batch([self.state, other.state])
        res.next_state = dgl.batch([self.next_state, other.next_state])
        res.action = torch.cat([self.action, other.action])  # type: ignore
        res.reward = torch.cat([self.reward, other.reward])
        res.game_over = torch.cat([self.game_over, other.game_over])  # type: ignore
        res.node_value = torch.cat([self.node_value, other.node_value])
        res.next_nodes = self.next_nodes + other.next_nodes
        res.xfer_mask = torch.cat([self.xfer_mask, other.xfer_mask])  # type: ignore
        res.xfer_logprob = torch.cat([self.xfer_logprob, other.xfer_logprob])
        return res

    def __len__(self) -> int:
        return len(self.next_nodes)


class TrainBatchExp(BatchedExperience):
    target_values: torch.Tensor
    advantages: torch.Tensor

    @staticmethod
    def new_empty() -> TrainBatchExp:
        return TrainBatchExp(*[None for _ in range(len(fields(TrainBatchExp)))])  # type: ignore


ExpListType = TypeVar('ExpListType', bound=ExperienceList)
BatchExpType = TypeVar('BatchExpType', bound=BatchedExperience)


class ExperienceListIterator:
    def __init__(
        self,
        src: ExpListType,
        batch_size: int = 1,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.src = src
        self.batch_size = batch_size
        self.device = device
        self.start_pos = 0

    def __iter__(self) -> ExperienceListIterator:
        return self

    def __next__(self) -> BatchExpType:
        if self.start_pos < len(self.src):
            ret = self.src.get_batch(self.start_pos, self.batch_size, self.device)
            ret = cast(BatchExpType, ret)
            self.start_pos += self.batch_size
            return ret
        else:
            raise StopIteration


@dataclass
class AllGraphDictValue:
    dist: int
    cost: int
    pre_graph: quartz.PyGraph
    action: Action


class GraphBuffer:
    """store PyGraph of a class of circuits for init state sampling and maintain some other infos"""

    def __init__(
        self,
        name: str,
        original_graph_qasm: str,
        cost_type: CostType,
        device: torch.device = torch.device('cpu'),
        max_len: int | float = math.inf,
    ) -> None:
        self.name = name
        self.max_len = max_len
        self.device = device
        self.cost_type = cost_type
        self.original_graph = qtz.qasm_to_graph(original_graph_qasm)
        self.original_cost = get_cost(self.original_graph, self.cost_type)

        self.cost_to_graph: SortedDict[int, List[quartz.PyGraph]] = SortedDict(
            {
                get_cost(self.original_graph, cost_type): [
                    self.original_graph,
                ],
            }
        )
        self.hashset: Set[int] = {hash(self.original_graph)}

        """other infos"""
        self.best_graph = self.original_graph

        self.eps_lengths: List[int] = []
        self.max_eps_length: int = 0
        self.rewards: List[List[float]] = []

        self.init_graph_gcs: List[int] = []
        self.graph_gcs: List[int] = []
        self.init_graph_ccs: List[int] = []
        self.graph_ccs: List[int] = []
        self.init_graph_costs: List[int] = []
        self.graph_costs: List[int] = []
        self.init_graph_depths: List[int] = []
        self.graph_depths: List[int] = []

        self.all_graphs: Dict[quartz.PyGraph, AllGraphDictValue] = {
            self.original_graph: AllGraphDictValue(
                0, self.original_cost, None, Action(0, 0)
            ),
        }

    def __len__(self) -> int:
        return len(self.hashset)

    def prepare_for_next_iter(self) -> None:
        self.eps_lengths.clear()
        self.rewards.clear()

        self.init_graph_gcs.clear()
        self.graph_gcs.clear()
        self.init_graph_ccs.clear()
        self.graph_ccs.clear()
        self.init_graph_costs.clear()
        self.graph_costs.clear()

        self.shrink()

    def shrink(self) -> None:
        mem_perct_th = 82.5
        vmem_perct = vmem_used_perct()
        old_len = len(self)
        if vmem_perct > mem_perct_th:
            if self.max_len == math.inf:  # the first time mem usage exceeds threshold
                self.max_len = old_len  # don't use more memory

        if old_len > self.max_len:
            printfl(f'Buffer {self.name} starts to shrink.')
            while len(self) > self.max_len:
                self.pop_some(len(self) - int(self.max_len))
            printfl(
                f'Buffer {self.name} shrinked from {old_len} to {len(self)}. (Mem: {vmem_perct} % -> {vmem_used_perct()} %).'
            )

        if vmem_used_perct() > 95.0:
            raise MemoryError(
                f'Used {vmem_used_perct()} % memory. Exit to avoid system crash.'
            )

    def push_back(self, graph: quartz.PyGraph, hash_value: int = None) -> bool:
        if hash_value is None:
            hash_value = hash(graph)
        if hash_value not in self.hashset:
            self.hashset.add(hash_value)
            gcost = get_cost(graph, self.cost_type)
            graphs: List[quartz.PyGraph]
            if gcost not in self.cost_to_graph:
                graphs = []
                self.cost_to_graph[gcost] = graphs
            else:
                graphs = self.cost_to_graph[gcost]
            graphs.append(graph)
            idx_to_pop = 0 if gcost != self.original_cost else 1
            while len(graphs) > int(500):  # NOTE: limit num of graphs of each kind
                popped_graph = graphs.pop(idx_to_pop)
                self.hashset.remove(hash(popped_graph))
            # while len(self) > self.max_len:
            #     assert self.pop_one(graph_to_remain=graph) is not graph
            # assert hash_value in self.hashset
            return True
        else:
            return False

    def pop_one(self, graph_to_remain: quartz.PyGraph = None) -> quartz.PyGraph | None:
        if len(self) > 0:
            max_key_idx: int = -1
            while True:
                max_key, graphs = self.cost_to_graph.peekitem(max_key_idx)
                idx_to_pop: int = 0 if max_key != self.original_cost else 1
                while idx_to_pop < len(graphs):
                    if graphs[idx_to_pop] is graph_to_remain:
                        idx_to_pop += 1
                    else:
                        break
                if idx_to_pop < len(graphs):
                    popped_graph = graphs.pop(idx_to_pop)
                    # assert popped_graph is not graph_to_remain, f'idx_to_pop = {idx_to_pop}'
                    self.hashset.remove(hash(popped_graph))
                    if len(graphs) == 0:
                        self.cost_to_graph.pop(max_key)
                    return popped_graph
                if len(graphs) > 0:
                    max_key_idx -= 1
            # end while
        return None

    def pop_some(self, num: int) -> None:
        if len(self) > 0:
            max_key_idx: int = -1
            while True:
                max_key, graphs = self.cost_to_graph.peekitem(max_key_idx)
                idx_to_pop = 0 if max_key != self.original_cost else 1
                while idx_to_pop < len(graphs) and num > 0:
                    popped_graph = graphs.pop(idx_to_pop)
                    self.hashset.remove(hash(popped_graph))
                    num -= 1
                if len(graphs) == 0:
                    self.cost_to_graph.pop(max_key)
                if num <= 0:
                    break
                elif len(graphs) > 0:
                    max_key_idx -= 1
            # end while

    def sample(self, greedy: bool) -> quartz.PyGraph:
        gcost_list = list(self.cost_to_graph.keys())
        gcost = torch.Tensor(gcost_list).to(self.device)
        if greedy:
            weights = 1 / (gcost - gcost.min() + 0.2)
        else:
            weights = 1 / gcost**4
        sampled_gcost_idx = int(torch.multinomial(weights, num_samples=1))
        sampled_gcost = gcost_list[sampled_gcost_idx]
        graphs = self.cost_to_graph[sampled_gcost]
        if greedy:
            graph_weights = torch.linspace(0.6, 1.000001, len(graphs)).to(self.device)
            sampled_graph_idx = int(torch.multinomial(graph_weights, num_samples=1))
            sampled_graph = graphs[sampled_graph_idx]
        else:
            sampled_graph = random.choice(graphs)
        return sampled_graph

    """Note that it's not concurrency-safe to call these functions."""

    def push_nonexist_best(self, qasm: str) -> bool:
        graph = qtz.qasm_to_graph(qasm)
        if self.push_back(graph):  # non-exist
            """update best graph and return whether the best info is updated"""
            if get_cost(graph, self.cost_type) < get_cost(
                self.best_graph, self.cost_type
            ):
                self.best_graph = graph
                return True
        # end if
        return False

    def append_costs_from_graph(self, graph: quartz.PyGraph):
        self.graph_gcs.append(graph.gate_count)
        self.graph_ccs.append(graph.cx_count)
        self.graph_depths.append(graph.depth)
        self.graph_costs.append(get_cost(graph, self.cost_type))

    def append_init_costs_from_graph(self, graph: quartz.PyGraph):
        self.init_graph_gcs.append(graph.gate_count)
        self.init_graph_ccs.append(graph.cx_count)
        self.init_graph_depths.append(graph.depth)
        self.init_graph_costs.append(get_cost(graph, self.cost_type))

    def eps_len_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {}
        max_eps_len = max(self.eps_lengths)
        info[f'min_epslen'] = min(self.eps_lengths)
        info[f'max_epslen'] = max_eps_len
        info[f'mean_epslen'] = sum(self.eps_lengths) / len(self.eps_lengths)

        self.max_eps_length = max(self.max_eps_length, max_eps_len)
        info[f'max_epslen_global'] = self.max_eps_length

        return info

    def rewards_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {}
        max_eps_reward: float = -math.inf
        mean_eps_reward: float = 0.0
        for eps_rewards in self.rewards:
            # assert len(eps_rewards) > 0
            eps_sum = sum(eps_rewards)
            max_eps_reward = max(max_eps_reward, eps_sum)
            mean_eps_reward += eps_sum / len(self.rewards)

        all_rewards = list(itertools.chain(*self.rewards))

        info['max_eps_reward'] = max_eps_reward
        info['mean_eps_reward'] = mean_eps_reward
        info['mean_exp_reward'] = sum(all_rewards) / len(all_rewards)

        return info

    def cost_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {}

        for name, values, init_values in [
            ('gate_count', self.graph_gcs, self.init_graph_gcs),
            ('cx_count', self.graph_ccs, self.init_graph_ccs),
            ('depth', self.graph_depths, self.init_graph_depths),
            ('cost', self.graph_costs, self.init_graph_costs),
        ]:
            info[f'min_init_{name}'] = min(init_values)
            info[f'max_init_{name}'] = max(init_values)
            info[f'mean_init_{name}'] = sum(init_values) / len(init_values)

            info[f'min_{name}_iter'] = min(values)
            info[f'max_{name}_iter'] = max(values)
            info[f'mean_{name}_iter'] = sum(values) / len(values)

        return info

    def basic_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {
            'buffer_size': len(self),
            'diff_costs': len(self.cost_to_graph),
            'min_cost': self.cost_to_graph.peekitem(0)[0],
            'max_cost': self.cost_to_graph.peekitem(-1)[0],
        }
        return info

    def push_back_all_graphs(
        self,
        graph: quartz.PyGraph,
        cost: int,
        pre_graph: quartz.PyGraph,
        action: Action,
    ) -> None:
        # assert pre_graph in self.all_graphs
        dist = 1 + self.all_graphs[pre_graph].dist
        pre_cand = self.all_graphs.pop(
            graph, AllGraphDictValue(math.inf, None, None, None)
        )
        if dist < pre_cand.dist + 1:
            pre_cand = AllGraphDictValue(dist, cost, pre_graph, action)
            # NOTE: action here is how this graph is got from pre_graph
        self.all_graphs[graph] = pre_cand
