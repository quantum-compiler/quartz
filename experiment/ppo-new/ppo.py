# this file is under mypy's checking
from __future__ import annotations
import os
import sys
import random
from typing import Callable, Set, Tuple, List, Dict, Any
import warnings
from collections import deque, namedtuple
from functools import partial
import threading
import time
import copy
import itertools
import math
from tqdm import tqdm # type: ignore

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

from ds import *
from utils import *
from model import ActorCritic
from IPython import embed # type: ignore
from icecream import ic # type: ignore

DDP_PORT = int(23333)
RPC_PORT = DDP_PORT + 1

"""global vars"""
quartz_context: quartz.QuartzContext
quartz_parser: quartz.PyQASMParser

def init_quartz_context(
    gate_set: List[str],
    ecc_file_path: str,
    no_increase: bool,
    include_nop: bool,
) -> None:
    global quartz_context
    global quartz_parser
    quartz_context = quartz.QuartzContext(
        gate_set=gate_set,
        filename=ecc_file_path,
        no_increase=no_increase,
        include_nop=include_nop,
    )
    quartz_parser = quartz.PyQASMParser(context=quartz_context)

def qasm_to_graph(qasm_str: str) -> quartz.PyGraph:
    global quartz_context
    global quartz_parser
    dag = quartz_parser.load_qasm_str(qasm_str)
    graph = quartz.PyGraph(context=quartz_context, dag=dag)
    return graph

def is_nop(xfer_id: int) -> bool:
    global quartz_context
    return quartz_context.get_xfer_from_id(id=xfer_id).is_nop

class Observer:
    
    def __init__(
        self,
        obs_id: int,
        batch_inference: bool,
        invalid_reward: float,
    ) -> None:
        self.id = obs_id
        self.batch_inference = batch_inference
        self.invalid_reward = invalid_reward
    
    def run_episode(
        self,
        agent_rref: rpc.RRef[PPOAgent],
        init_state_str: str,
        len_episode: int,
        max_gate_count_ratio: float,
        nop_stop: bool,
    ) -> List[SerializableExperience]:
        """Interact with env for many steps to collect data.
        If `batch_inference` is `True`, we call `select_action_batch` of the agent,
        so we have to run a fixed number of steps.
        If an episode stops in advance, we start a new one to continue running.
        
        If `batch_inference` is `False`, we can run just one episode.

        Returns:
            List[SerializableExperience]: a list of experiences collected in this function
        """
        init_graph = qasm_to_graph(init_state_str)
        exp_list: List[SerializableExperience] = []
        
        gs_time = get_time_ns()
        graph = init_graph
        graph_str = init_state_str
        info: Dict[str, Any] = { 'start': True }
        for i_step in range(len_episode):
            # print(f'obs {self.id} step {i_step}')
            """get action (action_node, xfer_dist) from agent"""
            s_time = get_time_ns()
            _action: ActionTmp
            if self.batch_inference:
                _action = agent_rref.rpc_sync(timeout=0).select_action_batch(
                    self.id, graph.to_qasm_str(),
                )
            else:
                _action = agent_rref.rpc_sync().select_action(
                    self.id, graph.to_qasm_str(),
                ) # NOTE not sure if it's OK because `select_action` doesn't return a `Future`
            # e_time = get_time_ns()
            # errprint(f'    Obs {self.id} : Action got in {dur_ms(e_time, s_time)} ms.')
            """sample action_xfer with mask"""
            av_xfers = graph.available_xfers_parallel(
                context=quartz_context, node=graph.get_node_from_id(id=_action.node))
            av_xfer_mask = torch.BoolTensor([0] * quartz_context.num_xfers)
            av_xfer_mask[av_xfers] = True
            # (action_dim, )  only sample from available xfers
            softmax_xfer = masked_softmax(_action.xfer_dist, av_xfer_mask)
            # action_xfer = torch.multinomial(softmax_xfer, num_samples=1)
            xfer_dist = Categorical(softmax_xfer)
            action_xfer = xfer_dist.sample()
            action_xfer_logp: torch.Tensor = xfer_dist.log_prob(action_xfer)
            
            """apply action"""
            action = Action(_action.node, action_xfer.item())
            next_nodes: List[int]
            next_graph, next_nodes = \
                graph.apply_xfer_with_local_state_tracking(
                    xfer=quartz_context.get_xfer_from_id(id=action.xfer),
                    node=graph.get_node_from_id(id=action.node)
                )
            """parse result, compute reward"""
            if next_graph is None:
                reward = self.invalid_reward
                game_over = True
                next_graph_str = graph_str # CONFIRM placeholder?
            elif is_nop(action.xfer):
                reward = 0
                game_over = nop_stop
                next_graph_str = graph_str # unchanged
                next_nodes = [action.node] # CONFIRM
            else:
                reward = graph.gate_count - next_graph.gate_count
                game_over = (next_graph.gate_count > init_graph.gate_count * max_gate_count_ratio)
                next_graph_str = next_graph.to_qasm_str()
            
            exp = SerializableExperience(
                graph_str, action, reward, next_graph_str, game_over,
                next_nodes, av_xfer_mask, action_xfer_logp.item(), copy.deepcopy(info),
            )
            exp_list.append(exp)
            info['start'] = False
            # s_time = get_time_ns()
            # errprint(f'    Obs {self.id} : Action applied in {dur_ms(e_time, s_time)} ms.')
            if game_over:
                if self.batch_inference:
                    graph = init_graph
                    graph_str = init_state_str
                    info['start'] = True
                else:
                    break
            else:
                graph = next_graph
                graph_str = next_graph_str
        # end for
        # ge_time = get_time_ns()
        # errprint(f'    Obs {self.id} : Trajectory finished in {dur_ms(ge_time, gs_time)} ms.')
        return exp_list

class GraphBuffer:
    """store PyGraph of a class of circuits for init state sampling and best info maintenance"""
    def __init__(
        self,
        name: str,
        original_graph_qasm: str,
        max_len: int | float,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.name = name
        self.max_len = max_len
        self.device = device
        self.original_graph = qasm_to_graph(original_graph_qasm)
        
        self.buffer: List[quartz.PyGraph] = []
        self.hashset: Set[int] = { hash(self.original_graph) }
        self.gate_counts: torch.Tensor = torch.Tensor([]).to(self.device)
        """other infos"""
        self.best_graph = self.original_graph
        self.traj_lengths: List[int] = []
    
    def __len__(self) -> int:
        return len(self.buffer) + 1
        
    def push_back(self, graph: quartz.PyGraph) -> None:
        hash_value = hash(graph)
        if hash_value not in self.hashset:
            self.hashset.add(hash_value)
            self.buffer.append(graph)
            self.gate_counts = torch.cat([
                self.gate_counts,
                torch.Tensor([graph.gate_count]).to(self.device)
            ])
            if len(self) > self.max_len:
                graph_to_remove = self.buffer.pop(0) # remove and return
                self.hashset.remove(hash(graph_to_remove))
                self.gate_counts = self.gate_counts[1:]                
        
    def sample(self) -> quartz.PyGraph:
        gate_counts = torch.cat([
            self.gate_counts,
            torch.Tensor([self.original_graph.gate_count]).to(self.device),
        ]) # always has the probability to start from the original graph
        weights = 1 / gate_counts ** 4
        sampled_idx = int(torch.multinomial(weights, num_samples=1))
        sampled_graph: quartz.PyGraph
        if sampled_idx == len(self.buffer):
            sampled_graph = self.original_graph
        else:
            sampled_graph = self.buffer[sampled_idx]
        return sampled_graph

class PPOAgent:
    
    def __init__(
        self,
        agent_id: int,
        num_observers: int,
        device: torch.device,
        batch_inference: bool,
        invalid_reward: float,
        ac_net: ActorCritic,
        input_graphs: List[Dict[str, str]],
        softmax_hit_rate: float,
        output_dir: str,
    ) -> None:
        self.id = agent_id
        self.device = device
        self.output_dir = output_dir
        """networks related"""
        self.ac_net = ac_net # NOTE: just a ref
        self.softmax_hit_rate = softmax_hit_rate
        
        """init Observers on the other processes and hold the refs to them"""
        self.obs_rrefs: List[rpc.RRef] = []
        for obs_rank in range(0, num_observers):
            ob_info = rpc.get_worker_info(get_obs_name(self.id, obs_rank))
            self.obs_rrefs.append(
                rpc.remote(ob_info, Observer, args=(obs_rank, batch_inference, invalid_reward,))
            )
        
        self.graph_buffers: List[GraphBuffer] = [
            GraphBuffer(
                input_graph['name'], input_graph['qasm'],
                math.inf, self.device
            ) for input_graph in input_graphs
        ]
        self.init_buffer_turn: int = 0

        """helper vars for select_action"""
        self.future_actions: Future[List[ActionTmp]] = Future()
        self.pending_states = len(self.obs_rrefs)
        self.states_buf: List[dgl.graph] = [None] * len(self.obs_rrefs)
        self.lock = threading.Lock()
        
            
    @torch.no_grad()
    def select_action(self, obs_id: int, state_str: str) -> Action:
        """respond to a single query"""
        pygraph: quartz.PyGraph = qasm_to_graph(state_str)
        dgl_graph: dgl.graph = pygraph.to_dgl_graph().to(self.device)
        
        # TODO inference on dgl_graph to get action
        action_node = 0
        action_xfer = 0
        
        return Action(action_node, action_xfer)
        
    @rpc.functions.async_execution
    @torch.no_grad()
    def select_action_batch(self, obs_id: int, state_str: str) -> Future[ActionTmp]:
        """inference a batch of queries queried by all of the observers at once"""
        future_action: Future[ActionTmp] = self.future_actions.then(
            lambda future_actions: future_actions.wait()[obs_id]
        ) # this single action is returned for the obs that calls this function
        # It is available after self.future_actions is set
        pygraph: quartz.PyGraph = qasm_to_graph(state_str)
        dgl_graph: dgl.graph = pygraph.to_dgl_graph().to(self.device)
        if self.states_buf[obs_id] is None:
            self.states_buf[obs_id] = dgl_graph
        else:
            raise Exception(f'Unexpected: self.states_buf[{obs_id}] is not None! Duplicated assignment occurs!')
        
        with self.lock: # avoid data race on self.pending_states
            self.pending_states -= 1
            # errprint(f'    Agent {self.id} : Obs {obs_id} requested.')
            if self.pending_states == 0:
                # errprint(f'    Agent {self.id} : Obs {obs_id} requested. Start inference.')
                # s_time = get_time_ns()
                """collected a batch, start batch inference"""
                b_state: dgl.graph = dgl.batch(self.states_buf)
                num_nodes: torch.Tensor = b_state.batch_num_nodes() # (num_graphs, ) assert each elem > 0
                """compute embeds and use Critic to evaluate each node"""
                # (batch_num_nodes, embed_dim)
                # TODO check whether this ac_net is updated
                b_node_embeds: torch.Tensor = self.ac_net.graph_embedding(b_state)
                # (batch_num_nodes, )
                b_node_values: torch.Tensor = self.ac_net.critic(b_node_embeds).squeeze()
                # list with length num_graphs; each member is a tensor of node values in a graph
                node_values_list: List[torch.Tensor] = torch.split(b_node_values, num_nodes.tolist())
                """sample node by softmax with temperature for each graph as a batch"""
                # (num_graphs, max_num_nodes)
                b_node_values_pad = nn.utils.rnn.pad_sequence(
                    node_values_list, batch_first=True, padding_value=-torch.inf)
                # (num_graphs, )
                temperature = 1 / (torch.log( self.softmax_hit_rate * (num_nodes - 1)/(1 - self.softmax_hit_rate) ))
                b_softmax_node_values_pad = F.softmax(b_node_values_pad / temperature.unsqueeze(1), dim=-1)
                b_sampled_nodes = torch.multinomial(b_softmax_node_values_pad, 1).flatten()
                """collect embeddings of sampled nodes"""
                # (num_graphs, )
                node_offsets = torch.zeros(b_sampled_nodes.shape[0], dtype=torch.long).to(self.device)
                node_offsets[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
                sampled_node_ids = b_sampled_nodes + node_offsets
                # (num_graphs, embed_dim)
                sampled_node_embeds = b_node_embeds[sampled_node_ids]
                """use Actor to evaluate xfers for sampled nodes"""
                # (num_graphs, action_dim)
                xfer_logits: torch.Tensor = self.ac_net.actor(sampled_node_embeds).cpu()
                # return the xfer dist. to observers who are responsible for sample xfer with masks
                actions = [
                    ActionTmp(int(b_sampled_nodes[i]), xfer_logits[i])
                    for i in range(len(self.obs_rrefs))
                ]
                self.future_actions.set_result(actions)
                # e_time = get_time_ns()
                # errprint(f'    Agent {self.id} : Obs {obs_id} requested. Finished inference in {dur_ms(e_time, s_time)} ms.')
                """re-init"""
                self.pending_states = len(self.obs_rrefs)
                self.future_actions = torch.futures.Future()
                self.states_buf = [None] * len(self.obs_rrefs)
        return future_action # return a future
    
    def other_info_dict(self) -> Dict[str, float]:
        info_dict: Dict[str, float] = {}
        for buffer in self.graph_buffers:
            info_dict[f'{buffer.name}_best_gc'] = float(buffer.best_graph.gate_count)
            info_dict[f'{buffer.name}_mean_traj_len'] = \
                torch.Tensor(buffer.traj_lengths).mean().item() \
                if len(buffer.traj_lengths) > 0 else 0.
        return info_dict
    
    def output_opt_path(
        self,
        name: str,
        best_gc: int,
        exp_seq: List[Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]]
    ) -> str:
        for try_i in range(int(1e8)):
            output_dir = os.path.join(self.output_dir, name, f'{best_gc}_{try_i}')
            # in case of duplication under multi-processing setting
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                break
        else:
            raise Exception(f'Unexpected: Couldn\'t find an available path for {os.path.join(self.output_dir, name, f"{best_gc}")}')
        """make a s_exp to output the starting graph"""
        first_s_exp = SerializableExperience.new_empty()
        first_s_exp.action = Action(0, 0)
        first_s_exp.reward = 0.0
        first_s_exp.next_state = exp_seq[0][0].state
        exp_seq = [(first_s_exp, None, exp_seq[0][1])] + exp_seq
        """output the seq"""
        for i_step, (s_exp, graph, next_graph) in enumerate(exp_seq):
            fname = f'{i_step}_{next_graph.gate_count}_{int(s_exp.reward)}_' \
                    f'{s_exp.action.node}_{s_exp.action.xfer}.qasm'
            with open(os.path.join(output_dir, fname), 'w') as f:
                f.write(s_exp.next_state)
        return output_dir
    
    @torch.no_grad()
    def collect_data(
        self,
        len_episode: int,
        max_gate_count_ratio: float,
        nop_stop: bool
    ) -> ExperienceList:
        """collect experiences from observers"""
        future_exp_lists: List[Future[List[SerializableExperience]]] = []
        init_buffer_ids: List[int] = []
        # s_time = get_time_ns()
        for obs_rref in self.obs_rrefs:
            """sample init state"""
            graph_buffer = self.graph_buffers[self.init_buffer_turn]
            init_graph: quartz.PyGraph = graph_buffer.sample()
            graph_buffer.traj_lengths = []
            init_buffer_ids.append(self.init_buffer_turn)
            self.init_buffer_turn = (self.init_buffer_turn + 1) % len(self.graph_buffers)
            """make async RPC to kick off an episode on observers"""
            future_exp_lists.append(obs_rref.rpc_async().run_episode(
                rpc.RRef(self),
                init_graph.to_qasm_str(),
                len_episode,
                max_gate_count_ratio,
                nop_stop,
            ))
            
        # wait until all obervers have finished their episode
        s_exp_lists: List[List[SerializableExperience]] = torch.futures.wait_all(future_exp_lists)
        # e_time = get_time_ns()
        # errprint(f'    Data collected in {dur_ms(e_time, s_time)} ms.')
        """convert graph and maintain graph_buffer"""
        state_dgl_list: List[dgl.graph] = []
        next_state_dgl_list: List[dgl.graph] = []
        for buffer_id, obs_res in zip(init_buffer_ids, s_exp_lists):
            """for each observer's results (several trajectories)"""
            graph_buffer = self.graph_buffers[buffer_id]
            init_graph = None
            exp_seq: List[Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]] = [] # for output optimization path
            for s_exp in obs_res:
                """for each experience"""
                if s_exp.info['start']:
                    init_graph = qasm_to_graph(obs_res[0].state)
                    exp_seq = []
                    i_step = 0
                # qasm_s_time = get_time_ns()
                graph = qasm_to_graph(s_exp.state)
                next_graph = qasm_to_graph(s_exp.next_state)
                # qasm_e_time = get_time_ns()
                # errprint(f'         Tow qasm graph convered in {dur_ms(qasm_e_time, qasm_s_time)} ms.')
                exp_seq.append((s_exp, graph, next_graph))
                if not s_exp.game_over and \
                    not is_nop(s_exp.action.xfer) and \
                    next_graph.gate_count <= init_graph.gate_count: # NOTE: only add graphs with less gate count
                    graph_buffer.push_back(next_graph)
                # dgl_s_time = get_time_ns()
                state_dgl_list.append(graph.to_dgl_graph())
                next_state_dgl_list.append(next_graph.to_dgl_graph())
                # dgl_e_time = get_time_ns()
                # errprint(f'         Tow graph convered to dgl in {dur_ms(dgl_e_time, dgl_s_time)} ms.')
                """best graph maintenance"""
                if next_graph.gate_count < graph_buffer.best_graph.gate_count:
                    seq_path = self.output_opt_path(graph_buffer.name, next_graph.gate_count, exp_seq)
                    msg = f'Agent {self.id} : {graph_buffer.name}: {graph_buffer.best_graph.gate_count} -> {next_graph.gate_count} ! Seq saved to {seq_path} .'
                    print(f'\n{msg}\n')
                    if self.id == 0: # TODO multi-processing logging
                        wandb.alert(
                            title='Better graph is found!',
                            text=msg, level=wandb.AlertLevel.INFO,
                            wait_duration=0,
                        ) # send alert to slack
                    
                    graph_buffer.best_graph = next_graph
                i_step += 1
                if s_exp.game_over:
                    graph_buffer.traj_lengths.append(i_step)
            # end for s_exp
            
        # end for obs
        # s_time = get_time_ns()
        # errprint(f'    Graph converted in {dur_ms(e_time, s_time)} ms.')
        """collect experiences together"""
        s_exps: List[SerializableExperience] = list(itertools.chain(*s_exp_lists))
        
        s_exps_zip = ExperienceList(*list(map(list, zip(*s_exps)))) # type: ignore
        s_exps_zip.state = state_dgl_list
        s_exps_zip.next_state = next_state_dgl_list
        # e_time = get_time_ns()
        # errprint(f'    Data batched in {dur_ms(e_time, s_time)} ms.')
        return s_exps_zip

class PPOMod:
    
    def __init__(
        self, cfg, output_dir: str
    ) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.wandb_mode = 'online'
        if cfg.wandb.offline:
            self.wandb_mode = 'offline'
        elif cfg.wandb.en is False:
            self.wandb_mode = 'disabled'
        self.print_cfg()
        seed_all(cfg.seed)
        
        """init quartz"""
        self.init_quartz_context_func = partial(
            init_quartz_context,
            cfg.gate_set,
            cfg.ecc_file,
            cfg.no_increase,
            cfg.include_nop,
        )
        self.input_graphs: List[Dict[str, str]] = []
        for input_graph in cfg.input_graphs:
            with open(input_graph.path) as f:
                self.input_graphs.append({
                    'name': input_graph['name'],
                    'qasm': f.read(),
                })
        self.num_gate_type: int = 29
        
    def print_cfg(self) -> None:
        print('================ Configs ================')
        for k, v in self.cfg.items():
            print(f'{k} : {v}')
        print(f'output_dir : {self.output_dir}')
        print('=========================================')
    
    def init_process(self, rank: int, ddp_processes: int, obs_processes: int) -> None:
        """init Quartz for each process"""
        global quartz_context
        global quartz_parser
        self.init_quartz_context_func()
        
        """RPC and DDP initialization"""
        # Ref: https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html
        self.rank = rank
        self.ddp_processes = ddp_processes
        tot_processes = ddp_processes + obs_processes
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=f'tcp://localhost:{RPC_PORT}'
        )
        
        if rank < ddp_processes:
            """init agent processes"""
            agent_name = get_agent_name(rank)
            rpc.init_rpc(
                name=agent_name, rank=rank, world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://localhost:{DDP_PORT}',
                rank=rank, world_size=ddp_processes,
            )
            self.train()
        else:
            """init observer processes"""
            obs_rank = rank - ddp_processes
            agent_rref_id = int(obs_rank // self.cfg.obs_per_agent)
            obs_in_agent_rank = int(obs_rank % self.cfg.obs_per_agent)
            obs_name = get_obs_name(agent_rref_id, obs_in_agent_rank)
            rpc.init_rpc(
                name=obs_name, rank=rank, world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
            # print(f'{obs_name} initialized')
        # block until all rpcs finish
        rpc.shutdown()
    
    def train(self) -> None:
        """init agent and network"""
        if self.cfg.gpus is None or len(self.cfg.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.cfg.gpus[self.rank]}')
        torch.cuda.set_device(self.device)
        self.ac_net = ActorCritic(
            num_gate_type=self.num_gate_type,
            graph_embed_size=self.cfg.graph_embed_size,
            actor_hidden_size=self.cfg.actor_hidden_size,
            critic_hidden_size=self.cfg.critic_hidden_size,
            action_dim=quartz_context.num_xfers,
            device=self.device,
        ).to(self.device)
        self.ac_net_old = copy.deepcopy(self.ac_net)
        # NOTE should not use self.ac_net later
        self.agent = PPOAgent(
            agent_id=self.rank,
            num_observers=self.cfg.obs_per_agent,
            device=self.device,
            batch_inference=self.cfg.batch_inference,
            invalid_reward=self.cfg.invalid_reward,
            ac_net=self.ac_net_old,
            input_graphs=self.input_graphs,
            softmax_hit_rate=self.cfg.softmax_hit_rate,
            output_dir=self.output_dir,
        )
        self.ddp_ac_net = DDP(self.ac_net, device_ids=[self.device])
        self.optimizer = torch.optim.Adam([
            {
                'params': self.ddp_ac_net.module.graph_embedding.parameters(), # type: ignore
                'lr': self.cfg.lr_graph_embedding,
            },
            {
                'params': self.ddp_ac_net.module.actor.parameters(), # type: ignore
                'lr': self.cfg.lr_actor,
            },
            {
                'params': self.ddp_ac_net.module.critic.parameters(), # type: ignore
                'lr': self.cfg.lr_critic,
            }
        ])
        if self.rank == 0:
            wandb.init(
                project='PPO',
                entity=self.cfg.wandb.entity,
                mode=self.wandb_mode,
                config=self.cfg,
            )
        print(f'rank {self.rank} on {self.device} initialized', flush=True)
        
        max_iterations = int(self.cfg.max_iterations)
        self.i_iter = 0
        if self.cfg.resume:
            self.load_ckpt(self.cfg.ckpt_path)
        """train loop"""
        while self.i_iter < max_iterations:
            # s_time = get_time_ns()
            self.train_iter()
            if self.i_iter % self.cfg.update_policy_interval == 0:
                self.ac_net_old.load_state_dict(self.ddp_ac_net.module.state_dict())
            if self.i_iter % self.cfg.save_ckpt_interval == 0:
                self.save_ckpt(f'iter_{self.i_iter}.pt')
                
            # e_time = get_time_ns()
            # errprint(f'Iter {self.i_iter} finished in {dur_ms(s_time, e_time)} ms.')
            self.i_iter += 1

        
    def train_iter(self) -> None:
        """collect batched data in dgl or tensor format"""
        s_time_collect = get_time_ns()
        exp_list: ExperienceList = self.agent.collect_data(self.cfg.len_episode, self.cfg.max_gate_count_ratio, self.cfg.nop_stop)
        # support the case that (self.agent_batch_size > self.cfg.obs_per_agent)
        for _i in range(self.cfg.num_trajs_per_iter // self.cfg.obs_per_agent - 1):
            exp_list += self.agent.collect_data(self.cfg.len_episode, self.cfg.max_gate_count_ratio, self.cfg.nop_stop)
        e_time_collect = get_time_ns()
        """evaluate, compute loss, and update (DDP)"""
        # Each agent has different data, so it is DDP training
        if self.rank == 0:
            pbar = tqdm(
                total=self.cfg.k_epochs * math.ceil(len(exp_list) / self.cfg.mini_batch_size),
                desc=f'Iter {self.i_iter}',
                bar_format='{desc} : {n}/{total} |{bar}| {elapsed} {postfix}',
            )
            wandb.log(self.agent.other_info_dict())
        
        for epoch_k in range(self.cfg.k_epochs):
            # print(f'epoch {epoch_k}', flush=True)
            for i_step, exps in enumerate(
                ExperienceListIterator(exp_list, self.cfg.mini_batch_size, self.device)
            ):
                # print(f'  step {i_step} {len(exps)}', flush=True)
                self.optimizer.zero_grad()
                """get embeds of seleted nodes and evaluate them by Critic"""
                num_nodes: torch.LongTensor = exps.state.batch_num_nodes()
                # (batch_num_nodes, embed_dim)
                b_graph_embeds: torch.Tensor = self.ddp_ac_net.module.graph_embedding(exps.state) # type: ignore
                nodes_offset: torch.LongTensor = torch.LongTensor([0] * num_nodes.shape[0]).to(self.device) # type: ignore
                nodes_offset[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
                selected_nodes = exps.action[:, 0] + nodes_offset
                selected_node_embeds = b_graph_embeds[selected_nodes]
                selected_node_values: torch.Tensor = self.ddp_ac_net.module.critic(selected_node_embeds).squeeze() # type: ignore
                """get xfer dist by Actor"""
                # (batch_num_graphs, action_dim)
                xfer_logits: torch.Tensor = self.ddp_ac_net.module.actor(selected_node_embeds) # type: ignore
                softmax_xfer = masked_softmax(xfer_logits, exps.xfer_mask)
                xfer_dists = Categorical(softmax_xfer)
                # (batch_num_graphs, )
                xfer_logprobs: torch.Tensor = xfer_dists.log_prob(exps.action[:, 1])
                xfer_entropys = xfer_dists.entropy()
                """get embeds of next nodes and evaluate them by Critic without grad"""
                with torch.no_grad():
                    # (num_next_graphs, )
                    next_num_nodes: torch.LongTensor = exps.next_state.batch_num_nodes()
                    """get embeds"""
                    # (batch_next_graphs_nodes, embed_dim)
                    b_next_graph_embeds: torch.Tensor = self.ddp_ac_net.module.graph_embedding(exps.next_state) # type: ignore
                    next_graph_embeds_list: List[torch.Tensor] = torch.split(b_next_graph_embeds, next_num_nodes.tolist())
                    """select embeds"""
                    # ( sum(num_next_nodes), embed_dim )
                    next_node_embeds: torch.Tensor = torch.cat([
                        graph_embed[next_node_ids]
                        for (next_node_ids, graph_embed) in zip(exps.next_nodes, next_graph_embeds_list)
                    ])
                    """evaluate"""
                    # ( sum(num_next_nodes), )
                    next_node_values: torch.Tensor = self.ddp_ac_net.module.critic(next_node_embeds).squeeze() # type: ignore
                    num_next_nodes = list(map(len, exps.next_nodes))
                    next_node_values_list: List[torch.Tensor] = torch.split(next_node_values, num_next_nodes)
                    """get max next value for each graph"""
                    max_next_values_list: List[torch.Tensor] = []
                    for i in range(len(exps)):
                        max_next_value: torch.Tensor
                        # next_nodes == [] means invalid xfer
                        if next_node_values_list[i].shape[0] == 0 or \
                            is_nop(int(exps.action[i, 1])) and self.cfg.nop_stop:
                            max_next_value = torch.zeros(1).to(self.device)
                        # CONFIRM how to deal with NOP?
                        else:
                            max_next_value, _ = torch.max(next_node_values_list[i], dim=0, keepdim=True)
                        max_next_values_list.append(max_next_value)
                    max_next_values = torch.cat(max_next_values_list)
                # end with
                """compute loss for Actor (policy_net, theta)"""
                # prob ratio = (pi_theta / pi_theta__old)
                ratios = torch.exp(xfer_logprobs - exps.xfer_logprob)
                advantages = exps.reward + self.cfg.gamma * max_next_values - selected_node_values
                with torch.no_grad():
                    # NOTE: is clone().detach() necessary?
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(
                        ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip
                    ) * advantages
                actor_loss = - torch.sum(torch.min(surr1, surr2)) / len(exps)
                """compute loss for Critic (value_net, phi)"""
                critic_loss = torch.sum(advantages ** 2) / len(exps)
                xfer_entropy = torch.sum(xfer_entropys) / len(exps)
                """compute overall loss""" 
                loss = actor_loss + 0.5 * critic_loss - float(self.cfg.entropy_coeff) * xfer_entropy
                """update"""
                loss.backward()
                for param in self.ddp_ac_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                """logging"""
                if self.rank == 0:
                    pbar.update(1)
                    log_dict = {
                        'actor_loss': float(actor_loss),
                        'critic_loss': float(critic_loss),
                        'xfer_entropy': float(xfer_entropy),
                        'loss': float(loss),
                    }
                    pbar.set_postfix({
                        **log_dict,
                        'collect_time': dur_ms(e_time_collect, s_time_collect) / 1e3,
                        'num_exps': len(exp_list),
                    })
                    pbar.refresh()
                    wandb.log({
                        'actor_loss': actor_loss,
                        'critic_loss': critic_loss,
                        'xfer_entropy': xfer_entropy,
                        'loss': loss,
                    })
            # end for i_step
        # end for k_epochs
        # e_time = get_time_ns()
        # errprint(f'  {self.cfg.k_epochs} epochs finished in {dur_ms(s_time, e_time)} ms.')
        
    def save_ckpt(self, ckpt_name: str, only_rank_zero: bool = True) -> None:
        # TODO save top-k model
        ckpt_path = os.path.join(self.output_dir, ckpt_name)
        if not only_rank_zero or self.rank == 0:
            torch.save({
                'i_iter': self.i_iter,
                'model_state_dict': self.ddp_ac_net.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'loss': LOSS,
            }, ckpt_path)
            print(f'saved "{ckpt_path}"!')
        
    def load_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.agent.device)
        self.i_iter = ckpt['i_iter']
        model_state_dict = ckpt['model_state_dict']
        self.ddp_ac_net.module.load_state_dict(model_state_dict)
        self.ac_net_old.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f'resumed from "{ckpt}"!')
        

@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    output_dir = os.path.abspath(os.curdir) # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd()) # set working dir to the original one
    
    warnings.simplefilter('ignore')
    
    ppo_mod = PPOMod(cfg, output_dir)
    
    mp.set_start_method(cfg.mp_start_method)
    ddp_processes = 1
    if len(cfg.gpus) > 1:
        ddp_processes = len(cfg.gpus)
    obs_processes = int(ddp_processes * cfg.obs_per_agent)
    tot_processes = ddp_processes + obs_processes
    print(f'spawning {tot_processes} processes...')
    mp.spawn(
        fn=ppo_mod.init_process,
        args=(ddp_processes, obs_processes,),
        nprocs=tot_processes,
        join=True,
    )    
    # TODO make sure qasm <-> graph conversion is correct
    # TODO select_action_batch may timeout
    # TODO confirm params in config.yaml
    # TODO find an optimal config of mp, or it will OOM
    # TODO profiling; some parts of this code are slow
    
    # 4 gpus, 2 obs per agent, len_episode = 80 -> 7994 MiB / GPU

if __name__ == '__main__':
    main()
