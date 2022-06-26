# this file is under mypy's checking
from __future__ import annotations
import os
from typing import Tuple, List, Dict, Any
import threading
import copy
import itertools
import math
import json

import qtz

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.futures import Future
import dgl # type: ignore
# import quartz # type: ignore

import wandb
from omegaconf.dictconfig import DictConfig

from ds import *
from utils import *
from model import ActorCritic
from IPython import embed # type: ignore
from icecream import ic # type: ignore

class Observer:
    
    def __init__(
        self,
        obs_id: int,
        agent_id: int,
        batch_inference: bool,
        invalid_reward: float,
    ) -> None:
        self.id = obs_id
        self.agent_id = agent_id
        self.batch_inference = batch_inference
        self.invalid_reward = invalid_reward
    
    def run_episode(
        self,
        agent_rref: rpc.RRef[PPOAgent],
        init_state_str: str,
        max_eps_len: int,
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
        init_graph = qtz.qasm_to_graph(init_state_str)
        exp_list: List[SerializableExperience] = []
        
        graph = init_graph
        graph_str = init_state_str
        info: Dict[str, Any] = { 'start': True }
        last_eps_end: int = -1
        for i_step in range(max_eps_len):
            """get action (action_node, xfer_dist) from agent"""
            _action: ActionTmp
            if self.batch_inference:
                _action = agent_rref.rpc_sync().select_action_batch(
                    self.id, graph.to_qasm_str(),
                )
            else:
                _action = agent_rref.rpc_sync().select_action(
                    self.id, graph.to_qasm_str(),
                )
            """sample action_xfer with mask"""
            av_xfers = graph.available_xfers_parallel(
                context=qtz.quartz_context, node=graph.get_node_from_id(id=_action.node))
            av_xfer_mask = torch.BoolTensor([0] * qtz.quartz_context.num_xfers)
            av_xfer_mask[av_xfers] = True
            # (action_dim, )  only sample from available xfers
            softmax_xfer = masked_softmax(_action.xfer_dist, av_xfer_mask)
            xfer_dist = Categorical(softmax_xfer)
            action_xfer = xfer_dist.sample()
            action_xfer_logp: torch.Tensor = xfer_dist.log_prob(action_xfer)
            
            """apply action"""
            action = Action(_action.node, action_xfer.item())
            next_nodes: List[int]
            next_graph, next_nodes = \
                graph.apply_xfer_with_local_state_tracking(
                    xfer=qtz.quartz_context.get_xfer_from_id(id=action.xfer),
                    node=graph.get_node_from_id(id=action.node),
                    eliminate_rotation=qtz.has_parameterized_gate,
                )
            """parse result, compute reward"""
            if next_graph is None:
                reward = self.invalid_reward
                game_over = True
                next_graph_str = graph_str # CONFIRM placeholder?
            elif qtz.is_nop(action.xfer):
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
            if game_over:
                if self.batch_inference:
                    graph = init_graph
                    graph_str = init_state_str
                    info['start'] = True
                    last_eps_end = i_step
                else:
                    break
            else:
                graph = next_graph
                graph_str = next_graph_str
        # end for
        if i_step - last_eps_end >= max_eps_len:
            exp_list[-1].game_over = True # eps len exceeds limit
        return exp_list
    

class PPOAgent:
    
    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        num_observers: int,
        device: torch.device,
        batch_inference: bool,
        invalid_reward: float,
        ac_net: ActorCritic,
        input_graphs: List[Dict[str, str]],
        softmax_temp_en: bool,
        hit_rate: float,
        dyn_eps_len: bool,
        max_eps_len: int,
        min_eps_len: int,
        output_dir: str,
    ) -> None:
        self.id = agent_id
        self.num_agents = num_agents
        self.device = device
        self.output_dir = output_dir
        self.dyn_eps_len = dyn_eps_len
        self.max_eps_len = max_eps_len
        self.min_eps_len = min_eps_len
        """networks related"""
        self.ac_net = ac_net # NOTE: just a ref
        self.softmax_temp_en = softmax_temp_en
        self.hit_rate = hit_rate
        
        """init Observers on the other processes and hold the refs to them"""
        self.obs_rrefs: List[rpc.RRef] = []
        for obs_rank in range(0, num_observers):
            ob_info = rpc.get_worker_info(get_obs_name(self.id, obs_rank))
            self.obs_rrefs.append(
                rpc.remote(ob_info, Observer, args=(obs_rank, self.id, batch_inference, invalid_reward,))
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
        
    def run_episode(
        self,
        init_graph: quartz.PyGraph,
        max_eps_len: int,
        max_gate_count_ratio: float,
        nop_stop: bool,
    ) -> List[Experience]:
        exp_list: List[Experience] = []
        
        # gs_time = get_time_ns()
        graph = init_graph
        info: Dict[str, Any] = { 'start': True }
        last_eps_end: int = -1
        for i_step in range(max_eps_len):
            """get action (action_node, xfer_dist) from agent"""
            dgl_graph: dgl.graph = graph.to_dgl_graph().to(self.device)
            num_nodes: int = dgl_graph.num_nodes()
            """compute embeds and use Critic to evaluate each node"""
            node_embeds: torch.Tensor = self.ac_net.graph_embedding(dgl_graph)
            node_values: torch.Tensor = self.ac_net.critic(node_embeds).squeeze()
            temperature: float
            if not self.softmax_temp_en:
                temperature = 1.0
            else:
                temperature = 1 / (math.log( self.hit_rate * (num_nodes - 1)/(1 - self.hit_rate) ))
            softmax_node_values = F.softmax(node_values / temperature, dim=0)
            action_node = int(torch.multinomial(softmax_node_values, 1))
            """use Actor to evaluate xfers for the sampled node"""
            action_node_embed = node_embeds[action_node]
            xfer_logits: torch.Tensor = self.ac_net.actor(action_node_embed)
            """sample action_xfer with mask"""
            av_xfers = graph.available_xfers_parallel(
                context=qtz.quartz_context, node=graph.get_node_from_id(id=action_node))
            av_xfer_mask = torch.BoolTensor([0] * qtz.quartz_context.num_xfers)
            av_xfer_mask[av_xfers] = True
            # (action_dim, )  only sample from available xfers
            softmax_xfer = masked_softmax(xfer_logits, av_xfer_mask)
            xfer_dist = Categorical(softmax_xfer)
            action_xfer = xfer_dist.sample()
            action_xfer_logp: torch.Tensor = xfer_dist.log_prob(action_xfer)
            
            """apply action"""
            action = Action(action_node, action_xfer.item())
            next_nodes: List[int]
            next_graph, next_nodes = \
                graph.apply_xfer_with_local_state_tracking(
                    xfer=qtz.quartz_context.get_xfer_from_id(id=action.xfer),
                    node=graph.get_node_from_id(id=action.node),
                    eliminate_rotation=qtz.has_parameterized_gate,
                )
            """parse result, compute reward"""
            if next_graph is None:
                reward = -1.0
                game_over = True
                next_graph = graph
            elif qtz.is_nop(action.xfer):
                reward = 0
                game_over = nop_stop
                next_nodes = [action.node] # CONFIRM
            else:
                reward = graph.gate_count - next_graph.gate_count
                game_over = (next_graph.gate_count > init_graph.gate_count * max_gate_count_ratio)
                    
            exp = Experience(
                graph, action, reward, next_graph, game_over,
                next_nodes, av_xfer_mask, action_xfer_logp.item(), copy.deepcopy(info),
            )
            exp_list.append(exp)
            info['start'] = False
            if game_over:
                if False:
                    graph = init_graph
                    info['start'] = True
                    last_eps_end = i_step
                else:
                    break
            else:
                graph = next_graph
        
        return exp_list
        
    @torch.no_grad()
    def collect_data_self(
        self,
        max_gate_count_ratio: float,
        nop_stop: bool,
    ) -> ExperienceList:        
        """collect experiences from observers"""
        future_exp_lists: List[List[Experience]] = []
        init_buffer_ids: List[int] = []
        for obs_rref in self.obs_rrefs:
            """sample init state"""
            graph_buffer = self.graph_buffers[self.init_buffer_turn]
            init_graph: quartz.PyGraph = graph_buffer.sample()
            init_buffer_ids.append(self.init_buffer_turn)
            self.init_buffer_turn = (self.init_buffer_turn + 1) % len(self.graph_buffers)
            if self.dyn_eps_len:
                max_len = graph_buffer.max_eps_length
                if max_len < 40:
                    max_eps_len = math.ceil(max_len * 1.5)
                else:
                    max_eps_len = math.ceil(max_len * 1.2)
                max_eps_len = max(max_eps_len, self.min_eps_len)
                max_eps_len = min(max_eps_len, self.max_eps_len)
            else:
                max_eps_len = self.max_eps_len
            """make async RPC to kick off an episode on observers"""
            future_exp_lists.append(self.run_episode(
                init_graph,
                max_eps_len,
                max_gate_count_ratio,
                nop_stop,
            ))
            
        """convert graph and maintain graph_buffer"""
        state_dgl_list: List[dgl.graph] = []
        next_state_dgl_list: List[dgl.graph] = []
        for buffer_id, obs_res in zip(init_buffer_ids, future_exp_lists):
            """for each observer's results (several episodes)"""
            graph_buffer = self.graph_buffers[buffer_id]
            init_graph = None
            exp_seq: List[Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]] = [] # for output optimization path
            for s_exp in obs_res:
                """for each experience"""
                if s_exp.info['start']:
                    init_graph = obs_res[0].state
                    exp_seq = []
                    i_step = 0
                graph = s_exp.state
                next_graph = s_exp.next_state
                ss_exp = SerializableExperience(*s_exp)
                ss_exp.state = s_exp.state.to_qasm_str()
                ss_exp.next_state = s_exp.next_state.to_qasm_str()
                exp_seq.append((ss_exp, graph, next_graph))
                if not s_exp.game_over and \
                    not qtz.is_nop(s_exp.action.xfer) and \
                    next_graph.gate_count <= init_graph.gate_count: # NOTE: only add graphs with less gate count
                    graph_buffer.push_back(next_graph)
                state_dgl_list.append(graph.to_dgl_graph())
                next_state_dgl_list.append(next_graph.to_dgl_graph())
                """best graph maintenance"""
                if next_graph.gate_count < graph_buffer.best_graph.gate_count:
                    seq_path = self.output_opt_path(graph_buffer.name, next_graph.gate_count, exp_seq)
                    msg = f'Agent {self.id} : {graph_buffer.name}: {graph_buffer.best_graph.gate_count} -> {next_graph.gate_count} ! Seq saved to {seq_path} .'
                    printfl(f'\n{msg}\n')
                    if self.id == 0: # TODO multi-processing logging
                        wandb.alert(
                            title='Better graph is found!',
                            text=msg, level=wandb.AlertLevel.INFO,
                            wait_duration=0,
                        ) # send alert to slack
                    
                    graph_buffer.best_graph = next_graph
                i_step += 1
                if s_exp.game_over:
                    graph_buffer.eps_lengths.append(i_step)
            # end for s_exp
            if len(obs_res) > 0 and not obs_res[-1].game_over:
                graph_buffer.eps_lengths.append(i_step)
        # end for res of each obs
            
        """collect experiences together"""
        s_exps: List[Experience] = list(itertools.chain(*future_exp_lists))
        
        s_exps_zip = ExperienceList(*map(list, zip(*s_exps))) # type: ignore
        s_exps_zip.state = state_dgl_list
        s_exps_zip.next_state = next_state_dgl_list
        return s_exps_zip
            
    @torch.no_grad()
    def select_action(self, obs_id: int, state_str: str) -> ActionTmp:
        """respond to a single query"""
        pygraph: quartz.PyGraph = qtz.qasm_to_graph(state_str)
        dgl_graph: dgl.graph = pygraph.to_dgl_graph().to(self.device)
        num_nodes: int = dgl_graph.num_nodes()
        """compute embeds and use Critic to evaluate each node"""
        node_embeds: torch.Tensor = self.ac_net.graph_embedding(dgl_graph)
        node_values: torch.Tensor = self.ac_net.critic(node_embeds).squeeze()
        temperature: float
        if not self.softmax_temp_en:
            temperature = 1.0
        else:
            temperature = 1 / (math.log( self.hit_rate * (num_nodes - 1)/(1 - self.hit_rate) ))
        softmax_node_values = F.softmax(node_values / temperature, dim=0)
        action_node = int(torch.multinomial(softmax_node_values, 1))
        """use Actor to evaluate xfers for the sampled node"""
        action_node_embed = node_embeds[action_node]
        xfer_logits: torch.Tensor = self.ac_net.actor(action_node_embed).cpu()
        
        return ActionTmp(action_node, xfer_logits)
        
    @rpc.functions.async_execution
    @torch.no_grad()
    def select_action_batch(self, obs_id: int, state_str: str) -> Future[ActionTmp]:
        """inference a batch of queries queried by all of the observers at once"""
        future_action: Future[ActionTmp] = self.future_actions.then(
            lambda future_actions: future_actions.wait()[obs_id]
        ) # this single action is returned for the obs that calls this function
        # It is available after self.future_actions is set
        pygraph: quartz.PyGraph = qtz.qasm_to_graph(state_str)
        dgl_graph: dgl.graph = pygraph.to_dgl_graph().to(self.device)
        if self.states_buf[obs_id] is None:
            self.states_buf[obs_id] = dgl_graph
        else:
            raise Exception(f'Unexpected: self.states_buf[{obs_id}] is not None! Duplicate assignment occurs!')

        with self.lock: # avoid data race on self.pending_states
            self.pending_states -= 1
            if self.pending_states == 0:
                """collected a batch, start batch inference"""
                b_state: dgl.graph = dgl.batch(self.states_buf)
                num_nodes: torch.Tensor = b_state.batch_num_nodes() # (num_graphs, ) assert each elem > 0
                """compute embeds and use Critic to evaluate each node"""
                # (batch_num_nodes, embed_dim)
                b_node_embeds: torch.Tensor = self.ac_net.graph_embedding(b_state)
                # (batch_num_nodes, )
                b_node_values: torch.Tensor = self.ac_net.critic(b_node_embeds).squeeze()
                # list with length num_graphs; each member is a tensor of node values in a graph
                node_values_list: List[torch.Tensor] = torch.split(b_node_values, num_nodes.tolist())
                """sample node by softmax with temperature for each graph as a batch"""
                # (num_graphs, max_num_nodes)
                b_node_values_pad = nn.utils.rnn.pad_sequence(
                    node_values_list, batch_first=True, padding_value=-math.inf)
                # (num_graphs, )
                temperature: torch.Tensor
                if not self.softmax_temp_en:
                    temperature = torch.ones(1).to(self.device)
                else:
                    temperature = 1 / (torch.log( self.hit_rate * (num_nodes - 1)/(1 - self.hit_rate) ))
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
                this_future_actions = self.future_actions # NOTE something magic to avoid duplicate assignment
                """re-init"""
                self.pending_states = len(self.obs_rrefs)
                self.future_actions = Future()
                self.states_buf = [None] * len(self.obs_rrefs)
                
                this_future_actions.set_result(actions)
        # end with
        return future_action # return a future
    
    def perpare_buf_for_next_iter(self) -> None:
        for buffer in self.graph_buffers:
            buffer.prepare_for_next_iter()

    def other_info_dict(self) -> Dict[str, float | int]:
        info_dict: Dict[str, float | int] = {}
        for buffer in self.graph_buffers:
            info_dict[f'{buffer.name}_best_gc'] = buffer.best_graph.gate_count
            info_dict[f'{buffer.name}_buffer_size'] = len(buffer)
            
            info_dict[f'{buffer.name}_mean_eps_len'] = \
                torch.Tensor(buffer.eps_lengths).mean().item() \
                if len(buffer.eps_lengths) > 0 else 0.
            
            max_eps_len, this_max_eps_len, _ = buffer.update_max_eps_length()
            info_dict[f'{buffer.name}_max_eps_len'] = this_max_eps_len
            info_dict[f'{buffer.name}_max_eps_len_global'] = max_eps_len
            
            rewards_info = buffer.rewards_info()
            for k, v in rewards_info.items():
                info_dict[f'{buffer.name}_{k}'] = v
            
            gate_count_info = buffer.gate_count_info()
            for k, v in gate_count_info.items():
                info_dict[f'{buffer.name}_{k}'] = v
            
        return info_dict
    
    def output_opt_path(
        self,
        name: str,
        best_gc: int,
        exp_seq: List[Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]]
    ) -> str:
        output_dir = os.path.join(self.output_dir, name, f'{best_gc}_{self.id}')
        os.makedirs(output_dir)
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
        max_gate_count_ratio: float,
        nop_stop: bool,
    ) -> ExperienceList:        
        """collect experiences from observers"""
        future_exp_lists: List[Future[List[SerializableExperience]]] = []
        init_buffer_ids: List[int] = []
        init_graph_qasm_list: List[str] = []
        max_eps_len_for_all: int = 0
        for obs_rref in self.obs_rrefs:
            """sample init state"""
            graph_buffer = self.graph_buffers[self.init_buffer_turn]
            init_graph: quartz.PyGraph = graph_buffer.sample()
            init_buffer_ids.append(self.init_buffer_turn)
            self.init_buffer_turn = (self.init_buffer_turn + 1) % len(self.graph_buffers)
            if self.dyn_eps_len:
                max_len = graph_buffer.max_eps_length
                if max_len < 40:
                    max_eps_len = math.ceil(max_len * 1.5)
                else:
                    max_eps_len = math.ceil(max_len * 1.2)
                max_eps_len = max(max_eps_len, self.min_eps_len)
                max_eps_len = min(max_eps_len, self.max_eps_len)
            else:
                max_eps_len = self.max_eps_len
            init_graph_qasm_list.append(init_graph.to_qasm_str())
            max_eps_len_for_all = max(max_eps_len_for_all, max_eps_len)
        # end for
        """make async RPC to kick off an episode on observers"""
        for obs_rref, init_qasm in zip(self.obs_rrefs, init_graph_qasm_list):
            future_exp_lists.append(obs_rref.rpc_async().run_episode(
                rpc.RRef(self),
                init_qasm,
                max_eps_len_for_all, # make sure all observers have the same max_eps_len
                max_gate_count_ratio,
                nop_stop,
            ))

        # wait until all obervers have finished their episode
        s_exp_lists: List[List[SerializableExperience]] = torch.futures.wait_all(future_exp_lists)
        """convert graph and maintain infos in graph_buffer"""
        state_dgl_list: List[dgl.graph] = []
        next_state_dgl_list: List[dgl.graph] = []
        for buffer_id, obs_res in zip(init_buffer_ids, s_exp_lists):
            """for each observer's results (several episodes)"""
            graph_buffer = self.graph_buffers[buffer_id]
            init_graph = None
            exp_seq: List[Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]] = [] # for output optimization path
            for s_exp in obs_res:
                """for each experience"""
                graph = qtz.qasm_to_graph(s_exp.state)
                next_graph = qtz.qasm_to_graph(s_exp.next_state)
                state_dgl_list.append(graph.to_dgl_graph())
                next_state_dgl_list.append(next_graph.to_dgl_graph())
                exp_seq.append((s_exp, graph, next_graph))
                """add graphs into buffer"""
                if not s_exp.game_over and \
                    not qtz.is_nop(s_exp.action.xfer) and \
                    next_graph.gate_count <= init_graph.gate_count: # NOTE: only add graphs with less or equal gate count
                    graph_buffer.push_back(next_graph)
                if s_exp.info['start']:
                    init_graph = qtz.qasm_to_graph(obs_res[0].state)
                    exp_seq = []
                    i_step = 0
                    graph_buffer.rewards.append([])
                    graph_buffer.init_graph_gcs.append(graph.gate_count)
                    graph_buffer.graph_gcs.append(graph.gate_count)
                
                graph_buffer.rewards[-1].append(s_exp.reward)
                graph_buffer.graph_gcs.append(next_graph.gate_count)
                """best graph maintenance"""
                if next_graph.gate_count < graph_buffer.best_graph.gate_count:
                    seq_path = self.output_opt_path(graph_buffer.name, next_graph.gate_count, exp_seq)
                    msg = f'Agent {self.id} : {graph_buffer.name}: {graph_buffer.best_graph.gate_count} -> {next_graph.gate_count} ! Seq saved to {seq_path} .'
                    printfl(f'\n{msg}\n')
                    if self.id == 0: # TODO multi-processing logging
                        wandb.alert(
                            title='Better graph is found!',
                            text=msg, level=wandb.AlertLevel.INFO,
                            wait_duration=0,
                        ) # send alert to slack
                    # end if
                    graph_buffer.best_graph = next_graph
                # end if better
                i_step += 1
                if s_exp.game_over:
                    graph_buffer.eps_lengths.append(i_step)
            # end for s_exp
            if len(obs_res) > 0 and not obs_res[-1].game_over:
                graph_buffer.eps_lengths.append(i_step)
        # end for obs_res
        
        # end for obs
        """collect experiences together"""
        s_exps: List[SerializableExperience] = list(itertools.chain(*s_exp_lists))
        
        s_exps_zip = ExperienceList(*map(list, zip(*s_exps))) # type: ignore
        s_exps_zip.state = state_dgl_list
        s_exps_zip.next_state = next_state_dgl_list
        return s_exps_zip

    def sync_best_graph(self) -> None:
        """broadcast the best graph of each buffer to other agents"""
        sync_dir = os.path.join(self.output_dir, 'sync_dir')
        os.makedirs(sync_dir, exist_ok=True)
        best_info = [
            {
                'name': buffer.name,
                'gate_count': buffer.best_graph.gate_count,
                'qasm': buffer.best_graph.to_qasm_str(),
            }
            for buffer in self.graph_buffers
        ]
        with open(os.path.join(sync_dir, f'best_info_{self.id}.json'), 'w') as f:
            json.dump(best_info, fp=f, indent=2)
        printfl(f'Agent {self.id} : waiting for others to sync')
        dist.barrier()
        printfl(f'Agent {self.id} : finish waiting')
        """read in other agents' results"""
        for r in range(self.num_agents):
            if r != self.id:
                with open(os.path.join(sync_dir, f'best_info_{r}.json')) as f:
                    other_best_info = json.load(f)
                for i in range(len(self.graph_buffers)):
                    buffer = self.graph_buffers[i]
                    other_info = other_best_info[i]
                    assert buffer.name == other_info['name']
                    if buffer.push_nonexist_best(other_info['qasm']):
                        printfl(f'  Agent {self.id} : read in new best graph ({buffer.best_graph.gate_count}) from agent {r}')
                    # end if
                # end for i
            # end if r
        # end for r
    