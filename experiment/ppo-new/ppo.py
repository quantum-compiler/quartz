from __future__ import annotations
import os
from posixpath import expanduser
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

from utils import *
from model import ActorCritic
from IPython import embed # type: ignore

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
        
        graph = init_graph
        graph_str = init_state_str
        for i_step in range(len_episode):
            # print(f'obs {self.id} step {i_step}')
            """get action (action_node, xfer_dist) from agent"""
            _action: ActionTmp
            if self.batch_inference:
                _action = agent_rref.rpc_sync().select_action_batch(
                    self.id, graph.to_qasm_str(),
                )
            else:
                _action = agent_rref.rpc_sync().select_action(
                    self.id, graph.to_qasm_str(),
                ) # NOTE not sure if it's OK because `select_action` doesn't return a `Future`
           
            """sample action_xfer with mask"""
            av_xfers = graph.available_xfers_parallel(
                context=quartz_context, node=graph.get_node_from_id(id=_action.node))
            av_xfer_mask = torch.BoolTensor([0] * quartz_context.num_xfers)
            av_xfer_mask[av_xfers] = True
            # TODO use softmax temperature
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
                next_graph_str = graph_str # TODO placeholder?
            elif quartz_context.get_xfer_from_id(id=action.xfer).is_nop:
                reward = 0
                game_over = True
                next_graph_str = graph_str # unchanged
                next_nodes = [action.node]
            else:
                reward = graph.gate_count - next_graph.gate_count
                game_over = False
                next_graph_str = next_graph.to_qasm_str()
            
            exp = SerializableExperience(
                graph_str, action, reward, next_graph_str, game_over,
                next_nodes, av_xfer_mask, action_xfer_logp.item(),
            )
            exp_list.append(exp)
            
            if game_over and not self.batch_inference:
                break
            else:
                if self.batch_inference:
                    graph = init_graph
                    graph_str = init_state_str
                else:
                    graph = next_graph
                    graph_str = next_graph_str
        # end for
        return exp_list

class PPOAgent:
    
    def __init__(
        self,
        agent_id: int,
        num_observers: int,
        device: torch.device,
        batch_inference: bool,
        invalid_reward: float,
        ac_net: ActorCritic,
        input_qasm_str: str,
        softmax_hit_rate: float,
    ) -> None:
        self.id = agent_id
        self.device = device
        """networks related"""
        self.ac_net = ac_net # NOTE: just a ref
        self.softmax_hit_rate = softmax_hit_rate
        
        """init Observers on the other processes and hold the refs to them"""
        self.ob_rrefs: List[rpc.RRef] = []
        for obs_rank in range(0, num_observers):
            ob_info = rpc.get_worker_info(get_obs_name(self.id, obs_rank))
            self.ob_rrefs.append(
                rpc.remote(ob_info, Observer, args=(obs_rank, batch_inference, invalid_reward,))
            )
        
        """helper vars for select_action"""
        self.future_actions: Future[List[ActionTmp]] = Future()
        self.pending_states = len(self.ob_rrefs)
        self.states_buf: List[dgl.graph] = [None] * len(self.ob_rrefs)
        self.lock = threading.Lock()
        
        # TODO init state buffer
        self.input_qasm_str = input_qasm_str
            
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
        self.states_buf[obs_id] = dgl_graph
        
        with self.lock: # avoid data race on self.pending_states
            self.pending_states -= 1
            if self.pending_states == 0:
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
                b_softmax_node_values_pad = F.softmax(b_node_values_pad / temperature, dim=-1)
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
                    for i in range(len(self.ob_rrefs))
                ]
                self.future_actions.set_result(actions)
                """re-init"""
                self.pending_states = len(self.ob_rrefs)
                self.future_actions = torch.futures.Future()
                self.states_buf = [None] * len(self.ob_rrefs)
        return future_action # return a future
    
    @torch.no_grad()
    def collect_data(self, len_episode: int) -> BatchedExperience:
        """collect experiences from observers"""
        future_exp_lists: List[Future[List[SerializableExperience]]] = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            future_exp_lists.append(ob_rref.rpc_async().run_episode(
                rpc.RRef(self),
                self.input_qasm_str, # TODO init state buffer and sampling
                len_episode,
            ))
        # wait until all obervers have finished their episode
        s_exp_lists: List[List[SerializableExperience]] = torch.futures.wait_all(future_exp_lists)
        """convert collected experiences to batched Quartz format"""
        s_exps: List[SerializableExperience] = list(itertools.chain(*s_exp_lists))
        s_exps_zip = BSerializableExperience(*zip(*s_exps)) # type: ignore
        # state, action, reward, next_state, game_over, xfer_mask, xfer_logprob
        exps = BatchedExperience.new_empty()
        exps.state = dgl.batch(
            [qasm_to_graph(g).to_dgl_graph() for g in s_exps_zip.state]
        ).to(self.device) # TODO add state into the init state buffer
        exps.next_state = dgl.batch(
            [qasm_to_graph(g).to_dgl_graph() for g in s_exps_zip.next_state]
        ).to(self.device)
        exps.action = torch.stack([a.to_tensor() for a in s_exps_zip.action]).to(self.device) # type: ignore
        exps.reward = torch.Tensor(s_exps_zip.reward).to(self.device)
        exps.game_over = torch.BoolTensor(s_exps_zip.game_over).to(self.device) # type: ignore
        exps.next_nodes = [ torch.LongTensor(ns).to(self.device) for ns in s_exps_zip.next_nodes ] # type: ignore
        exps.xfer_mask = torch.stack(s_exps_zip.xfer_mask).to(self.device) # type: ignore
        exps.xfer_logprob = torch.Tensor(s_exps_zip.xfer_logprob).to(self.device)
        return exps

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
        with open(cfg.input_graph_path) as f:
            self.input_qasm_str = f.read()
        self.num_gate_type: int = 29
        
        """init training related parameters"""
        self.global_batch_size = int(cfg.global_batch_size)
        
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
        self.agent_batch_size = int(self.global_batch_size // ddp_processes)
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
            input_qasm_str=self.input_qasm_str,
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
        print(f'rank {self.rank} on {self.device} initialized')
        
        max_iterations = int(self.cfg.max_iterations)
        self.i_iter = 0
        if self.cfg.resume:
            self.load_ckpt(self.cfg.ckpt_path)
        """train loop"""
        while self.i_iter < max_iterations:
            self.train_iter()
            if self.i_iter % self.cfg.update_policy_interval == 0:
                self.ac_net_old.load_state_dict(self.ddp_ac_net.module.state_dict())
            if self.i_iter % self.cfg.save_ckpt_interval == 0:
                self.save_ckpt(f'iter_{self.i_iter}.pt') # TODO add loss and best_gc in the name
            self.i_iter += 1
        
    def train_iter(self) -> None:
        """collect batched data in dgl or tensor format"""
        exps: BatchedExperience = self.agent.collect_data(self.cfg.len_episode)
        # support the case that (self.agent_batch_size > self.cfg.obs_per_agent)
        for _i in range(self.agent_batch_size // self.cfg.obs_per_agent - 1):
            exps += self.agent.collect_data(self.cfg.len_episode)
        # print(exps)
        """evaluate, compute loss, and update (DDP)"""
        # Each agent has different data, so it is DDP training
        for _ in range(self.cfg.k_epochs):
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
                    if exps.game_over[i] or next_node_values_list[i].shape[0] == 0:
                        max_next_value = torch.zeros(1).to(self.device)
                    # elif quartz_context.get_xfer_from_id(id=int(exps.action[i, 1])).is_nop:
                    #     pass
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
        # end for k_epochs
        
        """logging"""
        if self.rank == 0:
            wandb.log({
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'xfer_entropy': xfer_entropy,
                "loss": loss,
            })
        
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
    mp.spawn(
        fn=ppo_mod.init_process,
        args=(ddp_processes, obs_processes,),
        nprocs=tot_processes,
        join=True,
    )
    

if __name__ == '__main__':
    main()
