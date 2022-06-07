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

def convert_exp(s_exp: SerializableExperience) -> Experience:
    return Experience(
        qasm_to_graph(s_exp.state),
        s_exp.action, s_exp.reward,
        qasm_to_graph(s_exp.next_state),
        s_exp.game_over,
    )

class RolloutBuffer:
    
    def __init__(self) -> None:
        self.exps: List[Experience] = []
    
    def append(self, exp: Experience) -> RolloutBuffer:
        self.exps.append(exp)
        return self

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
            """get action from agent"""
            action: Action
            if self.batch_inference:
                action = agent_rref.rpc_sync().select_action_batch(
                    self.id, graph.to_qasm_str(),
                )
            else:
                action = agent_rref.rpc_sync().select_action(
                    self.id, graph.to_qasm_str(),
                ) # NOTE not sure if it's OK because `select_action` doesn't return a `Future`
            
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
                reward = (graph.gate_count - next_graph.gate_count) * 3
                next_graph_str = next_graph.to_qasm_str()
            
            exp = SerializableExperience(
                graph_str, action, reward, next_graph_str, game_over,
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
        policy_net: ActorCritic,
    ) -> None:
        self.id = agent_id
        self.device = device
        """networks related"""
        self.policy_net = policy_net # NOTE: just a ref
        
        """init Observers on the other processes and hold the refs to them"""
        self.ob_rrefs: List[rpc.RRef] = []
        for obs_rank in range(0, num_observers):
            ob_info = rpc.get_worker_info(get_obs_name(self.id, obs_rank))
            self.ob_rrefs.append(
                rpc.remote(ob_info, Observer, args=(obs_rank, batch_inference, invalid_reward,))
            )
        
        """helper vars for select_action"""
        self.future_actions: Future[List[Action]] = Future()
        self.pending_states = len(self.ob_rrefs)
        self.states_buf: List[dgl.graph] = [None] * len(self.ob_rrefs)
        
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
    def select_action_batch(self, obs_id: int, state_str: str) -> Future[Action]:
        """inference a batch of queries queried by all of the observers at once"""
        future_action: Future[Action] = self.future_actions.then(
            lambda future_actions: future_actions.wait()[obs_id]
        ) # this single action is returned for the obs that calls this function
        pygraph: quartz.PyGraph = qasm_to_graph(state_str)
        dgl_graph: dgl.graph = pygraph.to_dgl_graph().to(self.device)
        self.states_buf[obs_id] = dgl_graph
        
        with self.lock: # avoid data race on self.pending_states
            self.pending_states -= 1
            if self.pending_states == 0:
                """collected a batch, start batch inference"""
                b_state: dgl.graph = dgl.batch(self.states_buf)
                
                # TODO get action to respond each observer's query
                
                # TODO store actions in obs_id order
                actions = [Action(0, 0) for i in range(len(self.ob_rrefs))]
                self.future_actions.set_result(actions)
                """re-init"""
                self.pending_states = len(self.ob_rrefs)
                self.future_actions = torch.futures.Future()
        return future_action
    
    def update(self):
        pass
    
    def collect_data(
        self,
        len_episode: int,
        input_qasm_str: str,
    ) -> List[Experience]:
        """collect experiences from observers"""
        future_exp_lists: List[Future[List[SerializableExperience]]] = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            future_exp_lists.append(ob_rref.rpc_async().run_episode(
                rpc.RRef(self),
                input_qasm_str,
                len_episode,
            ))
        # wait until all obervers have finished their episode
        s_exp_lists: List[List[SerializableExperience]] = torch.futures.wait_all(future_exp_lists)
        """convert collected experiences to Quartz format"""
        s_exps: List[SerializableExperience] = list(itertools.chain(*s_exp_lists))
        exps: List[Experience] = [convert_exp(s_exp) for s_exp in s_exps]
        return exps

class PPOMod:
    
    def __init__(
        self, cfg, output_dir: str
    ) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        wandb_mode = 'online'
        if cfg.wandb.offline:
            wandb_mode = 'offline'
        elif cfg.wandb.en is False:
            wandb_mode = 'disabled'
        wandb.init(
            project='PPO',
            entity=cfg.wandb.entity,
            mode=wandb_mode,
            config=cfg,
        ) # NOTE Not sure if it's ok to init wandb here under multiprocessing setting.
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
        self.collect_batch_size = int(cfg.collect_batch_size)
                
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
            agent_rref_id = int(obs_rank / self.cfg.obs_per_agent)
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
            agent_device = torch.device('cpu')
        else:
            agent_device = torch.device(f'cuda:{self.cfg.gpus[self.rank]}')
        self.policy_net = ActorCritic(
            num_gate_type=self.num_gate_type,
            graph_embed_size=self.cfg.graph_embed_size,
            actor_hidden_size=self.cfg.actor_hidden_size,
            critic_hidden_size=self.cfg.critic_hidden_size,
            action_dim=quartz_context.num_xfers,
            device=agent_device,
        ).to(agent_device)
        self.policy_net_old = copy.deepcopy(self.policy_net)
        self.agent = PPOAgent(
            agent_id=self.rank,
            num_observers=self.cfg.obs_per_agent,
            device=agent_device,
            batch_inference=self.cfg.batch_inference,
            invalid_reward=self.cfg.invalid_reward,
            policy_net=self.policy_net_old,
        )
        self.ddp_policy_net = DDP(self.policy_net, device_ids=[self.rank])
        self.optimizer = torch.optim.Adam([
            {
                'params': self.policy_net.graph_embedding.parameters(),
                'lr': self.cfg.lr_graph_embedding,
            }, 
            {
                'params': self.policy_net.actor.parameters(),
                'lr': self.cfg.lr_actor,
            },
            {
                'params': self.policy_net.critic.parameters(),
                'lr': self.cfg.lr_critic,
            }
        ])
        print(f'rank {self.rank} initialized')
        """train"""
        max_iterations = int(self.cfg.max_iterations)
        self.i_iter = 0
        if self.cfg.resume:
            self.load_ckpt(self.cfg.ckpt_path)
        while self.i_iter < max_iterations:
            self.train_iter()
            if self.i_iter % self.cfg.save_ckpt_interval == 0:
                self.save_ckpt(f'iter_{self.i_iter}.pt') # TODO add loss and best_gc in the name
            self.i_iter += 1            
        
    def train_iter(self) -> None:
        """collect data and build batched data in dgl or tensor format"""
        exps = self.agent.collect_data(self.cfg.len_episode, self.input_qasm_str)
        # print(exps)
        """evaluate, compute loss, and update (DDP)"""
        # NOTE: Each agent has different data, so it is DDP training
        # TODO
        """logging"""
        
        
    def save_ckpt(self, ckpt_name: str, only_rank_zero: bool = True) -> None:
        # TODO save top-k model
        ckpt_path = os.path.join(self.output_dir, ckpt_name)
        if not only_rank_zero or self.rank == 0:
            torch.save({
                'i_iter': self.i_iter,
                'model_state_dict': self.ddp_policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'loss': LOSS,
            }, ckpt_path)
            print(f'saved "{ckpt_path}"!')
        
    def load_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path)
        self.i_iter = ckpt['i_iter']
        self.ddp_policy_net.load_state_dict(ckpt['model_state_dict'])
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
