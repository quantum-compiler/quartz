from __future__ import annotations
import os
import random
from typing import Callable, Tuple, List, Any
import warnings
from collections import deque, namedtuple
from functools import partial
import threading
import time

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

# global vars to avoid serialization when multiprocessing
quartz_context: quartz.QuartzContext
quartz_parser: quartz.PyQASMParser
rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
    rpc_timeout=10000,
    init_method='env://',
    _transports=["uv"],
) # uv means TCP, to overwrie default SHM which may hit ulimit

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

AGENT_NAME = 'agent_{}'
OBS_NAME = 'obs_{}_{}'

def get_quartz_context(init_args: QuartzInitArgs) -> Tuple[quartz.QuartzContext, quartz.PyQASMParser]:
    quartz_context = quartz.QuartzContext(
        gate_set=init_args.gate_set,
        filename=init_args.ecc_file_path,
        no_increase=init_args.no_increase,
        include_nop=init_args.include_nop,
    )
    quartz_parser = quartz.PyQASMParser(context=quartz_context)
    return quartz_context, quartz_parser

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

class RolloutBuffer:
    
    def __init__(self) -> None:
        self.exps: List[Experience] = []
    
    def append(self, exp: Experience) -> RolloutBuffer:
        self.exps.append(exp)
        return self

def qasm_to_graph(qasm_str: str) -> quartz.PyGraph:
    global quartz_context
    global quartz_parser
    dag = quartz_parser.load_qasm_str(qasm_str)
    graph = quartz.PyGraph(context=quartz_context, dag=dag)
    return graph
    
class Observer:
    
    def __init__(
        self,
        batch_inference: bool,
        invalid_reward: float,
    ) -> None:
        self.id = rpc.get_worker_info().id - 1
        self.batch_inference = batch_inference
        self.invalid_reward = invalid_reward
    
    def run_episode(
        self,
        agent_rref: rpc.RRef[PPOAgent],
        init_state_str: str,
        len_episode: int,
    ) -> List[SerializableExperience]:
        """Interact many steps with env to collect data.
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
        for i_step in range(len_episode):
            print(f'obs {self.id} step {i_step}')
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
                next_graph_str = '' # TODO
            elif quartz_context.get_xfer_from_id(id=action.xfer).is_nop:
                reward = 0
                game_over = True
                next_nodes = [action.node]
            else:
                reward = (graph.gate_count - next_graph.gate_count) * 3
                next_graph_str = next_graph.to_qasm_str()
            
            exp = SerializableExperience(
                graph.to_qasm_str(), action, reward, next_graph_str, game_over,
            )
            exp_list.append(exp)
            
            if game_over and not self.batch_inference:
                break
            else:
                if self.batch_inference:
                    graph = init_graph
                else:
                    graph = next_graph
        # end for
        return exp_list
            

class QConv(nn.Module):
    def __init__(self, in_feat, inter_dim, out_feat):
        super(QConv, self).__init__()
        self.linear2 = nn.Linear(in_feat + inter_dim, out_feat)
        self.linear1 = nn.Linear(in_feat + 3, inter_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)

    def message_func(self, edges):
        #print(f'node h {edges.src["h"].shape}')
        #print(f'node w {edges.data["w"].shape}')
        return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}

    def reduce_func(self, nodes):
        # print(f'node m {nodes.mailbox["m"].shape}')
        tmp = self.linear1(nodes.mailbox['m'])
        tmp = F.leaky_relu(tmp)
        h = torch.mean(tmp, dim=1)
        # h = torch.max(tmp, dim=1).values
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        #g.edata['w'] = w #self.embed(torch.unsqueeze(w,1))
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        h_linear = self.linear2(h_total)
        h_relu = F.relu(h_linear)
        # h_norm = torch.unsqueeze(torch.linalg.norm(h_relu, dim=1), dim=1)
        # h_normed = torch.divide(h_relu, h_norm)
        # return h_normed
        return h_relu

class QGNN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, inter_dim) -> None:
        super(QGNN, self).__init__()
        self.embedding = nn.Embedding(in_feats, in_feats)
        self.conv_0 = QConv(in_feats, inter_dim, h_feats)
        convs: List[nn.Module] = []
        for _ in range(num_layers - 1):
            convs.append(QConv(h_feats, inter_dim, h_feats))
        self.convs: nn.Module = nn.ModuleList(convs)

    def forward(self, g):
        #print(g.ndata['gate_type'])
        #print(self.embedding)
        g.ndata['h'] = self.embedding(g.ndata['gate_type'])
        w = torch.cat([
            torch.unsqueeze(g.edata['src_idx'], 1),
            torch.unsqueeze(g.edata['dst_idx'], 1),
            torch.unsqueeze(g.edata['reversed'], 1)
        ],
                      dim=1)
        g.edata['w'] = w
        h = self.conv_0(g, g.ndata['h'])
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
        return h


class PPOAgent:
    
    def __init__(
        self,
        agent_id: int,
        num_observers: int,
        device: torch.device,
        batch_inference: bool,
        num_gate_type: int,
        invalid_reward: float,
        graph_embed_size: int,
        actor_hidden_size: int,
        critic_hidden_size: int,
        action_dim: int,
        lr_graph_embedding: float,
        lr_actor: float,
        lr_critic: float,
        
    ) -> None:
        self.id = agent_id
        self.device = device
        """networks related"""
        # TODO may combine these networks into a single module ActorCritic
        # self.graph_embedding = QGNN(6, num_gate_type, graph_embed_size, graph_embed_size)
        # self.actor = nn.Sequential(
        #     nn.Linear(graph_embed_size, actor_hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(actor_hidden_size, action_dim)
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(graph_embed_size, critic_hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(critic_hidden_size, 1)
        # )
        self.naive_model = nn.Linear(64, 64).to(device)
        self.ddp_model = DDP(self.naive_model, [device],)
        
        """init Observers on the other processes and hold the refs to them"""
        self.ob_rrefs: List[rpc.RRef] = []
        for obs_rank in range(0, num_observers):
            ob_info = rpc.get_worker_info(OBS_NAME.format(self.id, obs_rank))
            self.ob_rrefs.append(
                rpc.remote(ob_info, Observer, args=(batch_inference, invalid_reward,))
            )
        
        """helper vars for select_action"""
        self.future_actions: Future[List[Action]] = Future()
        self.pending_states = len(self.ob_rrefs)
        self.states_buf: List[dgl.graph] = [None] * len(self.ob_rrefs)
        
        self.lock = threading.Lock()
        
        # self.buffer = RolloutBuffer()
    
    def select_action(self, obs_id: int, state_str: str) -> Action:
        pygraph: quartz.PyGraph = qasm_to_graph(state_str)
        dgl_graph: dgl.graph = pygraph.to_dgl_graph().to(self.device)
        
        # TODO inference on dgl_graph to get action
        action_node = 0
        action_xfer = 0
        
        return Action(action_node, action_xfer)
        
    @rpc.functions.async_execution
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
                
                # TODO get action for each observer's query
                
                # TODO store actions in obs_id order
                actions = [Action(0, 0) for i in range(len(self.ob_rrefs))]
                self.future_actions.set_result(actions)
                """re-init"""
                self.pending_states = len(self.ob_rrefs)
                self.future_actions = torch.futures.Future()
        return future_action
    
    def update(self):
        pass
    
    def run_iter(
        self,
        i_iter: int,
        len_episode: int,
        input_qasm_str: str,
    ):
        """_summary_

        Args:
            i_iter (int): _description_
            len_episode (int): _description_

        Returns:
            _type_: _description_
        """
        """collect experiences from observers"""
        print(f'Run iter #{i_iter}')
        future_exp_lists: List[Future[List[SerializableExperience]]] = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            future_exp_lists.append(ob_rref.rpc_async().run_episode(
                rpc.RRef(self),
                input_qasm_str,
                len_episode,
            ))
        # wait until all obervers have finished their episode
        exp_lists = torch.futures.wait_all(future_exp_lists)
        """parse collected experiences"""
        
        """train networks"""
        



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
        )
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
        self.max_iterations = int(cfg.max_iterations)
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
        tot_processes = ddp_processes + obs_processes
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method='tcp://localhost:39501'
        )
        
        if rank < ddp_processes:
            dist.init_process_group(
                backend='nccl',
                init_method="tcp://localhost:39500",
                rank=rank, world_size=ddp_processes,
            )
            agent_name = AGENT_NAME.format(rank)
            rpc.init_rpc(
                name=agent_name, rank=rank, world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
            """init agent network"""
            if self.cfg.gpus is None or len(self.cfg.gpus) == 0:
                agent_device = torch.device('cpu')
            else:
                agent_device = torch.device(f'cuda:{self.cfg.gpus[rank]}')
            
            agent = PPOAgent(
                agent_id=rank,
                num_observers=self.cfg.obs_per_agent,
                device=agent_device,
                batch_inference=self.cfg.batch_inference,
                num_gate_type=self.num_gate_type,
                invalid_reward=self.cfg.invalid_reward,
                graph_embed_size=self.cfg.graph_embed_size,
                actor_hidden_size=self.cfg.actor_hidden_size,
                critic_hidden_size=self.cfg.critic_hidden_size,
                action_dim=quartz_context.num_xfers,
                lr_graph_embedding=self.cfg.lr_graph_embedding,
                lr_actor=self.cfg.lr_actor,
                lr_critic=self.cfg.lr_critic,
            )
            print(f'{agent_name} initialized')
            
            
        else:
            obs_rank = rank - ddp_processes
            agent_rref_id = int(obs_rank / self.cfg.obs_per_agent)
            obs_in_agent_rank = int(obs_rank % self.cfg.obs_per_agent)
            obs_name = OBS_NAME.format(agent_rref_id, obs_in_agent_rank)
            rpc.init_rpc(
                name=obs_name, rank=rank, world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
            # print(f'{obs_name} initialized')
        # block until all rpcs finish
        rpc.shutdown()
    

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
