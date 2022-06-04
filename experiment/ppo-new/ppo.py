from __future__ import annotations
import os
import random
from typing import Tuple, List
import warnings
from collections import deque, namedtuple
from functools import partial

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import quartz # type: ignore
import numpy as np

import hydra
import wandb

from IPython import embed # type: ignore

# global vars to avoid serialization when multiprocessing
quartz_context: quartz.QuartzContext
quartz_parser: quartz.PyQASMParser
shared_net: nn.Module

def init_shared_net(net: nn.Module) -> None:
    global shared_net
    shared_net = net

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
    ['state', 'action', 'reward', 'next_state', 'game_over']
)

SerializableExperience = namedtuple(
    'SerializableExperience',
    ['state', 'action', 'reward', 'next_state', 'game_over']
)

class RolloutBuffer:
    
    def __init__(self) -> None:
        self.exps: List[Experience] = []
    
    def append(self, exp: Experience) -> RolloutBuffer:
        self.exps.append(exp)
        return self
    

class PPOAgent():
    
    def __init__(self) -> None:
        # self.actor_net = torch.rand(3, 5).cuda(2)
        # self.buffer = RolloutBuffer()
        pass
    
    def select_action(self, graph: quartz.PyGraph) -> Tuple[int, int]:
        
        return 0, 0


def qasm_to_graph(qasm_str: str) -> quartz.PyGraph:
    global quartz_context
    global quartz_parser
    dag = quartz_parser.load_qasm_str(qasm_str)
    graph = quartz.PyGraph(context=quartz_context, dag=dag)
    return graph
    

def get_trajectory(
    init_graph_str: str,
    agent: PPOAgent,
    max_steps: int = 300,
    invalid_reward: float = 1.0,
) -> List[SerializableExperience]:
    global quartz_context
    
    graph = qasm_to_graph(init_graph_str)
    
    exp_list: List[SerializableExperience] = []
    trajectory_reward: float = 0.
    
    for i_step in range(max_steps):
        node, xfer = agent.select_action(graph)
        # next_graph: quartz.PyGraph
        next_nodes: List[int]
        next_graph, next_nodes = \
            graph.apply_xfer_with_local_state_tracking(
                xfer=quartz_context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node)
            )
        
        if next_graph is None:
            reward = invalid_reward
            game_over = True
            next_graph_str = ''
        elif quartz_context.get_xfer_from_id(id=xfer).is_nop:
            reward = 0
            game_over = True
            next_nodes = [node]
        else:
            reward = (graph.gate_count - next_graph.gate_count) * 3
            next_graph_str = next_graph.to_qasm_str()
        
        trajectory_reward += reward
        
        exp = SerializableExperience(
            graph.to_qasm_str(), (node, xfer), reward, next_graph_str, game_over,
        )
        exp_list.append(exp)
        
        if game_over:
            break
    # end for 
    

    return exp_list
    
    

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
        
        # init quartz
        global quartz_context
        global quartz_parser
        self.init_quartz_context_func = partial(
            init_quartz_context,
            cfg.gate_set,
            cfg.ecc_file,
            cfg.no_increase,
            cfg.include_nop,
        )
        self.init_quartz_context_func()
        self.context = quartz_context
        self.parser = quartz_parser
        with open(cfg.init_graph_path) as f:
            qasm_str = f.read()
        self.init_graph = qasm_to_graph(qasm_str)
        self.num_gate_type = 29
        
        # init training related parameters
        self.max_iterations = int(cfg.max_iterations)
        self.collect_batch = int(cfg.collect_batch)
        mp.set_start_method('spawn')
        
        # networks
        self.agent = PPOAgent()
        
        
        
    
    def print_cfg(self) -> None:
        print('================ Configs ================')
        for k, v in self.cfg.items():
            print(f'{k} : {v}')
        print(f'output_dir : {self.output_dir}')
        print('=========================================')
    
    def train(self) -> None:
        for i_iteration in range(self.max_iterations):
            
            
            pf_get_trajectory = partial(
                get_trajectory,
                agent=self.agent,
                max_steps=self.cfg.max_steps,
                invalid_reward=self.cfg.invalid_reward,
            )
            
            init_graphs: List[str] = [self.init_graph.to_qasm_str()] * self.cfg.collect_batch
            with mp.Pool(
                processes=4, #self.cfg.num_workers,
                initializer=self.init_quartz_context_func
            ) as pool: # we use spawn to support CUDA here, so we need to init global vars in each process again
                sr_exp_lists = pool.map(pf_get_trajectory, init_graphs)
            
            embed()
    

@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    output_dir = os.path.abspath(os.curdir) # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd()) # set working dir to the original one
    
    warnings.simplefilter('ignore')
    
    ppo_mod = PPOMod(cfg, output_dir)
    ppo_mod.train()
    

if __name__ == '__main__':
    main()
