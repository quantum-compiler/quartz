import os
import random
import warnings

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

def seed_all(seed: int) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_trajectory(i):
    return i

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
        quartz_context = quartz.QuartzContext(
            gate_set=cfg.gate_set,
            filename=cfg.ecc_file,
            no_increase=cfg.no_increase,
            include_nop=cfg.include_nop,
        )
        self.context = quartz_context
        self.num_gate_type = 29
        self.parser = quartz.PyQASMParser(context=self.context)
        init_dag = self.parser.load_qasm(filename=cfg.init_graph_path)
        self.init_graph = quartz.PyGraph(context=quartz_context, dag=init_dag)
        
        # init training related things
        self.max_iterations = int(cfg.max_iterations)
        
        
        
    
    def print_cfg(self) -> None:
        print('================ Configs ================')
        for k, v in self.cfg.items():
            print(f'{k} : {v}')
        print(f'output_dir : {self.output_dir}')
        print('=========================================')
    
    def train(self) -> None:
        for i_iteration in range(self.max_iterations):
            with mp.Pool(processes=4) as p:
                ts = p.map(get_trajectory, list(range(32)))
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
