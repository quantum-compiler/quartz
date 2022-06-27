from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING, OmegaConf  # Do not confuse with dataclass.MISSING

@dataclass
class WandbConfig:
    en: bool = True
    offline: bool = False
    entity: str = 'quartz'
    project: str = 'PPO'
    
    @staticmethod
    def new_project(proj: str) -> WandbConfig:
        config = WandbConfig()
        config.project = proj
        return config

@dataclass
class InputGraph:
    name: str
    path: str

@dataclass
class BaseConfig:
    
    mode: str = 'train'
    resume: bool = False
    ckpt_path: str = 'outputs/2022-06-21/17-33-40/ckpts/iter_174.pt'
    load_best_info: bool = False
    best_info_dir: str = 'outputs/2022-06-26/08-45-11/sync_dir'

    gpus: List[int] = field(default_factory=lambda: [
        0, 1, 2, 3, 
    ])
    ddp_port: int = 23333
    omp_num_threads: int = 1

    seed: int = 98765
    wandb: WandbConfig = WandbConfig()
    
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 't', 'tdg',
    ])
    ecc_file: str = '../ecc_set/t_tdg.json.ecc'
    no_increase: bool = False
    include_nop: bool = True
    num_gate_type: int = 29
    
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_circs/barenco_tof_3.qasm',
        ),
    ])
    
    # network
    graph_embed_size: int = 128
    actor_hidden_size: int = 256
    critic_hidden_size: int = 128

    # algorithm
    gamma: float = 0.95
    entropy_coeff: float = 0.02
    eps_clip: float = 0.2
    softmax_temp_en: bool = True
    hit_rate: float = 0.9

    # multiprocessing
    mp_start_method: str = 'spawn' # fork
    obs_per_agent: int = 3

    # exp collection
    nop_stop: bool = True
    invalid_reward: float = -1.0
    max_gate_count_ratio: float = 1.2
    batch_inference: bool = True
    agent_collect: bool = False
    dyn_eps_len: bool = True
    max_eps_len: int = 300
    min_eps_len: int = 20

    # training
    max_iterations: int = int(1e8)
    num_eps_per_iter: int = 30
    mini_batch_size: int = 3600 # per DDP process; < num_eps_per_iter * len_episode
    k_epochs: int = 25
    lr_graph_embedding: float = 3e-4
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    update_policy_interval: int = 1
    save_ckpt_interval: int = 5



