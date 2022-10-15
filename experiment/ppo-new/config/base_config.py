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
    resume_optimizer: bool = True
    ckpt_path: str = 'outputs/2022-07-28/15-47-54/ckpts/iter_2.pt'
    load_non_ddp_ckpt: bool = True
    load_best_info: bool = False
    best_info_dir: str = 'outputs/2022-06-26/08-45-11/sync_dir'

    gpus: List[int] = field(
        default_factory=lambda: [
            0,
            1,
            2,
            3,
        ]
    )
    ddp_port: int = 23333
    omp_num_threads: int = 4

    seed: int = 98765
    wandb: WandbConfig = WandbConfig()

    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            't',
            'tdg',
        ]
    )
    ecc_file: str = '../ecc_set/t_tdg_ecc.json'
    no_increase: bool = False
    include_nop: bool = True
    num_gate_types: int = 40
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../t_tdg_circs/barenco_tof_3.qasm',
            ),
        ]
    )

    # network
    gnn_type: str = 'QGNN'  # 'QGNN' or 'QGIN'
    gate_type_embed_dim: int = 16
    gnn_num_layers: int = 6
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 128
    gin_num_mlp_layers: int = 2
    gin_learn_eps: bool = False
    gin_neighbor_pooling_type: str = 'sum'  # 'mean', 'max'
    gin_graph_pooling_type: str = 'none'  # 'sum', 'mean', 'max'
    actor_hidden_size: int = 256
    critic_hidden_size: int = 128

    # algorithm
    gamma: float = 0.95
    entropy_coeff: float = 0.02
    eps_clip: float = 0.2
    softmax_temp_en: bool = True
    hit_rate: float = 0.9

    # multiprocessing
    mp_start_method: str = 'spawn'  # fork
    obs_per_agent: int = 0

    # exp collection
    cost_type: str = 'gate_count'
    nop_stop: bool = True
    invalid_reward: float = -1.0
    max_cost_ratio: float = 1.2
    limit_total_gate_count: bool = False
    batch_inference: bool = True
    dyn_eps_len: bool = True
    max_eps_len: int = 300
    min_eps_len: int = 20
    greedy_sample: bool = False
    agent_collect: bool = True
    agent_batch_size: int = 128
    subgraph_opt: bool = True
    # training
    max_iterations: int = int(1e8)
    num_eps_per_iter: int = 128  # 30
    mini_batch_size: int = 3840  # per DDP process; < num_eps_per_iter * len_episode
    k_epochs: int = 25
    lr_gnn: float = 3e-4
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    lr_scheduler: str = 'none'  # linear
    lr_start_factor: float = 0.1
    lr_warmup_epochs: int = 50
    update_policy_interval: int = 1
    save_ckpt_interval: int = 1
    time_budget: str = ''

    # logging
    best_graph_output_dir: str = 'best_graphs'
    output_full_seq: bool = False
    full_seq_path: str = ''  # for read in


@dataclass
class TestConfig(BaseConfig):
    wandb: WandbConfig = WandbConfig(False)
    topk: int = 3
    mode: str = 'test'
    resume: bool = True
    budget: int = int(1e8)
    input_graphs: List[InputGraph] = field(default_factory=lambda: [])
    input_graph_dir: str = '../nam_circs'


@dataclass
class ConvertConfig(BaseConfig):
    wandb: WandbConfig = WandbConfig(False)
    mode: str = 'convert'
    resume: bool = True
    ckpt_output_dir: str = '.'
    ckpt_output_path: str = ''
