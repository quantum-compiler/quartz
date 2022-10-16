from config.base_config import *


@dataclass
class RigConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'rx1',
            'rx3',
            'x',
            'rz',
            'cz',
            'add',
        ]
    )
    ecc_file: str = '../ecc_set/rigetti_5_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../rigetti_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class RigFTConfig(RigConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Finetune-Rig-01')
    greedy_sample: bool = True
    k_epochs: int = 20
    lr_gnn: float = 3e-5
    lr_actor: float = 3e-5
    lr_critic: float = 5e-5
    lr_scheduler: str = 'linear'
    lr_start_factor: float = 0.1
    lr_warmup_epochs: int = 50
    resume_optimizer: bool = False
    num_eps_per_iter: int = 64
    max_eps_len: int = 600
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../rigetti_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class RigPretrainConfig(RigConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../rigetti_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class RigMPConfig(RigConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../rigetti_circs/{circ}.qasm',
            )
            for circ in [
                'barenco_tof_3',
                'vbe_adder_3',
                'mod5_4',
                'mod_mult_55',
                'tof_5',
                'gf2^4_mult',
            ]
        ]
    )


@dataclass
class RigRMMPConfig(RigConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../rigetti_rm_circs/{circ}.qasm',
            )
            for circ in [
                'barenco_tof_3',
                'vbe_adder_3',
                'mod5_4',
                'mod_mult_55',
                'tof_5',
                'gf2^4_mult',
            ]
        ]
    )
