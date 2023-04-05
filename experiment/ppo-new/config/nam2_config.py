from config.base_config import *


@dataclass
class Nam2Config(BaseConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            'x',
            'rz',
            'add',
            'neg',
        ]
    )
    ecc_file: str = '../ecc_set/nam_325_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/nam_circs/barenco_tof_3.qasm',
            ),
        ]
    )
    # num_gate_types: int = 29


@dataclass
class Nam2FTConfig(Nam2Config):
    wandb: WandbConfig = WandbConfig.new_project('Nam2-Finetune-04')
    greedy_sample: bool = True
    k_epochs: int = 20
    lr_gnn: float = 3e-4
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    lr_scheduler: str = 'linear'
    lr_start_factor: float = 0.1
    lr_warmup_epochs: int = 15
    resume_optimizer: bool = False
    num_eps_per_iter: int = 64
    max_eps_len: int = 600
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/nam_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class Nam2PretrainConfig(Nam2Config):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/nam_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class Nam2MPConfig(Nam2Config):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../circs/nam_circs/{circ}.qasm',
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
class Nam2RMMPConfig(Nam2Config):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../circs/nam_rm_circs/{circ}.qasm',
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
class Nam2TestConfig(TestConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            'x',
            'rz',
            'add',
            'neg',
        ]
    )
    ecc_file: str = '../ecc_set/nam_325_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/nam_circs/barenco_tof_3.qasm',
            ),
            # InputGraph(
            #     'qcla_mod_7',
            #     '../nam_circs/qcla_mod_7.qasm',
            # ),
        ]
    )
