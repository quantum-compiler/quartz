from config.base_config import *


@dataclass
class IBM2Config(BaseConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'cx',
            'x',
            'sx',
            'rz',
            't',
            'tdg',
            'z',
        ]
    )
    ecc_file: str = '../ecc_set/ibm_special_angles_3_2_5_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/ibm_special_angle_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class IBM2FTConfig(IBM2Config):
    wandb: WandbConfig = WandbConfig.new_project('IBM-Finetune-01')
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


@dataclass
class IBM2PretrainConfig(IBM2Config):
    wandb: WandbConfig = WandbConfig.new_project('IBM-Pretrain')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/ibm_special_angle_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class IBM2MPConfig(IBM2Config):
    wandb: WandbConfig = WandbConfig.new_project('IBM-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../circs/ibm_special_angle_circs/{circ}.qasm',
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
class IBM2TestConfig(TestConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'cx',
            'x',
            'sx',
            'rz',
            't',
            'tdg',
            'z',
        ]
    )
    ecc_file: str = '../ecc_set/ibm_special_angles_3_2_5_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../circs/ibm_special_angle_circs/barenco_tof_3.qasm',
            ),
        ]
    )
