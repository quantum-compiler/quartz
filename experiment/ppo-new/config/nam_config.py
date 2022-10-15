from config.base_config import *


@dataclass
class NamConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            'x',
            'rz',
            'add',
        ]
    )
    ecc_file: str = '../ecc_set/nam_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../nam_circs/barenco_tof_3.qasm',
            ),
        ]
    )
    num_gate_types: int = 29


@dataclass
class NamFTConfig(NamConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Finetune-03')
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
                '../nam_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class NamPretrainConfig(NamConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../nam_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class NamMPConfig(NamConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../nam_circs/{circ}.qasm',
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
class NamRMMPConfig(NamConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../nam_rm_circs/{circ}.qasm',
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
class NamTestConfig(TestConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            'x',
            'rz',
            'add',
        ]
    )
    ecc_file: str = '../ecc_set/nam_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../nam_circs/barenco_tof_3.qasm',
            ),
            # InputGraph(
            #     'qcla_mod_7',
            #     '../nam_circs/qcla_mod_7.qasm',
            # ),
        ]
    )


@dataclass
class NamAllTestConfig(NamTestConfig):
    input_graphs: List[InputGraph] = field(default_factory=lambda: [])
    input_graph_dir: str = '../nam_circs'
    budget: int = 4000


@dataclass
class NamConvertConfig(ConvertConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            'x',
            'rz',
            'add',
        ]
    )
    ecc_file: str = '../ecc_set/nam_ecc.json'
