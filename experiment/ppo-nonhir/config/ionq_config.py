from config.base_config import *


@dataclass
class IonQConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'rx1',
            'x',
            'rx3',
            'ry1',
            'y',
            'ry3',
            'rxx1',
            'rxx3',
            'rz',
            'add',
        ]
    )
    ecc_file: str = '../ecc_set/ionq_4_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'dj_nativegates_ionq_qiskit_opt0_10_norm',
                '../ionq_circs/dj_nativegates_ionq_qiskit_opt0_10_norm.qasm',
            ),
        ]
    )


@dataclass
class IonQFTConfig(IonQConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Finetune-IonQ-01')
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
                'dj_nativegates_ionq_qiskit_opt0_10_norm',
                '../ionq_circs/dj_nativegates_ionq_qiskit_opt0_10_norm.qasm',
            ),
        ]
    )


@dataclass
class IonQPretrainConfig(IonQConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'dj_nativegates_ionq_qiskit_opt0_10_norm',
                '../ionq_circs/dj_nativegates_ionq_qiskit_opt0_10_norm.qasm',
            ),
        ]
    )


@dataclass
class IonQMPConfig(IonQConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../ionq_circs/{circ}.qasm',
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
