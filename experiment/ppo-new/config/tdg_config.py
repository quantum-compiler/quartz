from config.base_config import *


@dataclass
class TdgConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(
        default_factory=lambda: [
            'h',
            'cx',
            't',
            'tdg',
            'x',
        ]
    )
    ecc_file: str = '../ecc_set/t_tdg_ecc.json'
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../t_tdg_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class TdgFTConfig(TdgConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Finetune-cx')
    k_epochs: int = 10
    greedy_sample: bool = True
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                'barenco_tof_3',
                '../t_tdg_circs/barenco_tof_3.qasm',
            ),
        ]
    )


@dataclass
class TdgMPConfig(TdgConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../t_tdg_circs/{circ}.qasm',
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
class TdgRMMPConfig(TdgConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(
        default_factory=lambda: [
            InputGraph(
                name=f'{circ}',
                path=f'../t_tdg_rm_circs/{circ}.qasm',
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
