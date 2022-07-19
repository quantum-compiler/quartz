from config.base_config import *

@dataclass
class TdgConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 't', 'tdg',
    ])
    ecc_file: str = '../ecc_set/t_tdg.json.ecc'
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_circs/barenco_tof_3.qasm',
        ),
    ])
    
@dataclass
class TdgFTConfig(TdgConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Finetune')
    k_epochs: int = 10
    greedy_sample: bool = True
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_circs/barenco_tof_3.qasm',
        ),
    ])

@dataclass
class TdgMPConfig(TdgConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_circs/barenco_tof_3.qasm',
        ),
        InputGraph(
            'vbe_adder_3',
            '../t_tdg_circs/vbe_adder_3.qasm',
        ),
        InputGraph(
            'mod5_4',
            '../t_tdg_circs/mod5_4.qasm',
        ),
        InputGraph(
            'mod_mult_55',
            '../t_tdg_circs/mod_mult_55.qasm',
        ),
        InputGraph(
            'tof_5',
            '../t_tdg_circs/tof_5.qasm',
        ),
        InputGraph(
            'gf2^4_mult',
            '../t_tdg_circs/gf2^4_mult.qasm',
        ),
    ])

@dataclass
class TdgRMMPConfig(TdgConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_rm_circs/barenco_tof_3.qasm',
        ),
        InputGraph(
            'vbe_adder_3',
            '../t_tdg_rm_circs/vbe_adder_3.qasm',
        ),
        InputGraph(
            'mod5_4',
            '../t_tdg_rm_circs/mod5_4.qasm',
        ),
        InputGraph(
            'mod_mult_55',
            '../t_tdg_rm_circs/mod_mult_55.qasm',
        ),
        InputGraph(
            'tof_5',
            '../t_tdg_rm_circs/tof_5.qasm',
        ),
        InputGraph(
            'gf2^4_mult',
            '../t_tdg_rm_circs/gf2^4_mult.qasm',
        ),
    ])

