from config.base_config import *

@dataclass
class NamConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 'x', 'rz', 'add',
    ])
    ecc_file: str = '../nam.json.ecc'
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../nam_circs/barenco_tof_3.qasm',
        ),
    ])

@dataclass
class NamPretrainConfig(NamConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain')
    # quartz
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../nam_circs/barenco_tof_3.qasm',
        ),
    ])
