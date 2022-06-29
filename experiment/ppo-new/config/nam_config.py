from config.base_config import *

@dataclass
class NamConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 'x', 'rz', 'add',
    ])
    ecc_file: str = '../ecc_set/nam.json.ecc'
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

@dataclass
class NamMultiPretrainConfig(NamConfig):
    wandb: WandbConfig = WandbConfig.new_project('PPO-Pretrain-Multi')
    # quartz
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../nam_circs/barenco_tof_3.qasm',
        ),
        InputGraph(
            'vbe_adder_3',
            '../nam_circs/vbe_adder_3.qasm',
        ),
        InputGraph(
            'mod5_4',
            '../nam_circs/mod5_4.qasm',
        ),
        InputGraph(
            'mod_mult_55',
            '../nam_circs/mod_mult_55.qasm',
        ),
        InputGraph(
            'tof_5',
            '../nam_circs/tof_5.qasm',
        ),
    ])

@dataclass
class NamTestConfig(TestConfig):
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 'x', 'rz', 'add',
    ])
    ecc_file: str = '../ecc_set/nam.json.ecc'
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../nam_circs/barenco_tof_3.qasm',
        ),
        # InputGraph(
        #     'qcla_mod_7',
        #     '../nam_circs/qcla_mod_7.qasm',
        # ),
    ])

@dataclass
class NamAllTestConfig(NamTestConfig):
    input_graphs: List[InputGraph] = field(default_factory=lambda:[])
    input_graph_dir: str = '../nam_circs'
    budget: int = 4000

@dataclass
class NamConvertConfig(ConvertConfig):
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 'x', 'rz', 'add',
    ])
    ecc_file: str = '../ecc_set/nam.json.ecc'
