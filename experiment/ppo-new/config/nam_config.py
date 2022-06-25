from config.base_config import *

@dataclass
class NamConfig(BaseConfig):
    # quartz
    gate_set: List[str] = field(default_factory=lambda: [
        'h', 'cx', 'x', 'rz', 'add',
    ])
    ecc_file: str = 'Nam_complete_ECC_set.json'
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_h_cx_toffoli_flip_dataset/barenco_tof_3.qasm.toffoli_flip',
        ),
    ])

@dataclass
class NamPretrainConfig(NamConfig):
    # quartz
    input_graphs: List[InputGraph] = field(default_factory=lambda:[
        InputGraph(
            'barenco_tof_3',
            '../t_tdg_h_cx_toffoli_flip_dataset/barenco_tof_3.qasm.toffoli_flip',
        ),
    ])
