from typing import List, Tuple, cast

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import qtz

from icecream import ic

qtz.init_quartz_context(
    ['h', 'cx', 'x', 'rz', 'add',],
    '../../ecc_set/nam.json.ecc',
    False,
    True,
)
ic(qtz.quartz_context.num_xfers)

def check_sv_eq(qc_x: QuantumCircuit, qc_y: QuantumCircuit) -> bool:
    return Statevector.from_instruction(qc_x).equiv(
        Statevector.from_instruction(qc_y)
    )

def check_sv_eq_from_path(path_x: str, path_y: str) -> bool:
    qc_x = QuantumCircuit.from_qasm_file(path_x)
    qc_y = QuantumCircuit.from_qasm_file(path_y)
    return check_sv_eq(qc_x, qc_y)

def check_op_eq(qc_x: QuantumCircuit, qc_y: QuantumCircuit) -> bool:
    return Operator(qc_x).equiv(
       Operator(qc_y)
    )

def check_op_eq_from_path(path_x: str, path_y: str) -> bool:
    qc_x = QuantumCircuit.from_qasm_file(path_x)
    qc_y = QuantumCircuit.from_qasm_file(path_y)
    return check_op_eq(qc_x, qc_y)

if __name__ == '__main__':
    import sys

    check_seq = sys.argv[1] == 'seq'

    if not check_seq:
        file_x = sys.argv[2]
        file_y = sys.argv[3]

        ic(check_sv_eq_from_path(file_x, file_y))
        ic(check_op_eq_from_path(file_x, file_y))
    else:
        import os
        from natsort import natsorted
        from tqdm import tqdm

        def filename_to_info(qc_file):
            qc_file_root = os.path.splitext(qc_file)[0]
            return list(map(int, qc_file_root.split('_')))

        seq_dir = sys.argv[2]
        qc_files = cast(List[str], natsorted(os.listdir(seq_dir)))
        last_qc_path: str | None = None
        last_action: Tuple[int, int] = (0, 0)
        for qc_file in tqdm(qc_files):
            # 0_36_-2_28_78.qasm: i_step, cost, reward, action_node, action_xfer
            i_step, cost, reward, action_node, action_xfer = filename_to_info(qc_file)
            full_qc_path = os.path.join(seq_dir, qc_file)
            if last_qc_path is not None:
                sv_eq = check_sv_eq_from_path(last_qc_path, full_qc_path)
                op_eq = check_op_eq_from_path(last_qc_path, full_qc_path)
                if not (sv_eq and op_eq):
                    ic(sv_eq, op_eq)
                    print(f'Non equivalent circs: {last_qc_path}, {full_qc_path}')
                    xfer = qtz.quartz_context.get_xfer_from_id(id=last_action[1])
                    print(
                        f'{full_qc_path} is transformed from {last_qc_path} by\n'
                        f'Action{last_action} {xfer}:\n'
                        f'src:\n{xfer.src_str}\ndst:\n{xfer.dst_str}'
                    )

            # end if
            last_qc_path = full_qc_path
            last_action = (action_node, action_xfer)