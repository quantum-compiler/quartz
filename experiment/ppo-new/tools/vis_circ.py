import os
from typing import Any

from genericpath import isfile
from qiskit import QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from tqdm import tqdm


def vis_qc(qc: QuantumCircuit, out_path: str | None = None) -> Any:
    return qc.draw(output='mpl', filename=out_path)


if __name__ == '__main__':
    import sys

    if os.path.isfile(sys.argv[1]):
        inp_circ = sys.argv[1]
        inp_circ_name = os.path.basename(inp_circ).split('.')[0]

        if len(sys.argv) > 2:
            out_path = sys.argv[2]
        else:
            out_path = f'{inp_circ_name}.png'

        vis_qc(QuantumCircuit.from_qasm_file(sys.argv[1]), out_path)
    else:
        assert os.path.isdir(sys.argv[1])
        inp_dir = sys.argv[1]
        if len(sys.argv) > 2:
            out_dir = sys.argv[2]
        else:
            out_dir = f'{inp_dir}_vis'
        os.makedirs(out_dir, exist_ok=True)
        circs = os.listdir(inp_dir)
        for circ in tqdm(circs):
            circ_name = circ.split('.')[0]
            inp_file = os.path.join(inp_dir, circ)
            out_file = os.path.join(out_dir, f'{circ_name}.png')
            vis_qc(QuantumCircuit.from_qasm_file(inp_file), out_file)
