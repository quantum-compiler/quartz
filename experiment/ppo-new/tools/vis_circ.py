from typing import Any

from qiskit import QuantumCircuit
from qiskit.tools.visualization import circuit_drawer


def vis_qc(qc: QuantumCircuit, filename: str | None = None) -> Any:
    return qc.draw(output='mpl', filename=filename)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = 'qc.img'

    vis_qc(QuantumCircuit.from_qasm_file(sys.argv[1]), filename)
