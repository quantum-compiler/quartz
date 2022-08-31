import sys

from qiskit import QuantumCircuit, transpile


def draw_from_qasm(qasm_file, **kwargs):
    circuit = QuantumCircuit.from_qasm_file(qasm_file)
    print(circuit.draw())
    circuit.draw(circuit, kwargs)


if __name__ == '__main__':
    draw_from_qasm(sys.argv[1], fliename=sys.argv[2])
