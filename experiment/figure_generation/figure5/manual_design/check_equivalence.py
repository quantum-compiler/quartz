from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def check():
    qc_origin = QuantumCircuit.from_qasm_file("./qasm/source.qasm")
    qc_optimized = QuantumCircuit.from_qasm_file("./qasm/target.qasm")
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized))


if __name__ == '__main__':
    res = check()
    print(f"Is source and target equivalent = {res}.")
