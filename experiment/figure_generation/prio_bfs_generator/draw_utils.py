from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def check(graph):
    graph.to_qasm(filename='check.qasm')
    qc_origin = QuantumCircuit.from_qasm_file(
        'barenco_tof_3_opt_path/subst_history_39.qasm')
    qc_optimized = QuantumCircuit.from_qasm_file('check.qasm')
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized))
