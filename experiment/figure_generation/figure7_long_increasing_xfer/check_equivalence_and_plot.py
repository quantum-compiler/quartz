from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator


def check_equivalence():
    qc_1 = QuantumCircuit.from_qasm_file('./circuits/step_1.qasm')
    qc_2 = QuantumCircuit.from_qasm_file('./circuits/step_2.qasm')
    qc_3 = QuantumCircuit.from_qasm_file('./circuits/step_3.qasm')
    qc_4 = QuantumCircuit.from_qasm_file('./circuits/step_4.qasm')
    qc_5 = QuantumCircuit.from_qasm_file('./circuits/step_5.qasm')
    qc_6 = QuantumCircuit.from_qasm_file('./circuits/step_6.qasm')
    print(f"Check circuit 1 and circuit 2 are equivalent:",
          Statevector.from_instruction(qc_1).equiv(Statevector.from_instruction(qc_2)),
          Operator(qc_1).equiv(Operator(qc_2)))
    print(f"Check circuit 1 and circuit 3 are equivalent:",
          Statevector.from_instruction(qc_1).equiv(Statevector.from_instruction(qc_3)),
          Operator(qc_1).equiv(Operator(qc_3)))
    print(f"Check circuit 1 and circuit 4 are equivalent:",
          Statevector.from_instruction(qc_1).equiv(Statevector.from_instruction(qc_4)),
          Operator(qc_1).equiv(Operator(qc_4)))
    print(f"Check circuit 1 and circuit 5 are equivalent:",
          Statevector.from_instruction(qc_1).equiv(Statevector.from_instruction(qc_5)),
          Operator(qc_1).equiv(Operator(qc_5)))
    print(f"Check circuit 1 and circuit 6 are equivalent:",
          Statevector.from_instruction(qc_1).equiv(Statevector.from_instruction(qc_6)),
          Operator(qc_1).equiv(Operator(qc_6)))


def plot():
    qc_1 = QuantumCircuit.from_qasm_file('./circuits/step_1.qasm')
    qc_2 = QuantumCircuit.from_qasm_file('./circuits/step_2.qasm')
    qc_3 = QuantumCircuit.from_qasm_file('./circuits/step_3.qasm')
    qc_4 = QuantumCircuit.from_qasm_file('./circuits/step_4.qasm')
    qc_5 = QuantumCircuit.from_qasm_file('./circuits/step_5.qasm')
    qc_6 = QuantumCircuit.from_qasm_file('./circuits/step_6.qasm')

    qc_1.draw(output='mpl', filename='./circuits_figs/step_1.jpg')
    qc_2.draw(output='mpl', filename='./circuits_figs/step_2.jpg')
    qc_3.draw(output='mpl', filename='./circuits_figs/step_3.jpg')
    qc_4.draw(output='mpl', filename='./circuits_figs/step_4.jpg')
    qc_5.draw(output='mpl', filename='./circuits_figs/step_5.jpg')
    qc_6.draw(output='mpl', filename='./circuits_figs/step_6.jpg')


if __name__ == '__main__':
    check_equivalence()
    plot()
