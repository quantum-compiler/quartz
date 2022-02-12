import qiskit
import numpy as np

from .utils import *
from verifier.gates import compute


def build_circuit(dag):
    dag_meta = dag[0]
    qc = qiskit.QuantumCircuit(dag_meta[meta_index_num_qubits])

    gates = dag[1]
    parameters = [i + 0.1 for i in
                  range(dag_meta[meta_index_num_total_parameters])]  # Suppose the |i|-th parameter is |i| + 0.1
    for gate in gates:
        parameter_values = []
        qubit_indices = []
        for input in gate[2]:
            if input.startswith('P'):
                # parameter input
                parameter_values.append(parameters[int(input[1:])])
            else:
                assert (input.startswith('Q'))
                # qubit input
                qubit_indices.append(int(input[1:]))
        if gate[1][0].startswith('P'):
            # parameter gate
            assert len(gate[1]) == 1
            parameter_index = int(gate[1][0][1:])
            parameters[parameter_index] = compute(gate[0], *parameter_values)
        else:
            assert (gate[1][0].startswith('Q'))
            # quantum gate
            print(f"qc.{gate[0]}({str(parameter_values)[1:-1]}{', ' if len(parameter_values) > 0 else ''}{str(qubit_indices)[1:-1]})")
            exec(
                f"qc.{gate[0]}({str(parameter_values)[1:-1]}{', ' if len(parameter_values) > 0 else ''}{str(qubit_indices)[1:-1]})")

    return qc


def draw_circuit(dag):
    print(build_circuit(dag))
