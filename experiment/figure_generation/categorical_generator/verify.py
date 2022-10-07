import json

import quartz
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def check(graph):
    graph.to_qasm(filename='check.qasm')
    qc_origin = QuantumCircuit.from_qasm_file("../barenco_tof_3.qasm")
    qc_optimized = QuantumCircuit.from_qasm_file('check.qasm')
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized))


def main():
    # initialize
    file_prefix_list = [46, 48, 50, 51, 52, 53, 54, 56, 58]
    quartz_context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg'],
                                          filename="../bfs_verified_simplified.json",
                                          no_increase=True, include_nop=False)
    qasm_parser = quartz.PyQASMParser(context=quartz_context)

    # verify
    for file_prefix in file_prefix_list:
        print(f"Start check dataset {file_prefix}.")
        checked_circuit = 0
        with open(f"./output/{file_prefix}.json", 'r') as handle:
            circuit_dict = json.load(handle)
            for circuit_hash in circuit_dict:
                # unpack
                qasm_str = circuit_dict[circuit_hash][0]
                gate_count = circuit_dict[circuit_hash][1]

                # restore to circuit
                dag = qasm_parser.load_qasm_str(qasm_str=qasm_str)
                cur_graph = quartz.PyGraph(context=quartz_context, dag=dag)

                # check gate count
                assert gate_count == file_prefix and cur_graph.gate_count == file_prefix

                # check equality with original graph
                assert check(cur_graph)

                # log
                checked_circuit += 1
                if (checked_circuit % 1000) == 0:
                    print(f"Finished {checked_circuit} circuits.")


if __name__ == '__main__':
    main()
