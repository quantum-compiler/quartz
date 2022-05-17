import quartz
import time
import heapq
from concurrent.futures import ProcessPoolExecutor
import json
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import math
from collections import deque


def check(graph):
    graph.to_qasm(filename='check.qasm')
    # qc_origin = QuantumCircuit.from_qasm_file(
    #     'barenco_tof_3_opt_path/subst_history_39.qasm')
    qc_origin = QuantumCircuit.from_qasm_file(
        't_tdg_h_cx_toffoli_flip_dataset/barenco_tof_4_after_toffoli_flip.qasm'
    )
    # qc_origin = QuantumCircuit.from_qasm_file(
    #     '../circuit/nam-circuits/qasm_files/adder_8_before.qasm')
    qc_optimized = QuantumCircuit.from_qasm_file('check.qasm')
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized))


quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg', 'x'],
    filename='../bfs_verified_simplified.json',
    no_increase=True)
parser = quartz.PyQASMParser(context=quartz_context)
init_dag = parser.load_qasm(
    filename=
    't_tdg_h_cx_toffoli_flip_dataset/barenco_tof_4_after_toffoli_flip.qasm')
# init_dag = parser.load_qasm(
#     filename='../circuit/nam-circuits/qasm_files/adder_8_before.qasm')
init_graph = quartz.PyGraph(context=quartz_context, dag=init_dag)

# init_dag = parser.load_qasm(
#     filename='../circuit/nam-circuits/qasm_files/adder_8_before.qasm')
init_graph = quartz.PyGraph(context=quartz_context, dag=init_dag)

candidate_hq = []
heapq.heappush(candidate_hq, init_graph)
hash_set = set()
hash_set.add(init_graph.hash())
best_graph = init_graph
best_gate_cnt = init_graph.gate_count
max_gate_cnt = 64
budget = 5_000_000

start = time.time()

while candidate_hq != [] and budget >= 0:
    first_graph = heapq.heappop(candidate_hq)
    all_nodes = first_graph.all_nodes()
    first_cnt = first_graph.gate_count

    def ax(i):
        node = all_nodes[i]
        return first_graph.available_xfers(context=quartz_context, node=node)

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = executor.map(ax, list(range(len(all_nodes))), chunksize=2)
        appliable_xfers_nodes = []
        for r in results:
            appliable_xfers_nodes.append(r)

    for i in range(len(all_nodes)):
        node = all_nodes[i]
        appliable_xfers = appliable_xfers_nodes[i]
        for xfer in appliable_xfers:
            new_graph = first_graph.apply_xfer(
                xfer=quartz_context.get_xfer_from_id(id=xfer), node=node)
            new_hash = new_graph.hash()
            if new_hash not in hash_set:
                new_cnt = new_graph.gate_count
                hash_set.add(new_hash)
                heapq.heappush(candidate_hq, new_graph)
                if new_cnt < best_gate_cnt:
                    best_graph = new_graph
                    best_gate_cnt = new_cnt
            budget -= 1
            if budget % 10_000 == 0:
                print(
                    f'{budget}: minimum gate count is {best_gate_cnt}, after {time.time() - start:.2f} seconds'
                )
