import copy
import heapq
import json
import time
from concurrent.futures import ProcessPoolExecutor

from dgl import load_graphs, save_graphs
from pos_data import PosRewardData
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import quartz


def check(graph):
    graph.to_qasm(filename='best.qasm')
    qc_origin = QuantumCircuit.from_qasm_file(
        'barenco_tof_3_opt_path/subst_history_39.qasm'
    )
    qc_optimized = QuantumCircuit.from_qasm_file('best.qasm')
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized)
    )


quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'], filename='../bfs_verified_simplified.json'
)
parser = quartz.PyQASMParser(context=quartz_context)
my_dag = parser.load_qasm(filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_graph = quartz.PyGraph(context=quartz_context, dag=my_dag)

candidate_hq = []
heapq.heappush(candidate_hq, init_graph)
hash_set = set()
hash_set.add(init_graph.hash())
best_graph = init_graph
best_gate_cnt = init_graph.gate_count

budget = 5_000_000

buffer = PosRewardData(50)
finish = False
start = time.time()

while candidate_hq != [] and budget >= 0 and not finish:
    first_candidate = heapq.heappop(candidate_hq)
    all_nodes = first_candidate.all_nodes()
    first_cnt = first_candidate.gate_count

    def ax(i):
        node = all_nodes[i]
        return first_candidate.available_xfers(context=quartz_context, node=node)

    with ProcessPoolExecutor(max_workers=64) as executor:
        results = executor.map(ax, list(range(len(all_nodes))), chunksize=2)
        appliable_xfers_nodes = []
        for r in results:
            appliable_xfers_nodes.append(r)

    for i in range(len(all_nodes)):
        node = all_nodes[i]
        appliable_xfers = appliable_xfers_nodes[i]
        for xfer in appliable_xfers:
            new_graph = first_candidate.apply_xfer(
                xfer=quartz_context.get_xfer_from_id(id=xfer), node=node
            )
            new_hash = new_graph.hash()
            if new_hash not in hash_set:
                hash_set.add(new_hash)
                heapq.heappush(candidate_hq, new_graph)
                new_cnt = new_graph.gate_count
                if new_cnt < best_gate_cnt:
                    best_graph = new_graph
                    best_gate_cnt = new_cnt
                if new_cnt < first_cnt:
                    if buffer.add_data(
                        first_candidate.to_dgl_graph(),
                        first_candidate.hash(),
                        i,
                        xfer,
                        first_cnt - new_cnt,
                        new_graph.to_dgl_graph(),
                        new_graph.hash(),
                    ):
                        print(f'Collected data count: {buffer.data_cnt}')
                    else:
                        finish = True
                budget -= 1
                if budget % 10_000 == 0:
                    print(
                        f'{budget}: minimum gate count is {best_gate_cnt}, after {time.time() - start:.2f} seconds'
                    )

buffer.save_data()
