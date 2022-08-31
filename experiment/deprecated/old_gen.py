import heapq
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import quartz


def check(graph):
    graph.to_qasm(filename='check.qasm')
    qc_origin = QuantumCircuit.from_qasm_file(
        'barenco_tof_3_opt_path/subst_history_39.qasm'
    )
    qc_optimized = QuantumCircuit.from_qasm_file('check.qasm')
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized)
    )


def get_graph_from_hash(context, h):
    s = str(h)
    parser = quartz.PyQASMParser(context=context)
    my_dag = parser.load_qasm(filename=f"dataset/{s}.qasm")
    return quartz.PyGraph(context=context, dag=my_dag)


def save_graph(graph):
    h = graph.hash()
    dir_path = f'dataset'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    graph.to_qasm(filename=f'dataset/{str(h)}.qasm')


class DataBuffer:
    def __init__(self, init_graph, gamma) -> None:
        init_hash = init_graph.hash()
        init_value = init_graph.gate_count

        self.reward_map = {}
        self.path_map = {}
        self.gate_count_map = {}

        self.gate_count_map[init_hash] = init_value
        self.path_map[init_hash] = {}  # stores predecessors

        self.origin_value = init_value
        self.gamma = gamma

    def update_path(self, pre_graph, node_id, xfer_id, graph):
        # may introduce loops; it's the caller's responsibility to handle it
        graph_hash = graph.hash()
        pre_graph_hash = pre_graph.hash()

        # clean  impossible of no_increase=True
        # graph_gate_cnt = graph.gate_count
        # pre_graph_gate_cnt = pre_graph.gate_count
        # if pre_graph_gate_cnt < graph_gate_cnt:
        #     print(f'!!! in update_path: {pre_graph_gate_cnt} < {graph_gate_cnt}')
        #     assert(False)

        if graph_hash not in self.path_map:
            self.path_map[graph_hash] = {pre_graph_hash: (node_id, xfer_id)}
        else:
            if pre_graph_hash not in self.path_map[graph_hash]:
                self.path_map[graph_hash][pre_graph_hash] = (node_id, xfer_id)

    def update_reward(self, pre_graph, node_id, xfer_id, graph):
        graph_hash = graph.hash()
        pre_graph_hash = pre_graph.hash()

        graph_gate_cnt = graph.gate_count
        pre_graph_gate_cnt = pre_graph.gate_count
        # TODO  if exists? min?
        old_graph_gate_cnt = self.gate_count_map.get(graph_hash, None)
        assert old_graph_gate_cnt is None or old_graph_gate_cnt == graph_gate_cnt
        self.gate_count_map[graph_hash] = graph_gate_cnt

        # update reward in this connected component by BFS
        # only when the gate count is reduced by this xfer (greedy?)
        if graph_gate_cnt < pre_graph_gate_cnt:
            reward = pre_graph_gate_cnt - graph_gate_cnt
            q = deque([])
            s = set([])
            q.append((pre_graph_hash, node_id, xfer_id, 0))
            s.add(pre_graph_hash)
            while len(q) != 0:
                g_hash, node_id, xfer_id, depth = q.popleft()
                if g_hash not in self.reward_map:
                    self.reward_map[g_hash] = {}
                    self.reward_map[g_hash][node_id] = {}
                    self.reward_map[g_hash][node_id][xfer_id] = reward * math.pow(
                        self.gamma, depth
                    )
                elif node_id not in self.reward_map[g_hash]:
                    self.reward_map[g_hash][node_id] = {}
                    self.reward_map[g_hash][node_id][xfer_id] = reward * math.pow(
                        self.gamma, depth
                    )
                elif xfer_id not in self.reward_map[g_hash][node_id]:
                    self.reward_map[g_hash][node_id][xfer_id] = reward * math.pow(
                        self.gamma, depth
                    )
                else:
                    self.reward_map[g_hash][node_id][xfer_id] = max(
                        reward * math.pow(self.gamma, depth),
                        self.reward_map[g_hash][node_id][xfer_id],
                    )

                # TODO  considering contiguous reduction
                # if reward of g_hash is not better, then stop find predecessors!
                # g_hash, node_id, xfer_id, depth, succ_gc, succ_reward = q.popleft()
                # reward = max(reward, this_gc - succ_gc + gamma * succ_reward)
                # assert( gamma * succ_reward == pow(init_reward, depth) )

                pre_dict = self.path_map[g_hash]
                for pre_hash in pre_dict:
                    assert self.gate_count_map[pre_hash] >= pre_graph_gate_cnt
                    if self.gate_count_map[pre_hash] > pre_graph_gate_cnt:
                        # TODO  global or local?
                        # print(f'!!! in update_reward: {self.gate_count_map[pre_hash]} > {pre_graph_gate_cnt}')
                        # assert(False)
                        continue
                    if pre_hash in s:
                        continue
                    else:
                        n_id, x_id = pre_dict[pre_hash]
                        q.append((pre_hash, n_id, x_id, depth + 1))
                        s.add(pre_hash)

    def save_data(self):
        with open('dataset/reward.json', 'w') as f:
            json.dump(self.reward_map, f)
        with open('dataset/path.json', 'w') as f:
            json.dump(self.path_map, f)


quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../bfs_verified_simplified.json',
    no_increase=True,
)
parser = quartz.PyQASMParser(context=quartz_context)
my_dag = parser.load_qasm(filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_graph = quartz.PyGraph(context=quartz_context, dag=my_dag)

candidate_hq = []
# min-heap by default; contains (gate_count, hash)
# gate_count is used to compare the pair by default
heapq.heappush(candidate_hq, (init_graph.gate_count, init_graph.hash()))
visited_hash_set = set()
visited_hash_set.add(init_graph.hash())
best_graph = init_graph
best_gate_cnt = init_graph.gate_count
max_gate_cnt = 64
budget = 5_000_000
buffer = DataBuffer(init_graph, 0.9)
save_graph(init_graph)  # TODO  check IO optimization
start = time.time()

while len(candidate_hq) > 0 and budget > 0:
    popped_cnt, first_hash = heapq.heappop(candidate_hq)
    first_graph = get_graph_from_hash(quartz_context, first_hash)
    all_nodes = first_graph.all_nodes()
    first_cnt = first_graph.gate_count
    # TODO  recompute? first_cnt, first_hash = heapq.heappop(candidate_hq)
    assert popped_cnt == first_cnt
    print(
        f'popped a graph with gate count: {first_cnt} , num of nodes: {len(all_nodes)}'
    )

    def available_xfers(i):
        return first_graph.available_xfers(context=quartz_context, node=all_nodes[i])

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = executor.map(
            available_xfers, list(range(len(all_nodes))), chunksize=2
        )
        appliable_xfers_nodes = list(results)

    for i in range(len(all_nodes)):
        node = all_nodes[i]
        appliable_xfers = appliable_xfers_nodes[i]
        for xfer in appliable_xfers:
            new_graph = first_graph.apply_xfer(
                xfer=quartz_context.get_xfer_from_id(id=xfer), node=node
            )
            new_hash = new_graph.hash()
            buffer.update_path(first_graph, i, xfer, new_graph)
            buffer.update_reward(first_graph, i, xfer, new_graph)
            if new_hash not in visited_hash_set:
                new_cnt = new_graph.gate_count
                visited_hash_set.add(new_hash)
                heapq.heappush(candidate_hq, (new_cnt, new_hash))
                save_graph(new_graph)  # TODO  check IO optimization
                if new_cnt < best_gate_cnt:
                    best_graph = new_graph
                    best_gate_cnt = new_cnt
            budget -= 1
            if budget % 10_000 == 0:
                print(
                    f'{budget}: minimum gate count is {best_gate_cnt}, after {time.time() - start:.2f} seconds'
                )
                buffer.save_data()
                print(buffer.reward_map)

buffer.save_data()
