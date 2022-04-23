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
    qc_origin = QuantumCircuit.from_qasm_file(
        'barenco_tof_3_opt_path/subst_history_39.qasm')
    qc_optimized = QuantumCircuit.from_qasm_file('check.qasm')
    return Statevector.from_instruction(qc_origin).equiv(
        Statevector.from_instruction(qc_optimized))


def get_graph_from_hash(context, h):
    s = str(h)
    parser = quartz.PyQASMParser(context=context)
    my_dag = parser.load_qasm(filename=f"dataset/{s}.qasm")
    return quartz.PyGraph(context=context, dag=my_dag)


def save_graph(graph):
    h = graph.hash()
    graph.to_qasm(filename=f'dataset/{str(h)}.qasm')


class DataBuffer:
    def __init__(self, init_graph, gamma) -> None:
        init_hash = init_graph.hash()
        init_value = init_graph.gate_count

        self.reward_map = {}
        self.path_map = {}
        self.gate_count_map = {}

        self.gate_count_map[init_hash] = init_value
        self.path_map[init_hash] = {}

        self.origin_value = init_value
        self.gamma = gamma

    def update_path(self, pre_graph, node_id, xfer_id, graph):
        graph_hash = graph.hash()
        pre_graph_hash = pre_graph.hash()

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

        self.gate_count_map[graph_hash] = graph_gate_cnt

        if graph_gate_cnt < pre_graph_gate_cnt:
            # print("reduced")
            reward = pre_graph_gate_cnt - graph_gate_cnt
            q = deque([])
            s = set([])
            q.append((pre_graph_hash, node_id, xfer_id, 0))
            s.add(pre_graph_hash)
            while len(q) != 0:
                # print("in queue")
                g_hash, node_id, xfer_id, depth = q.popleft()
                # if (g_hash, node_id, xfer_id) not in self.reward_map:
                #     self.reward_map[(
                #         g_hash, node_id,
                #         xfer_id)] = reward * math.pow(self.gamma, depth)
                # else:
                #     self.reward_map[(g_hash, node_id, xfer_id)] = max(
                #         reward * math.pow(0.9, depth),
                #         self.reward_map[(g_hash, node_id, xfer_id)])
                if g_hash not in self.reward_map:
                    self.reward_map[g_hash] = {}
                    self.reward_map[g_hash][node_id] = {}
                    self.reward_map[g_hash][node_id][
                        xfer_id] = reward * math.pow(self.gamma, depth)
                elif node_id not in self.reward_map[g_hash]:
                    self.reward_map[g_hash][node_id] = {}
                    self.reward_map[g_hash][node_id][
                        xfer_id] = reward * math.pow(self.gamma, depth)
                elif xfer_id not in self.reward_map[g_hash][node_id]:
                    self.reward_map[g_hash][node_id][
                        xfer_id] = reward * math.pow(self.gamma, depth)
                else:
                    self.reward_map[g_hash][node_id][xfer_id] = max(
                        reward * math.pow(0.9, depth),
                        self.reward_map[g_hash][node_id][xfer_id])

                pre_dict = self.path_map[g_hash]
                for pre_hash in pre_dict:
                    if self.gate_count_map[pre_hash] > pre_graph_gate_cnt:
                        continue
                    if pre_hash in s:
                        continue
                    else:
                        n_id, x_id = pre_dict[pre_hash]
                        q.append((pre_hash, n_id, x_id, depth + 1))

    # def update(self, pre_graph, node_id, xfer_id, graph):
    #     graph_hash = graph.hash()
    #     pre_graph_hash = pre_graph.hash()

    #     if graph_hash not in self.path_map:
    #         self.path_map[graph_hash] = {pre_graph_hash: (node_id, xfer_id)}
    #     else:
    #         if pre_graph_hash not in self.path_map[graph_hash]:
    #             self.path_map[graph_hash][pre_graph_hash] = (node_id, xfer_id)

    #     graph_gate_cnt = graph.gate_count
    #     pre_graph_gate_cnt = pre_graph.gate_count

    #     self.gate_count_map[graph_hash] = graph_gate_cnt

    #     if graph_gate_cnt < pre_graph_gate_cnt:
    #         # print("reduced")
    #         reward = pre_graph_gate_cnt - graph_gate_cnt
    #         q = deque([])
    #         q.append((pre_graph_hash, node_id, xfer_id, 0))
    #         while len(q) != 0:
    #             # print("in queue")
    #             g_hash, node_id, xfer_id, depth = q.popleft()
    #             # if (g_hash, node_id, xfer_id) not in self.reward_map:
    #             #     self.reward_map[(
    #             #         g_hash, node_id,
    #             #         xfer_id)] = reward * math.pow(self.gamma, depth)
    #             # else:
    #             #     self.reward_map[(g_hash, node_id, xfer_id)] = max(
    #             #         reward * math.pow(0.9, depth),
    #             #         self.reward_map[(g_hash, node_id, xfer_id)])
    #             if g_hash not in self.reward_map:
    #                 self.reward_map[g_hash] = {}
    #                 self.reward_map[g_hash][node_id] = {}
    #                 self.reward_map[g_hash][node_id][
    #                     xfer_id] = reward * math.pow(self.gamma, depth)
    #             elif node_id not in self.reward_map[g_hash]:
    #                 self.reward_map[g_hash][node_id] = {}
    #                 self.reward_map[g_hash][node_id][
    #                     xfer_id] = reward * math.pow(self.gamma, depth)
    #             elif xfer_id not in self.reward_map[g_hash][node_id]:
    #                 self.reward_map[g_hash][node_id][
    #                     xfer_id] = reward * math.pow(self.gamma, depth)
    #             else:
    #                 self.reward_map[g_hash][node_id][xfer_id] = max(
    #                     reward * math.pow(0.9, depth),
    #                     self.reward_map[g_hash][node_id][xfer_id])

    #             pre_dict = self.path_map[g_hash]
    #             for pre_hash in pre_dict:
    #                 if self.gate_count_map[pre_hash] > pre_graph_gate_cnt:
    #                     continue
    #                 else:
    #                     n_id, x_id = pre_dict[pre_hash]
    #                     q.append((pre_hash, n_id, x_id, depth + 1))
    #             # print(q)

    def save_data(self):
        with open('dataset/reward.json', 'w') as f:
            json.dump(self.reward_map, f)
        with open('dataset/path.json', 'w') as f:
            json.dump(self.path_map, f)


quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../bfs_verified_simplified.json',
    no_increase=True)
parser = quartz.PyQASMParser(context=quartz_context)
my_dag = parser.load_qasm(
    filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_graph = quartz.PyGraph(context=quartz_context, dag=my_dag)

candidate_hq = []
heapq.heappush(candidate_hq, (init_graph.gate_count, init_graph.hash()))
hash_set = set()
hash_set.add(init_graph.hash())
best_graph = init_graph
best_gate_cnt = init_graph.gate_count
max_gate_cnt = 64
budget = 5_000_000
buffer = DataBuffer(init_graph, 0.9)
save_graph(init_graph)

start = time.time()

while candidate_hq != [] and budget >= 0:
    _, first_hash = heapq.heappop(candidate_hq)
    first_graph = get_graph_from_hash(quartz_context, first_hash)
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
            buffer.update_path(first_graph, i, xfer, new_graph)
            buffer.update_reward(first_graph, i, xfer, new_graph)
            if new_hash not in hash_set:
                new_cnt = new_graph.gate_count
                hash_set.add(new_hash)
                heapq.heappush(candidate_hq, (new_cnt, new_hash))
                save_graph(new_graph)
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