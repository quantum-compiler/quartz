import quartz
import time
import os
import heapq
import json
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import math
from collections import deque
from functools import partial
import multiprocess as mp
from tqdm import tqdm

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
    dir_path = f'dataset'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    graph.to_qasm(filename=f'dataset/{str(h)}.qasm')


class DataBuffer:
    def __init__(self, init_graph, gamma) -> None:
        self.hash2graphs = {}
        self.reward_map = {}
        self.path_map = {}
        self.gate_count_map = {}
        self.gamma = gamma
        
        init_hash = init_graph.hash()
        init_value = init_graph.gate_count
        self.gate_count_map[init_hash] = init_value
        self.path_map[init_hash] = {} # stores predecessors
        self.origin_value = init_value
        
    def add_graph(self, graph: quartz.PyGraph, hash_v=None):
        if hash_v is None:
            hash_v = graph.hash()
        self.hash2graphs[hash_v] = graph

    def update_path(self, pre_graph, node_id, xfer_id, graph):
        # may introduce loops; it's the caller's responsibility to handle it
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
        # TODO  if exists? min?
        old_graph_gate_cnt = self.gate_count_map.get(graph_hash, None)
        assert(old_graph_gate_cnt is None or old_graph_gate_cnt == graph_gate_cnt)
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
                        reward * math.pow(self.gamma, depth),
                        self.reward_map[g_hash][node_id][xfer_id])
                
                # TODO  considering contiguous reduction
                # if reward of g_hash is not better, then stop find predecessors!
                # g_hash, node_id, xfer_id, depth, succ_gc, succ_reward = q.popleft()
                # reward = max(reward, this_gc - succ_gc + gamma * succ_reward)
                # assert( gamma * succ_reward == pow(init_reward, depth) )

                pre_dict = self.path_map[g_hash]
                for pre_hash in pre_dict:
                    assert(self.gate_count_map[pre_hash] >= pre_graph_gate_cnt)
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
            json.dump(self.reward_map, f, indent=2)
        with open('dataset/path.json', 'w') as f:
            json.dump(self.path_map, f, indent=2)

# these should be global variables to avoid serialization when multi-processing
# so we cannot run multiple generators concurrently, 
# unless we create lists of these global variables
first_graph = None
quartz_context = None
all_nodes = None

class Generator:
    def __init__(self,
        gate_set: list,
        ecc_file: str,
        input_circuit_file: str,
        gamma: int = 0.9,
        no_increase: bool = True,
        output_path: str = 'pretrain_dataset',
    ):
        self.output_path = output_path

        global quartz_context
        global first_graph
        quartz_context = quartz.QuartzContext(
            gate_set=gate_set,
            filename=ecc_file,
            no_increase=no_increase,
        )
        parser = quartz.PyQASMParser(context=quartz_context)
        dag = parser.load_qasm(filename=input_circuit_file)
        first_graph = quartz.PyGraph(context=quartz_context, dag=dag)
        
        self.buffer = DataBuffer(first_graph, gamma)

    def gen(self):
        global first_graph
        global quartz_context
        global all_nodes

        candidate_hq = []
        # min-heap by default; contains (gate_count, hash)
        # gate_count is used to compare the pair by default
        heapq.heappush(candidate_hq, (first_graph.gate_count, first_graph))
        visited_hash_set = set()
        visited_hash_set.add(first_graph.hash())
        best_graph = first_graph
        best_gate_cnt = first_graph.gate_count
        max_gate_cnt = 64
        budget = 5_000_000
        
        # save_graph(init_graph) # TODO  check IO optimization
        with tqdm(
            total=first_graph.gate_count,
            desc='num of gates reduced',
            bar_format='{desc}: {n}/{total} |{bar}| {elapsed} {postfix}',
        ) as pbar:
            while len(candidate_hq) > 0 and budget > 0:
                first_cnt, first_graph = heapq.heappop(candidate_hq)
                all_nodes = first_graph.all_nodes()
                # print(f'popped a graph with gate count: {first_cnt} , num of nodes: {len(all_nodes)}')
                
                def available_xfers(i):
                    return first_graph.available_xfers(context=quartz_context, node=all_nodes[i])

                # av_start = time.monotonic_ns()
                with mp.Pool() as pool:
                    appliable_xfers_nodes = pool.map(available_xfers, list(range(len(all_nodes))))
                # av_end = time.monotonic_ns()
                # print(f'av duration: { (av_end - av_start) / 1e6 } ms')

                for i in range(len(all_nodes)):
                    node = all_nodes[i]
                    appliable_xfers = appliable_xfers_nodes[i]
                    for xfer in appliable_xfers:
                        new_graph = first_graph.apply_xfer(
                            xfer=quartz_context.get_xfer_from_id(id=xfer), node=node)
                        new_hash = new_graph.hash()
                        self.buffer.update_path(first_graph, i, xfer, new_graph)
                        self.buffer.update_reward(first_graph, i, xfer, new_graph)
                        if new_hash not in visited_hash_set:
                            new_cnt = new_graph.gate_count
                            visited_hash_set.add(new_hash)
                            heapq.heappush(candidate_hq, (new_cnt, new_graph))
                            # save_graph(new_graph) # TODO  check IO optimization
                            if new_cnt < best_gate_cnt:
                                print()
                                pbar.update(best_gate_cnt - new_cnt)
                                best_graph = new_graph
                                best_gate_cnt = new_cnt
                                pbar.set_postfix({ 'best_gate_cnt': best_gate_cnt })
                                
                        budget -= 1
                        if budget % 1000 == 0:
                            pbar.refresh()
                        if budget % 10_000 == 0:
                            self.buffer.save_data()
                            print(self.buffer.reward_map)
                # end for all_nodes
            # end while
        # end with

        self.buffer.save_data()

if __name__ == '__main__':
    generator = Generator(
        gate_set=['h', 'cx', 't', 'tdg'],
        ecc_file='../bfs_verified_simplified.json',
        input_circuit_file="barenco_tof_3_opt_path/subst_history_39.qasm",
        gamma=0.9,
        no_increase=True,
        output_path='pretrain_dataset',
    )
    generator.gen()