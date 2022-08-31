import argparse
import heapq
import json
import math
import os
import time
from collections import deque
from functools import partial

import multiprocess as mp
from IPython import embed
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from tqdm import tqdm

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


class DataBuffer:
    def __init__(self, gamma) -> None:
        self.hash2graphs = {}  # graph qasm and gate count
        self.reward_map = {}
        self.path_map = {}  # predecessors
        self.gamma = gamma

    def add_graph(self, graph: quartz.PyGraph, graph_hash, graph_cnt):
        assert graph_hash not in self.hash2graphs  # TODO  remove
        self.hash2graphs[graph_hash] = (graph.to_qasm_str(), graph_cnt)

    def update_path(self, pre_graph_hash, node_id, xfer_id, graph_hash):
        # may introduce loops; it's the caller's responsibility to handle it
        if graph_hash not in self.path_map:
            self.path_map[graph_hash] = {pre_graph_hash: (node_id, xfer_id)}
        elif pre_graph_hash not in self.path_map[graph_hash]:
            self.path_map[graph_hash][pre_graph_hash] = (node_id, xfer_id)

    def _add_to_reward_map(self, reward, g_hash, node_id, xfer_id, depth):
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

    def update_reward(
        self, pre_graph_hash, pre_graph_cnt, node_id, xfer_id, graph_hash, graph_cnt
    ):
        # pre_graph -> graph
        # TODO  if exists? min?
        assert pre_graph_hash in self.hash2graphs and graph_hash in self.hash2graphs
        # old_graph_gate_cnt = self.gate_count_map.get(graph_hash, None)
        # assert(old_graph_gate_cnt is None or old_graph_gate_cnt == graph_gate_cnt)
        # self.gate_count_map[graph_hash] = graph_gate_cnt

        # update reward in this connected component by BFS
        # only when the gate count is reduced by this xfer (greedy?)
        reward = pre_graph_cnt - graph_cnt
        if reward == 0:
            self._add_to_reward_map(reward, pre_graph_hash, node_id, xfer_id, 0)

        if reward > 0:
            q = deque([])  # (g_hash, node_id, xfer_id, depth)
            s = set([])
            q.append((pre_graph_hash, node_id, xfer_id, 0))
            s.add(pre_graph_hash)
            while len(q) > 0:
                g_hash, node_id, xfer_id, depth = q.popleft()
                self._add_to_reward_map(reward, g_hash, node_id, xfer_id, depth)

                # TODO  considering contiguous reduction
                # if reward of g_hash is not better, then stop find predecessors!
                # g_hash, node_id, xfer_id, depth, succ_gc, succ_reward = q.popleft()
                # reward = max(reward, this_gc - succ_gc + gamma * succ_reward)
                # assert( gamma * succ_reward == pow(init_reward, depth) )

                pre_dict = self.path_map[g_hash]
                for pre_hash in pre_dict:
                    assert self.hash2graphs[pre_hash][1] >= pre_graph_cnt
                    if (
                        pre_hash not in s
                        and self.hash2graphs[pre_hash][1] <= pre_graph_cnt
                    ):
                        # TODO  global or local?
                        # print(f'!!! in update_reward: {self.gate_count_map[pre_hash]} > {pre_graph_gate_cnt}')
                        # assert(False)
                        n_id, x_id = pre_dict[pre_hash]
                        q.append((pre_hash, n_id, x_id, depth + 1))
                        s.add(pre_hash)

    def save_path(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.path_map, f, indent=2)

    def save_reward(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.reward_map, f, indent=2)

    def save_graph(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.hash2graphs, f, indent=2)


# these should be global variables to avoid serialization when multi-processing
# so we cannot run multiple generators concurrently,
# unless we create lists of these global variables
first_graph = None
quartz_context = None
all_nodes = None


class Generator:
    def __init__(
        self,
        gate_set: list,
        ecc_file: str,
        input_circuit_file: str,
        gamma: int = 0.9,
        no_increase: bool = True,
        output_path: str = 'pretrain_dataset',
    ):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

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

        self.buffer = DataBuffer(gamma)

    def save(self, prefix: str = ''):
        self.buffer.save_path(os.path.join(self.output_path, f'{prefix}path.json'))
        self.buffer.save_graph(os.path.join(self.output_path, f'{prefix}graph.json'))
        self.buffer.save_reward(os.path.join(self.output_path, f'{prefix}reward.json'))

    def gen(self):
        global first_graph
        global quartz_context
        global all_nodes

        first_graph_hash = first_graph.hash()
        first_graph_gate_count = first_graph.gate_count

        # min-heap by default; contains (gate_count, hash)
        # gate_count is used to compare the pair by default
        candidate_hq = []  # (graph.gate_count, graph, graph_hash)
        visited_hash_set = set()
        heapq.heappush(
            candidate_hq, (first_graph_gate_count, first_graph, first_graph_hash)
        )
        visited_hash_set.add(first_graph_hash)
        self.buffer.add_graph(first_graph, first_graph_hash, first_graph_gate_count)

        best_graph = first_graph
        best_gate_cnt = first_graph_gate_count
        max_gate_cnt = 64
        budget = 5_000_000

        with tqdm(
            total=first_graph.gate_count,
            desc='num of gates reduced',
            bar_format='{desc}: {n}/{total} |{bar}| {elapsed} {postfix}',
        ) as pbar:
            while len(candidate_hq) > 0 and budget > 0:
                first_cnt, first_graph, first_graph_hash = heapq.heappop(candidate_hq)
                all_nodes = first_graph.all_nodes()
                # print(f'popped a graph with gate count: {first_cnt} , num of nodes: {len(all_nodes)}')

                def available_xfers(i):
                    return first_graph.available_xfers(
                        context=quartz_context, node=all_nodes[i]
                    )

                # av_start = time.monotonic_ns()
                with mp.Pool() as pool:
                    appliable_xfers_nodes = pool.map(
                        available_xfers, list(range(len(all_nodes)))
                    )
                # av_end = time.monotonic_ns()
                # print(f'av duration: { (av_end - av_start) / 1e6 } ms')
                # print(sum([len(xfers) for xfers in appliable_xfers_nodes]))
                for i in range(len(all_nodes)):
                    node = all_nodes[i]
                    appliable_xfers = appliable_xfers_nodes[i]
                    for xfer in appliable_xfers:
                        new_graph = first_graph.apply_xfer(
                            xfer=quartz_context.get_xfer_from_id(id=xfer), node=node
                        )
                        new_hash = new_graph.hash()
                        new_cnt = new_graph.gate_count

                        if new_hash not in visited_hash_set:
                            visited_hash_set.add(new_hash)
                            heapq.heappush(candidate_hq, (new_cnt, new_graph, new_hash))
                            self.buffer.add_graph(new_graph, new_hash, new_cnt)
                            if new_cnt < best_gate_cnt:
                                print()
                                pbar.update(best_gate_cnt - new_cnt)
                                best_graph = new_graph
                                best_gate_cnt = new_cnt

                        self.buffer.update_path(first_graph_hash, i, xfer, new_hash)
                        self.buffer.update_reward(
                            first_graph_hash, first_cnt, i, xfer, new_hash, new_cnt
                        )

                        budget -= 1
                        if budget % 1000 == 0:
                            pbar.set_postfix(
                                {
                                    'best_gate_cnt': best_gate_cnt,
                                    '|reward_map|': len(self.buffer.reward_map),
                                    'budget': budget,
                                    '|graphs|': len(self.buffer.hash2graphs),
                                }
                            )
                            pbar.refresh()
                        if budget % 100_000 == 0:
                            self.save(f'{budget}_')
                # end for all_nodes
            # end while
        # end with

        self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset for pre-training.')
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    generator = Generator(
        gate_set=['h', 'cx', 't', 'tdg'],
        # ecc_file='../bfs_verified_simplified.json',
        ecc_file='../3_2_5_complete_ECC_set.json',
        input_circuit_file="barenco_tof_3_opt_path/subst_history_39.qasm",
        gamma=0.9,
        no_increase=True,
        output_path=args.output_path,
    )
    try:
        generator.gen()
    except KeyboardInterrupt:
        generator.save()
        embed()
