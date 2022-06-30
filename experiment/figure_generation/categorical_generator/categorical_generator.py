import argparse
import pickle
import os
import time
import json

import multiprocess as mp
import quartz
from tqdm import tqdm


class DataBuffer:
    def __init__(self) -> None:
        self.hash2graphs = {}  # hash -> (qasm_str, gate_count)

    def add_graph(self, graph: quartz.PyGraph, graph_hash, graph_cnt):
        assert (graph_hash not in self.hash2graphs)
        self.hash2graphs[graph_hash] = (graph.to_qasm_str(), graph_cnt)

    def save_graph(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.hash2graphs, f, indent=2)


# these should be global variables to avoid serialization when multi-processing
# so we cannot run multiple generators concurrently,
# unless we create lists of these global variables
initial_graph = None
quartz_context = None
all_nodes = None


class Generator:
    def __init__(self, gate_set: list, ecc_file: str, input_data_file: str, gate_count: int):
        # output path
        self.output_path = f"./outputs_{gate_count}"
        self.gate_count = gate_count
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # set context
        global quartz_context
        quartz_context = quartz.QuartzContext(gate_set=gate_set, filename=ecc_file,
                                              no_increase=True, include_nop=False)

        self.input_data_file = input_data_file
        self.buffer = DataBuffer()

    def save(self, suffix: str = ''):
        self.buffer.save_graph(os.path.join(self.output_path, f'graph_{suffix}'))

    def gen(self, budget):
        # open the dataset and create a set of qasm files
        with open(self.input_data_file, 'r') as handle:
            circuit_dict = json.load(handle)
        selected_circuit_list = []
        for circuit_hash in circuit_dict:
            circuit_pack = circuit_dict[circuit_hash]
            if circuit_pack[1] == self.gate_count:
                selected_circuit_list.append(circuit_pack[0])
        print(f"There are {len(selected_circuit_list)} circuits to start with.")

        global initial_graph
        global quartz_context
        global all_nodes

        # set original graph
        qasm_parser = quartz.PyQASMParser(context=quartz_context)
        dag = qasm_parser.load_qasm_str(qasm_str=selected_circuit_list[0])
        initial_graph = quartz.PyGraph(context=quartz_context, dag=dag)
        initial_graph_hash = initial_graph.hash()
        initial_graph_gate_count = initial_graph.gate_count

        # (graph.gate_count, graph, graph_hash)
        candidate_hq = [[initial_graph_gate_count, initial_graph, initial_graph_hash]]
        visited_hash_set = set()
        visited_hash_set.add(initial_graph_hash)
        assert initial_graph_gate_count == self.gate_count
        self.buffer.add_graph(graph=initial_graph, graph_hash=initial_graph_hash,
                              graph_cnt=initial_graph_gate_count)

        # put other graphs into search queue
        for idx in range(1, len(selected_circuit_list)):
            cur_qasm = selected_circuit_list[idx]
            cur_dag = qasm_parser.load_qasm_str(qasm_str=cur_qasm)
            cur_graph = quartz.PyGraph(context=quartz_context, dag=cur_dag)
            cur_graph_hash = cur_graph.hash()
            cur_graph_gate_count = cur_graph.gate_count
            candidate_hq.append([cur_graph_gate_count, cur_graph, cur_graph_hash])

        # start search
        with tqdm(total=budget, desc='num of circuits gathered',
                  bar_format='{desc}: {n}/{total} |{bar}| {elapsed} {postfix}') as bar:
            total_visited_circuits = 0
            total_budget = budget
            while len(candidate_hq) > 0 and budget > 0:
                # get graph of current loop
                packet = candidate_hq.pop(0)
                initial_graph_gate_cnt, initial_graph, initial_graph_hash = packet[0], packet[1], packet[2]
                all_nodes = initial_graph.all_nodes()

                # get all applicable transfers
                def available_xfers(idx):
                    return initial_graph.available_xfers_parallel(context=quartz_context, node=all_nodes[idx])

                with mp.Pool() as pool:
                    applicable_xfers_nodes = pool.map(available_xfers, list(range(len(all_nodes))))

                # apply them to get new graphs
                # enumerate all nodes in the graph
                for i in range(len(all_nodes)):
                    node = all_nodes[i]
                    applicable_xfers = applicable_xfers_nodes[i]
                    # enumerate all xfers on selected node
                    for xfer in applicable_xfers:
                        new_graph = initial_graph.apply_xfer(xfer=quartz_context.get_xfer_from_id(id=xfer), node=node)
                        new_hash = new_graph.hash()
                        new_cnt = new_graph.gate_count
                        # only visit the new graph if not visited & with same gate count
                        if new_hash not in visited_hash_set and new_cnt == self.gate_count:
                            visited_hash_set.add(new_hash)
                            candidate_hq.append([new_cnt, new_graph, new_hash])
                            assert new_cnt == self.gate_count
                            self.buffer.add_graph(graph=new_graph, graph_hash=new_hash, graph_cnt=new_cnt)
                            bar.update(1)
                            budget -= 1
                        total_visited_circuits += 1

                        # logging
                        if budget % 100 == 0:
                            bar.set_postfix({'total visited': total_visited_circuits,
                                             'collected': len(self.buffer.hash2graphs)})
                            bar.refresh()
                        if budget % 1000 == 0:
                            self.save(f'{total_budget-budget}')
            self.save("final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figure for paper.')
    parser.add_argument('--gate_count', type=int, required=True)
    args = parser.parse_args()

    generator = Generator(
        gate_set=['h', 'cx', 't', 'tdg'],
        ecc_file="../bfs_verified_simplified.json",
        input_data_file="../data/900000_graph.json",
        gate_count=args.gate_count
    )
    generator.gen(150_000)
