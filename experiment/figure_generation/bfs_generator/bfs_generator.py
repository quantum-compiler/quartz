import argparse
import pickle
import os
import time

import multiprocess as mp
import quartz
from tqdm import tqdm


class GraphRecord:
    def __init__(self, qasm_str, gate_cnt, parent_hash):
        self.qasm_str = qasm_str
        self.gate_cnt = gate_cnt
        self.parent_hash = parent_hash
        self.neighbour_hash_list = []
        self.sealed = False

    def add_neighbor(self, neighbor_hash):
        self.neighbour_hash_list.append(neighbor_hash)

    def seal(self):
        self.sealed = True


class DataBuffer:
    def __init__(self) -> None:
        self.hash2graphs = {}  # hash -> GraphRecord
        self.total_graphs = 0

    def add_graph(self, graph: quartz.PyGraph, graph_hash, graph_cnt, parent_hash):
        assert (graph_hash not in self.hash2graphs)
        self.hash2graphs[graph_hash] = GraphRecord(qasm_str=graph.to_qasm_str(),
                                                   gate_cnt=graph_cnt, parent_hash=parent_hash)
        self.total_graphs += 1

    def add_neighbor(self, graph_hash, neighbor_hash):
        self.hash2graphs[graph_hash].add_neighbor(neighbor_hash=neighbor_hash)

    def save_graph(self, output_path: str):
        # separate the graphs
        finished_graphs = {}
        unfinished_graphs = {}
        for graph_hash in self.hash2graphs:
            graph_record = self.hash2graphs[graph_hash]
            if graph_record.sealed:
                finished_graphs[graph_hash] = graph_record
            else:
                unfinished_graphs[graph_hash] = graph_record

        # save into file
        if not len(finished_graphs) == 0:
            with open(output_path + f"_{len(finished_graphs)}_finished.dat", 'wb') as f:
                pickle.dump(obj=finished_graphs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(output_path + f"_{len(unfinished_graphs)}_unfinished.dat", 'wb') as f:
                pickle.dump(obj=unfinished_graphs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        self.hash2graphs = unfinished_graphs


# these should be global variables to avoid serialization when multi-processing
# so we cannot run multiple generators concurrently,
# unless we create lists of these global variables
initial_graph = None
quartz_context = None
all_nodes = None


class Generator:
    def __init__(self, gate_set: list, ecc_file: str, input_circuit_file: str,
                 no_increase: bool = True, include_nop: bool = True):
        # output path
        self.output_path = "./outputs"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # set context
        global quartz_context
        quartz_context = quartz.QuartzContext(gate_set=gate_set, filename=ecc_file,
                                              no_increase=no_increase, include_nop=include_nop)

        # set original graph
        global initial_graph
        qasm_parser = quartz.PyQASMParser(context=quartz_context)
        dag = qasm_parser.load_qasm(filename=input_circuit_file)
        initial_graph = quartz.PyGraph(context=quartz_context, dag=dag)

        self.buffer = DataBuffer()

    def save(self, suffix: str = ''):
        self.buffer.save_graph(os.path.join(self.output_path, f'graph_{suffix}'))

    def gen(self, budget):
        global initial_graph
        global quartz_context
        global all_nodes

        initial_graph_hash = initial_graph.hash()
        initial_graph_gate_count = initial_graph.gate_count
        best_gate_cnt = initial_graph_gate_count

        # min-heap by default; contains (gate_count, hash)
        # gate_count is used to compare the pair by default
        candidate_hq = []  # (graph.gate_count, graph, graph_hash)
        visited_hash_set = set()
        candidate_hq.append([initial_graph_gate_count, initial_graph, initial_graph_hash])
        visited_hash_set.add(initial_graph_hash)
        self.buffer.add_graph(graph=initial_graph, graph_hash=initial_graph_hash,
                              graph_cnt=initial_graph_gate_count, parent_hash=0)

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
                        # only visit the new graph if not visited
                        if new_hash not in visited_hash_set:
                            visited_hash_set.add(new_hash)
                            candidate_hq.append([new_cnt, new_graph, new_hash])
                            self.buffer.add_graph(graph=new_graph, graph_hash=new_hash, graph_cnt=new_cnt,
                                                  parent_hash=initial_graph_hash)
                            self.buffer.add_neighbor(graph_hash=initial_graph_hash, neighbor_hash=new_hash)
                            if new_cnt < best_gate_cnt:
                                print(f"Graph with {new_cnt} gates found!")
                                best_gate_cnt = new_cnt
                            bar.update(1)
                            budget -= 1
                        else:
                            self.buffer.add_neighbor(graph_hash=initial_graph_hash, neighbor_hash=new_hash)
                        total_visited_circuits += 1

                        # logging
                        if budget % 1000 == 0:
                            bar.set_postfix({'best_cnt': best_gate_cnt, 'visited': total_visited_circuits,
                                             'graphs': self.buffer.total_graphs})
                            bar.refresh()
                        if budget % 10_000 == 0:
                            self.save(f'{total_budget-budget}')
                self.buffer.hash2graphs[initial_graph_hash].seal()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figure for paper.')
    parser.add_argument('--no_increase', type=bool, default=True, required=False)
    parser.add_argument('--include_nop', type=bool, default=False, required=False)
    args = parser.parse_args()

    generator = Generator(
        gate_set=['h', 'cx', 't', 'tdg'],
        ecc_file="../bfs_verified_simplified.json",
        input_circuit_file="../barenco_tof_3.qasm",
        no_increase=args.no_increase,
        include_nop=args.include_nop,
    )
    generator.gen(5_000_000)
