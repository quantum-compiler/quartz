import argparse
import os

import quartz


class Query:
    def __init__(self):
        self.parent_hash = None
        self.cur_hash = None
        self.graph = None


def gen_path(rank, qasm_str, max_depth):
    """
    Find the minimal number of steps needed to achieve a gate reduction from given circuit.
    Implemented by BFS.
    return: -1 if not found under max_depth, min #xfer o.w.
    """
    # prepare quartz context
    quartz_context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg'],
                                          filename="../bfs_verified_simplified.json",
                                          no_increase=True, include_nop=False)
    qasm_parser = quartz.PyQASMParser(context=quartz_context)
    dag = qasm_parser.load_qasm_str(qasm_str=qasm_str)
    initial_graph = quartz.PyGraph(context=quartz_context, dag=dag)
    initial_graph_hash = initial_graph.hash()
    initial_graph_gate_count = initial_graph.gate_count
    print(f"[Rank {rank}] New circuit {initial_graph_hash} with {initial_graph_gate_count} gates.")

    # start BFS search for circuits with fewer gates
    candidate_queue = [[initial_graph, initial_graph_hash]]
    visited_hash_set = {initial_graph_hash: 0}
    # hash -> [graph, parent hash, node idx, xfer idx]
    path_set = {initial_graph_hash: [initial_graph, None, None, None]}
    distribution_list = []
    for i in range(max_depth + 1):
        distribution_list.append(0)
    searched_count = 0
    while len(candidate_queue) > 0:
        # get graph for current loop
        graph_packet = candidate_queue.pop(0)
        cur_graph = graph_packet[0]
        cur_graph_hash = graph_packet[1]
        all_nodes = cur_graph.all_nodes()

        # apply xfers to get to new graphs
        for i in range(len(all_nodes)):
            node = all_nodes[i]
            applicable_xfers = cur_graph.available_xfers_parallel(context=quartz_context,
                                                                  node=node)
            for xfer in applicable_xfers:
                new_graph = cur_graph.apply_xfer(xfer=quartz_context.get_xfer_from_id(id=xfer), node=node)
                new_hash = new_graph.hash()
                new_cnt = new_graph.gate_count
                # otherwise we will continue search
                if new_hash not in visited_hash_set:
                    visited_hash_set[new_hash] = visited_hash_set[cur_graph_hash] + 1
                    if visited_hash_set[new_hash] > max_depth:
                        return distribution_list
                    distribution_list[visited_hash_set[new_hash]] += 1
                    path_set[new_hash] = [new_graph, cur_graph_hash, i, xfer]
                    candidate_queue.append([new_graph, new_hash])

                # this is for logging
                searched_count += 1
                if searched_count % 100 == 0:
                    print(f"[Rank {rank}] Searched {searched_count} circuits,"
                          f" now distribution is {distribution_list}.")
    assert False


def main():
    # read in the circuit
    with open(f"../barenco_tof_3.qasm", 'r') as handle:
        qasm_str = handle.read()
        final_circuit_count_list = gen_path(rank=0, qasm_str=qasm_str, max_depth=8)
        print(f"Final circuit count list is {final_circuit_count_list}.")


if __name__ == '__main__':
    main()
