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
                                          filename="../../bfs_verified_simplified.json",
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
                # if new circuit has fewer gates, then we are done
                if new_cnt < initial_graph_gate_count:
                    # save the path
                    path_set[new_hash] = [new_graph, cur_graph_hash, i, xfer]
                    circuit_idx = visited_hash_set[cur_graph_hash] + 1
                    save_hash = new_hash
                    while True:
                        # unpack
                        save_graph = path_set[save_hash][0]
                        par_hash = path_set[save_hash][1]
                        save_node_idx = path_set[save_hash][2]
                        save_xfer_idx = path_set[save_hash][3]
                        # save circuit
                        save_path_name = f"./path/step{circuit_idx}_{save_node_idx}_{save_xfer_idx}.qasm"
                        print(save_path_name)
                        with open(save_path_name, 'w') as handle:
                            handle.write(save_graph.to_qasm_str())
                        # change param
                        if par_hash is None and save_node_idx is None and save_xfer_idx is None:
                            break
                        circuit_idx -= 1
                        save_hash = par_hash
                    # return final depth
                    final_depth = visited_hash_set[cur_graph_hash] + 1
                    return final_depth
                # otherwise we will continue search
                if new_hash not in visited_hash_set:
                    visited_hash_set[new_hash] = visited_hash_set[cur_graph_hash] + 1
                    path_set[new_hash] = [new_graph, cur_graph_hash, i, xfer]
                    candidate_queue.append([new_graph, new_hash])
                # and stop when we reach max depth
                if visited_hash_set[new_hash] > max_depth:
                    assert False
                # this is for logging
                searched_count += 1
                if searched_count % 1000 == 0:
                    print(f"[Rank {rank}] Searched {searched_count} circuits,"
                          f" now at depth {visited_hash_set[new_hash]}.")
    assert False


def main():
    output_path = f"./path"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # read in the circuit
    with open(f"./candidate_5steps_t_tdg.qasm", 'r') as handle:
        qasm_str = handle.read()
        final_depth = gen_path(rank=0, qasm_str=qasm_str, max_depth=5)
        print(f"Final depth is {final_depth}.")


if __name__ == '__main__':
    main()
