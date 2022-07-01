"""
This is second step of figure1. It generates a graph with node and edges.
"""
import argparse
import os
import json
import pickle

import quartz


def main():
    # get neighbor set
    with open(f"./neighbor_set_13754371184132648763.json", 'r') as handle:
        multi_hop_neighbor_set = json.load(handle)
        initial_graph_hash = 13754371184132648763
    print(len(multi_hop_neighbor_set))

    # reconstruct map
    hash2graph_map = {}  # hash -> [idx, graph, gate_count]
    edge_map = {}  # idx -> [idx list] (edge set)
    # prepare quartz context
    quartz_context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg'],
                                          filename="../bfs_verified_simplified.json",
                                          no_increase=False, include_nop=False)
    qasm_parser = quartz.PyQASMParser(context=quartz_context)
    graph_idx = 0
    original_graph_idx = -1
    for graph_hash in multi_hop_neighbor_set:
        if graph_hash == f"{initial_graph_hash}":
            original_graph_idx = graph_idx
        graph_qasm = multi_hop_neighbor_set[graph_hash][0]
        gate_count = multi_hop_neighbor_set[graph_hash][1]
        dag = qasm_parser.load_qasm_str(qasm_str=graph_qasm)
        cur_graph = quartz.PyGraph(context=quartz_context, dag=dag)
        cur_graph_hash = cur_graph.hash()  # graph hash needs to be recomputed for each program
        cur_graph_gate_count = cur_graph.gate_count
        assert cur_graph_gate_count == gate_count
        # append
        hash2graph_map[cur_graph_hash] = [graph_idx, cur_graph, cur_graph_gate_count]
        edge_map[graph_idx] = []
        graph_idx += 1
        if graph_idx % 100 == 0:
            print(f"Finished preprocessing of {graph_idx} circuits.")
    assert not original_graph_idx == -1
    print(f"Preprocessing has finished")

    # check connectivity
    finished = 0
    for graph_hash in hash2graph_map:
        # get graph for current loop
        graph_packet = hash2graph_map[graph_hash]
        cur_graph_idx = graph_packet[0]
        cur_graph = graph_packet[1]
        all_nodes = cur_graph.all_nodes()
        finished += 1
        if finished % 10 == 0:
            print(f"Finished {finished} circuits.")

        # apply xfers to get to new graphs
        for i in range(len(all_nodes)):
            node = all_nodes[i]
            applicable_xfers = cur_graph.available_xfers_parallel(context=quartz_context,
                                                                  node=node)
            for xfer in applicable_xfers:
                new_graph = cur_graph.apply_xfer(xfer=quartz_context.get_xfer_from_id(id=xfer), node=node)
                new_hash = new_graph.hash()
                # check this circuit
                if new_hash in hash2graph_map:
                    new_graph_idx = hash2graph_map[new_hash][0]
                    edge_map[cur_graph_idx].append(new_graph_idx)

    # output
    output_file_name = f"./connectivity_graph_{original_graph_idx}.pkl"
    with open(output_file_name, 'wb') as handle:
        pickle.dump(edge_map, handle)


if __name__ == '__main__':
    main()
