import argparse
import json
import multiprocessing as mp
import pickle
import random
import time
import os

import quartz


class Query:
    def __init__(self):
        self.parent_hash = None
        self.cur_hash = None
        self.graph = None


def analyze_circuit(rank, qasm_str, max_depth):
    """
    Find the minimal number of steps needed to achieve a gate reduction from given circuit.
    Implemented by BFS.
    return: (-1, 0) if not found under max_depth, (min #xfer, reduced gate count) o.w.
    """
    # prepare quartz context
    quartz_context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg'],
                                          filename="../bfs_verified_simplified.json",
                                          no_increase=False, include_nop=False)
    qasm_parser = quartz.PyQASMParser(context=quartz_context)
    dag = qasm_parser.load_qasm_str(qasm_str=qasm_str)
    initial_graph = quartz.PyGraph(context=quartz_context, dag=dag)
    initial_graph_hash = initial_graph.hash()
    initial_graph_gate_count = initial_graph.gate_count
    print(f"[Rank {rank}] New circuit {initial_graph_hash} with {initial_graph_gate_count} gates.")

    # start BFS search for circuits with fewer gates
    candidate_queue = [[initial_graph, initial_graph_hash]]
    visited_hash_set = {initial_graph_hash: 0}
    searched_count = 0
    depth = 0
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
                    final_depth = visited_hash_set[cur_graph_hash] + 1
                    return final_depth, initial_graph_gate_count - new_cnt
                # otherwise we will continue search
                if new_hash not in visited_hash_set and new_cnt <= 58:
                    visited_hash_set[new_hash] = visited_hash_set[cur_graph_hash] + 1
                    depth = max(depth, visited_hash_set[new_hash])
                    candidate_queue.append([new_graph, new_hash])
                # and stop when we reach max depth
                if new_cnt <= 58 and visited_hash_set[new_hash] > max_depth:
                    return -1, 0
                # this is for logging
                searched_count += 1
                if searched_count % 100000 == 0:
                    print(f"[Rank {rank}] Searched {searched_count} circuits,"
                          f" now at depth {depth}.")
    assert False


def worker_proc(rank, circuit_batch, max_depth, gate_count_to_plot):
    result_list = []
    reduced_cnt_list = []
    finished_count = 0
    total_circuits = len(circuit_batch)
    for circuit_packet in circuit_batch:
        qasm_str = circuit_packet[0]
        gate_count = circuit_packet[1]
        print(f"[Rank {rank}] ({finished_count + 1}/{total_circuits})"
              f" Start analyzing new circuit with {gate_count} gates.")
        depth, reduced_cnt = analyze_circuit(rank=rank, qasm_str=qasm_str, max_depth=max_depth)
        result_list.append(depth)
        reduced_cnt_list.append(reduced_cnt)
        finished_count += 1

    # save to file
    with open(f"./tmp{gate_count_to_plot}/{rank}.tmp", 'wb') as handle:
        pickle.dump(obj=result_list, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"./tmp{gate_count_to_plot}/{rank}_reduced_cnt.tmp", 'wb') as handle:
        pickle.dump(obj=reduced_cnt_list, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    # more parameters
    random.seed(12345)
    sample_circuit_count = 1000
    max_search_depth = 8
    num_workers = 100

    # read json files and randomly sample a subset of circuits
    # we only consider circuits in first 8 layers
    y_list = [18, 176, 1208, 6463, 28513, 107788, 360775, 1106953]
    total_circuit_count = sum(y_list)
    circuit_dict_path = f"./final_dataset.json"
    with open(circuit_dict_path, 'r') as handle:
        circuit_dict = json.load(handle)
    print(f"There are {len(circuit_dict)} circuits in dataset.")
    print(f"Sample {sample_circuit_count} from first 8 layers ({total_circuit_count} circuits).")
    selected_hash_list = random.sample(list(circuit_dict)[:total_circuit_count], sample_circuit_count)
    selected_circuit_list = []
    for selected_hash in selected_hash_list:
        selected_circuit_list.append(circuit_dict[selected_hash])

    # make output dir
    if not os.path.exists(f"./tmp_gate_count"):
        os.makedirs(f"./tmp_gate_count")

    # compute min #xfers for these circuits
    circuit_per_worker = int(sample_circuit_count / num_workers)
    ctx = mp.get_context('spawn')
    process_group = []
    for idx in range(num_workers):
        begin = idx * circuit_per_worker
        end = begin + circuit_per_worker
        new_worker = ctx.Process(target=worker_proc,
                                 args=(idx, selected_circuit_list[begin:end],
                                       max_search_depth, "_gate_count"))
        new_worker.start()
        process_group.append(new_worker)
        time.sleep(0.2)

    # wait until all workers finish and gather information
    for worker in process_group:
        worker.join()
    final_results = {}
    for idx in range(num_workers):
        with open(f"./tmp_gate_count/{idx}.tmp", 'rb') as handle:
            result_list = pickle.load(file=handle)
            for result in result_list:
                if result not in final_results:
                    final_results[result] = 1
                else:
                    final_results[result] += 1
    print("/*************************************************/")
    print(f"Final results: {final_results}")
    print("/*************************************************/")
    with open(f"./final_results_gate_count.pickle", 'wb') as handle:
        pickle.dump(obj=final_results, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
