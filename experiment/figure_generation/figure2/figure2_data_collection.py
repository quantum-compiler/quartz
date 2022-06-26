import json
import random


def analyze_circuits(circuit_list):
    pass


def main():
    # input parameters
    circuit_dict_path = "../data/900000_graph.json"
    total_circuit_count = 10
    num_workers = 1

    # read json files and randomly sample a subset of circuits
    with open(circuit_dict_path, 'r') as handle:
        circuit_dict = json.load(handle)
    selected_hash_list = random.sample(list(circuit_dict), total_circuit_count)
    selected_circuit_list = []
    for selected_hash in selected_hash_list:
        selected_circuit_list.append(circuit_dict[selected_hash])

    # compute min #xfers for these circuits
    # TODO: change this to multi-processing
    analyze_circuits(selected_circuit_list)


if __name__ == '__main__':
    main()
