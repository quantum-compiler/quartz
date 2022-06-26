import json


def main():
    # input parameters
    circuit_dict_path = "../data/900000_graph.json"
    connectivity_dict_path = "../data/900000_path.json"
    num_workers = 1
    total_circuit_count = 10

    # read json files
    with open(circuit_dict_path, 'r') as handle:
        circuit_dict = json.load(handle)
    with open(connectivity_dict_path, 'r') as handle:
        connectivity_dict = json.load(handle)
    print()


if __name__ == '__main__':
    main()
