import json


def main():
    # input parameters
    circuit_dict_path = "../data/900000_graph.json"

    # read json files and randomly sample a subset of circuits
    with open(circuit_dict_path, 'r') as handle:
        circuit_dict = json.load(handle)

    gate_count_distribution = {}
    for key in circuit_dict:
        circuit_pack = circuit_dict[key]
        gate_count = circuit_pack[1]
        if gate_count not in gate_count_distribution:
            gate_count_distribution[gate_count] = 1
        else:
            gate_count_distribution[gate_count] += 1
    print(gate_count_distribution)


if __name__ == '__main__':
    main()
