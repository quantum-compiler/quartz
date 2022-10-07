import json


# Since we have already gathered 160k circuits with 48 gates in dataset,
# for categorical generator, we do not need to re-run on 48. Instead, we
# extract from our previous dataset.
def main():
    # open the dataset and create a set of qasm files
    with open(f"../data/900000_graph.json", 'r') as handle:
        circuit_dict = json.load(handle)
    circuit_48_set = {}
    for circuit_hash in circuit_dict:
        circuit_pack = circuit_dict[circuit_hash]
        if circuit_pack[1] == 48:
            circuit_48_set[circuit_hash] = circuit_pack
    print(f"There are {len(circuit_48_set)} circuits gathered with 48 gates.")
    with open(f"./48.json", 'w') as handle:
        json.dump(circuit_48_set, handle, indent=2)


if __name__ == '__main__':
    main()
