import pickle


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


def main():
    with open('./outputs/graph_10000_637_finished.dat', 'rb') as handle:
        recovered_obj = pickle.load(handle)
        print(len(recovered_obj))


if __name__ == '__main__':
    main()
