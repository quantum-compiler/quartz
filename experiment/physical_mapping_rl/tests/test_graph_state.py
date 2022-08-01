from quartz import PySimplePhysicalEnv


def main():
    env = PySimplePhysicalEnv(qasm_file_path="./sabre.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    graph_state = state.circuit
    print(f"{graph_state.node_id=}")
    print(f"{graph_state.is_input=}")
    print(f"{graph_state.input_logical_idx=}")
    print(f"{graph_state.input_physical_idx=}")
    print(f"{graph_state.node_type=}")
    print(f"{graph_state.number_of_edges=}")
    print(f"{graph_state.edge_from=}")
    print(f"{graph_state.edge_to=}")
    print(f"{graph_state.edge_reversed=}")
    print(f"{graph_state.edge_logical_idx=}")
    print(f"{graph_state.edge_physical_idx=}")
    print(f"{state.get_circuit_dgl()=}")


if __name__ == '__main__':
    main()
