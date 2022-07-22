from quartz import PySimplePhysicalEnv


def main():
    env = PySimplePhysicalEnv(qasm_file_path="../sabre.qasm", backend_type_str="IBM_Q20_TOKYO")
    # action space
    action_space = env.get_action_space()
    for action in action_space:
        print(f"Type {action.type}, qubit_idx_0={action.qubit_idx_0}, qubit_idx_1={action.qubit_idx_1}.")
    print()

    # state
    cur_state = env.get_state()
    for device_edge in cur_state.device_edges_list:
        print(f"Device edge: {device_edge}")
    print()
    logical2physical = cur_state.logical2physical_mapping
    for logical_idx in logical2physical:
        print(f"Logical {logical_idx} -> Physical {logical2physical[logical_idx]}")
    print()
    physical2logical = cur_state.physical2logical_mapping
    for physical_idx in physical2logical:
        print(f"Physical {physical_idx} -> Logical {physical2logical[physical_idx]}")
    print()

    # move one step forward
    total_gate_count = cur_state.circuit.gate_count
    print(f"Total gate count is {total_gate_count}")
    selected_action = action_space[0]
    reward = env.step(action=selected_action)
    print(f"Selected Action: Type={selected_action.type}, qubit_idx_0={selected_action.qubit_idx_0},"
          f" qubit_idx_1={selected_action.qubit_idx_1}, reward is {reward}")
    print(f"Is circuit finished = {env.is_finished()}")

    # print state again
    cur_state = env.get_state()
    for device_edge in cur_state.device_edges_list:
        print(f"Device edge: {device_edge}")
    print()
    logical2physical = cur_state.logical2physical_mapping
    for logical_idx in logical2physical:
        print(f"Logical {logical_idx} -> Physical {logical2physical[logical_idx]}")
    print()
    physical2logical = cur_state.physical2logical_mapping
    for physical_idx in physical2logical:
        print(f"Physical {physical_idx} -> Logical {physical2logical[physical_idx]}")
    print()


if __name__ == '__main__':
    main()
