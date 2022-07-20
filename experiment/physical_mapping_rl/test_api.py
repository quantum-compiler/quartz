from quartz import PySimplePhysicalEnv


def main():
    env = PySimplePhysicalEnv(qasm_file_path="./sabre.qasm", backend_type_str="IBM_Q20_TOKYO")
    action_space = env.get_action_space()
    for action in action_space:
        print(f"Type {action.type}, qubit_idx_0={action.qubit_idx_0}, qubit_idx_1={action.qubit_idx_1}.")


if __name__ == '__main__':
    main()
