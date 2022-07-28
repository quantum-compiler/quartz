from quartz import PySimplePhysicalEnv


def main():
    env = PySimplePhysicalEnv(qasm_file_path="./rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    print(state.get_circuit_dgl())
    print(state.get_device_dgl())


if __name__ == '__main__':
    main()
