from quartz import PySimplePhysicalEnv, BackendType


def main():
    circuit_file_name = "../sabre.qasm"
    env = PySimplePhysicalEnv(circuit_file_name, BackendType.IBM_Q20_TOKYO)
    state_before = env.get_state()
    action_space = env.get_action_space()
    print(state_before)
    print(action_space)


if __name__ == '__main__':
    main()
