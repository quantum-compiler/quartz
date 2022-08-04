import time

from quartz import PySimplePhysicalEnv

from src.model.representation_network import RepresentationNetworkSimple
from src.model.ppo_network import ValueNetwork, PolicyNetworkSimple


def main():
    # representation
    rep_network = RepresentationNetworkSimple(num_registers=20, device_out_dimension=56,
                                              circuit_num_layers=6, num_gate_types=30,
                                              gate_type_embedding_dim=96, circuit_conv_internal_dim=128,
                                              circuit_out_dimension=128,
                                              final_mlp_hidden_dimension_ratio=4,
                                              num_attention_heads=8,
                                              attention_qk_dimension=64,
                                              attention_v_dimension=64)
    env = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    rep, attention = rep_network(state)
    print(f"{rep.shape=}")
    print(f"{attention.shape=}")

    # value
    value_network = ValueNetwork(128 + 56)
    value = value_network(rep)
    print(f"{value=}")

    # policy
    action_space = env.get_action_space()
    policy = PolicyNetworkSimple(attention_score=attention, action_space=action_space)
    print(f"{policy=}")
    print(f"{policy.shape=}", f"{len(action_space)}")


if __name__ == '__main__':
    main()
