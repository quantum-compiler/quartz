from quartz import PySimplePhysicalEnv

from src.model.actor_critic import ValueNetwork, PolicyNetworkSimple
from src.model.representation_network import RepresentationNetworkSimple
from src.utils.utils import py_action_2_list


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
    env1 = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    env2 = PySimplePhysicalEnv(qasm_file_path="tests/sabre.qasm", backend_type_str="IBM_Q20_TOKYO")
    state1 = env1.get_state()
    state2 = env2.get_state()
    rep, attention = rep_network.forward(circuit_batch=[state1.circuit, state2.circuit],
                                         physical2logical_mapping_batch=[state1.physical2logical_mapping,
                                                                         state2.physical2logical_mapping])
    print(f"{rep.shape=}")
    print(f"{attention.shape=}")

    # value
    value_network = ValueNetwork(128 + 56)
    value = value_network(rep)
    print(f"{value=}")

    # policy
    policy = PolicyNetworkSimple(attention_score_batch=attention,
                                 action_space_batch=[py_action_2_list(env1.get_action_space()),
                                                     py_action_2_list(env2.get_action_space())])
    print(f"{policy[0]=}")
    print(sum(state1.circuit.is_input))
    print(state1.logical2physical_mapping)
    print(f"{py_action_2_list(env1.get_action_space())}")

    print(f"{policy[1]=}")
    print(sum(state2.circuit.is_input))
    print(state2.logical2physical_mapping)
    print(f"{py_action_2_list(env2.get_action_space())}")


if __name__ == '__main__':
    main()
