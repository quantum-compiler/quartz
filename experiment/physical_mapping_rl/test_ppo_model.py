from quartz import PySimplePhysicalEnv

from src.model.ppo_network import PPONetwork
from src.utils.utils import py_action_2_list, action_2_id


def main():
    # representation
    network = PPONetwork(num_registers=20, device_out_dimension=56,
                         circuit_num_layers=6, num_gate_types=30,
                         gate_type_embedding_dim=96, circuit_conv_internal_dim=128,
                         circuit_out_dimension=128,
                         final_mlp_hidden_dimension_ratio=4,
                         num_attention_heads=8,
                         attention_qk_dimension=64,
                         attention_v_dimension=64)
    network.eval()
    env1 = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    env2 = PySimplePhysicalEnv(qasm_file_path="tests/sabre.qasm", backend_type_str="IBM_Q20_TOKYO")
    state1 = env1.get_state()
    state2 = env2.get_state()
    action, log_prob = network.policy_forward(circuit_batch=[state1.circuit, state2.circuit],
                                              physical2logical_mapping_batch=[state1.physical2logical_mapping,
                                                                              state2.physical2logical_mapping],
                                              action_space_batch=[py_action_2_list(env1.get_action_space()),
                                                                  py_action_2_list(env2.get_action_space())])
    print(f"{action=}")
    print(f"{log_prob=}")
    print()

    log_prob2, entropy = network.evaluate_action(circuit_batch=[state1.circuit, state2.circuit],
                                                 physical2logical_mapping_batch=[state1.physical2logical_mapping,
                                                                                 state2.physical2logical_mapping],
                                                 action_space_batch=[py_action_2_list(env1.get_action_space()),
                                                                     py_action_2_list(env2.get_action_space())],
                                                 action_id_batch=[action_2_id(action[0],
                                                                              py_action_2_list(
                                                                                  env1.get_action_space())),
                                                                  action_2_id(action[1],
                                                                              py_action_2_list(
                                                                                  env2.get_action_space()))])
    print(f"{log_prob2}")
    print(f"{entropy}")
    print()

    log_prob2, entropy = network.evaluate_action(circuit_batch=[state1.circuit, state2.circuit],
                                                 physical2logical_mapping_batch=[state1.physical2logical_mapping,
                                                                                 state2.physical2logical_mapping],
                                                 action_space_batch=[py_action_2_list(env1.get_action_space()),
                                                                     py_action_2_list(env2.get_action_space())],
                                                 action_id_batch=[action_2_id(action[0],
                                                                              py_action_2_list(
                                                                                  env1.get_action_space())),
                                                                  action_2_id(action[1],
                                                                              py_action_2_list(
                                                                                  env2.get_action_space()))])
    print(f"This is supposed to be slightly different because of dropout in attention when training")
    print(f"{log_prob2}")
    print(f"{entropy}")
    print()

    value = network.value_forward(circuit_batch=[state1.circuit, state2.circuit],
                                  physical2logical_mapping_batch=[state1.physical2logical_mapping,
                                                                  state2.physical2logical_mapping])
    print(f"{value=}")
    print()


if __name__ == '__main__':
    main()
