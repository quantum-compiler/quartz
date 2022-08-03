from quartz import PySimplePhysicalEnv
from src.model.representation_network import RepresentationNetworkSimple


def main():
    network = RepresentationNetworkSimple(num_registers=20, device_out_dimension=7,
                                          circuit_num_layers=6, num_gate_types=30,
                                          gate_type_embedding_dim=16, circuit_conv_internal_dim=32,
                                          circuit_out_dimension=8,
                                          final_mlp_hidden_dimension_ratio=4,
                                          num_attention_heads=3,
                                          attention_qk_dimension=16,
                                          attention_v_dimension=17)
    env = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    rep, attention = network(state)
    print(f"{rep=}")
    print(f"{rep.shape=}")
    print(f"{attention=}")
    print(f"{attention.shape=}")


if __name__ == '__main__':
    main()
