import time

from quartz import PySimplePhysicalEnv
from src.model.representation_network import RepresentationNetworkSimple


def main():
    network = RepresentationNetworkSimple(num_registers=20, device_out_dimension=56,
                                          circuit_num_layers=6, num_gate_types=30,
                                          gate_type_embedding_dim=96, circuit_conv_internal_dim=128,
                                          circuit_out_dimension=128,
                                          final_mlp_hidden_dimension_ratio=4,
                                          num_attention_heads=8,
                                          attention_qk_dimension=64,
                                          attention_v_dimension=64)
    env = PySimplePhysicalEnv(qasm_file_path="rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    start = time.time()
    rep, attention = network.forward(circuit=state.circuit, circuit_dgl=state.get_circuit_dgl(),
                                     physical2logical_mapping=state.physical2logical_mapping)
    end = time.time()
    inference_time = "{:.2f}".format(end - start)
    print(f"Inference time: {inference_time}s")
    print(f"{rep.shape=}")
    print(f"{attention.shape=}")


if __name__ == '__main__':
    main()
