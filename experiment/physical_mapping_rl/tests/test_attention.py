from quartz import PySimplePhysicalEnv

from src.model.circuit_gnn import CircuitGNN
from src.model.multihead_self_attention import EncoderLayer


def main():
    # initialize circuit GNN and multi-head self-attention
    circuit_gnn = CircuitGNN(num_layers=6,
                             num_gate_types=10,
                             gate_type_embed_dim=64,
                             h_feats=96,
                             inter_dim=128)
    encoder = EncoderLayer(d_model=96,
                           d_inner=384,
                           n_head=6,
                           d_k=64,
                           d_v=64)
    env = PySimplePhysicalEnv(qasm_file_path="rollout.qasm", backend_type_str="IBM_Q20_TOKYO")

    # forward pass
    state = env.get_state()
    num_qubits = sum(state.circuit.is_input)
    circuit_dgl = state.get_circuit_dgl()
    raw_rep = circuit_gnn(circuit_dgl)[:num_qubits]
    raw_rep = raw_rep[None, :]  # add batch dimension
    print(f"Shape after circuit GNN {raw_rep.shape}")  # (num_qubits, 96)
    real_rep = encoder(raw_rep)
    print(f"Shape after attention Rep:{real_rep[0].shape}, Attention score:{real_rep[1].shape}")


if __name__ == '__main__':
    main()
