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
                           d_inner=128,
                           n_head=3,
                           d_k=48,
                           d_v=64)
    env = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")

    # forward pass
    state = env.get_state()
    circuit_dgl = state.get_circuit_dgl()
    raw_rep = circuit_gnn(circuit_dgl)[:10]
    print(circuit_dgl)
    print(raw_rep.shape)
    real_rep = encoder(raw_rep)
    print(real_rep)


if __name__ == '__main__':
    main()
