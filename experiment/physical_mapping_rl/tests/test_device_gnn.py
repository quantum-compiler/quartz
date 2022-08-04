from quartz import PySimplePhysicalEnv

from src.model.device_gnn import DeviceGNNSAGE, DeviceGNNGINLocal, DeviceEmbedding

import torch

def test_sage():
    device_gnn = DeviceGNNSAGE(feature_type='both',  # degree / id / both
                               num_degree_types=20,
                               num_id_types=20,
                               degree_embedding_dim=48,
                               id_embedding_dim=16,
                               num_layers=5,
                               hidden_dimension=128,
                               out_dimension=56,)
    env = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    device_dgl = state.get_device_dgl()
    res = device_gnn(device_dgl)
    print(device_dgl)
    print(res.shape)


def test_gin_local():
    device_gnn = DeviceGNNGINLocal(feature_type='both',  # degree / id / both
                                   num_degree_types=20,
                                   num_id_types=20,
                                   degree_embedding_dim=48,
                                   id_embedding_dim=16,
                                   num_layers=5,
                                   hidden_dimension=128,
                                   out_dimension=84,)
    env = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    device_dgl = state.get_device_dgl()
    res = device_gnn(device_dgl)
    print(device_dgl)
    print(res.shape)


def test_device_embedding():
    device_embedding_network = DeviceEmbedding(20, 4)
    res = device_embedding_network(torch.tensor(list(range(20))))
    print(res)
    print(res.shape)


def main():
    test_sage()
    print()
    test_gin_local()
    print()
    test_device_embedding()


if __name__ == '__main__':
    main()
