from quartz import PySimplePhysicalEnv
from src.model.device_gnn import DeviceGNN


def main():
    device_gnn = DeviceGNN(num_feature_types=10,
                           feature_embedding_dim=32,
                           num_layers=5,
                           hidden_dimension=32,
                           out_dimension=16)
    env = PySimplePhysicalEnv(qasm_file_path="tests/rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    state = env.get_state()
    device_dgl = state.get_device_dgl()
    res = device_gnn(device_dgl)
    print(res.shape)


if __name__ == '__main__':
    main()
