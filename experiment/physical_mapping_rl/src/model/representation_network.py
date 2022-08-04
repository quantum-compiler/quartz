from quartz import PyGraphState, PyMappingTable
import torch
import torch.nn as nn
import numpy as np
from dgl import DGLGraph
from src.model.circuit_gnn import CircuitGNN
from src.model.device_gnn import DeviceGNNGINLocal, DeviceEmbedding
from src.model.multihead_self_attention import EncoderLayer


class RepresentationNetwork(nn.Module):
    def __init__(self,
                 # DeviceGNN
                 device_feature_type,               # degree / id / both
                 num_degree_types,                  # max degree + 1
                 num_id_types,                      # max node id
                 degree_embedding_dim,              # dimension of feature embedding
                 id_embedding_dim,                  # dimension of id embedding
                 device_num_layers,                 # number of gnn convolution layers
                 device_hidden_dimension,           # dimension of each internal GNN layer
                 device_out_dimension,              # output dimension of final GNN layer
                 # Circuit GNN
                 circuit_num_layers,                # number of gnn convolution layers
                 num_gate_types,                    # number of different gate types
                 gate_type_embedding_dim,           # dimension of gate type embedding
                 circuit_conv_internal_dim,         # hidden layer dimension of each convolution layer's MLP
                 circuit_out_dimension,             # output dimension of final GNN layer
                 # Multi-head self-attention
                 final_mlp_hidden_dimension_ratio,  # final MLP's hidden dimension / raw representation dimension
                 num_attention_heads,               # number of attention heads
                 attention_qk_dimension,            # dimension of q vector and k vector in attention
                 attention_v_dimension,             # dimension of v vector in attention
                 ):
        super(RepresentationNetwork, self).__init__()

        # device gnn and circuit gnn
        self.device_gnn = DeviceGNNGINLocal(feature_type=device_feature_type, num_degree_types=num_degree_types,
                                            num_id_types=num_id_types, degree_embedding_dim=degree_embedding_dim,
                                            id_embedding_dim=id_embedding_dim, num_layers=device_num_layers,
                                            hidden_dimension=device_hidden_dimension,
                                            out_dimension=device_out_dimension)
        self.circuit_gnn = CircuitGNN(num_layers=circuit_num_layers, num_gate_types=num_gate_types,
                                      gate_type_embed_dim=gate_type_embedding_dim, h_feats=circuit_out_dimension,
                                      inter_dim=circuit_conv_internal_dim)

        # multi-head self attention
        raw_rep_dimension = device_out_dimension + circuit_out_dimension
        self.attention_encoder = EncoderLayer(d_model=raw_rep_dimension,
                                              d_inner=raw_rep_dimension * final_mlp_hidden_dimension_ratio,
                                              n_head=num_attention_heads,
                                              d_k=attention_qk_dimension, d_v=attention_v_dimension)

    def forward(self):
        pass


class RepresentationNetworkSimple(nn.Module):
    def __init__(self,
                 # DeviceGNN
                 num_registers,                     # number of registers
                 device_out_dimension,              # output dimension of device embedding network
                 # Circuit GNN
                 circuit_num_layers,                # number of gnn convolution layers
                 num_gate_types,                    # number of different gate types
                 gate_type_embedding_dim,           # dimension of gate type embedding
                 circuit_conv_internal_dim,         # hidden layer dimension of each convolution layer's MLP
                 circuit_out_dimension,             # output dimension of final GNN layer
                 # Multi-head self-attention
                 final_mlp_hidden_dimension_ratio,  # final MLP's hidden dimension / raw representation dimension
                 num_attention_heads,               # number of attention heads
                 attention_qk_dimension,            # dimension of q vector and k vector in attention
                 attention_v_dimension,             # dimension of v vector in attention
                 ):
        super(RepresentationNetworkSimple, self).__init__()

        # device gnn and circuit gnn
        self.device_embedding_network = DeviceEmbedding(num_registers=num_registers,
                                                        embedding_dimension=device_out_dimension)
        self.circuit_gnn = CircuitGNN(num_layers=circuit_num_layers, num_gate_types=num_gate_types,
                                      gate_type_embed_dim=gate_type_embedding_dim, h_feats=circuit_out_dimension,
                                      inter_dim=circuit_conv_internal_dim)

        # multi-head self attention
        self.device_out_dimension = device_out_dimension
        self.circuit_out_dimension = circuit_out_dimension
        self.raw_rep_dimension = device_out_dimension + circuit_out_dimension
        self.attention_encoder = EncoderLayer(d_model=self.raw_rep_dimension,
                                              d_inner=self.raw_rep_dimension * final_mlp_hidden_dimension_ratio,
                                              n_head=num_attention_heads,
                                              d_k=attention_qk_dimension, d_v=attention_v_dimension)

    def forward(self, circuit: PyGraphState, circuit_dgl: DGLGraph,
                physical2logical_mapping: PyMappingTable):
        # representation for each logical qubit
        num_qubits = sum(circuit.is_input)
        logical_qubit_rep = self.circuit_gnn(circuit_dgl)[:num_qubits]

        # append empty representation
        num_registers = len(physical2logical_mapping)
        logical_qubit_rep_padding = torch.zeros(num_registers - num_qubits, self.circuit_out_dimension)
        logical_qubit_rep = torch.concat([logical_qubit_rep, logical_qubit_rep_padding], dim=0)

        # reorder circuit representation
        qubit_physical_idx = circuit.input_physical_idx[:num_qubits]
        for i in range(num_registers):
            if i not in qubit_physical_idx:
                qubit_physical_idx.append(i)

        def invert_permutation(permutation):
            inv = np.empty_like(permutation)
            inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
            return inv

        inverted_physical_idx = invert_permutation(qubit_physical_idx)
        gather_indices = [inverted_physical_idx for _ in range(self.circuit_out_dimension)]
        gather_indices = np.array(gather_indices)
        gather_indices = torch.tensor(gather_indices).transpose(0, 1)
        sorted_logical_qubit_rep = torch.gather(input=logical_qubit_rep, dim=0, index=gather_indices)

        # concatenate with device embedding
        device_embedding_input = torch.tensor(list(range(num_registers)))
        device_embedding = self.device_embedding_network(device_embedding_input)
        concatenated_raw_rep = torch.concat([sorted_logical_qubit_rep, device_embedding], dim=1)
        concatenated_raw_rep = concatenated_raw_rep[None, :]

        # send into self attention layer and return
        register_representation, attention_score_mat = self.attention_encoder(concatenated_raw_rep)
        register_representation = register_representation[0, :]
        attention_score_mat = torch.sum(attention_score_mat[0, :], dim=0)
        return register_representation, attention_score_mat
