from quartz import PySimplePhysicalEnv
import torch
import torch.nn as nn
from circuit_gnn import CircuitGNN
from device_gnn import DeviceGNNGINLocal
from multihead_self_attention import EncoderLayer


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
