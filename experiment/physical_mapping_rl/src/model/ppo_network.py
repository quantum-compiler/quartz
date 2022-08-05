import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.model.representation_network import RepresentationNetworkSimple
from src.model.actor_critic import ValueNetwork, PolicyNetworkSimple

from torch.distributions import Categorical


class PPONetwork(nn.Module):
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
        super(PPONetwork, self).__init__()
        self.representation_network = RepresentationNetworkSimple(num_registers=num_registers,
                                                                  device_out_dimension=device_out_dimension,
                                                                  circuit_num_layers=circuit_num_layers,
                                                                  num_gate_types=num_gate_types,
                                                                  gate_type_embedding_dim=gate_type_embedding_dim,
                                                                  circuit_conv_internal_dim=circuit_conv_internal_dim,
                                                                  circuit_out_dimension=circuit_out_dimension,
                                                                  final_mlp_hidden_dimension_ratio=final_mlp_hidden_dimension_ratio,
                                                                  num_attention_heads=num_attention_heads,
                                                                  attention_qk_dimension=attention_qk_dimension,
                                                                  attention_v_dimension=attention_v_dimension)
        self.value_network = ValueNetwork(register_embedding_dimension=circuit_out_dimension+device_out_dimension)
        self.policy_network = PolicyNetworkSimple   # this is a function

    def policy_forward(self, circuit, circuit_dgl, physical2logical_mapping, action_space):
        """
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
                action_space: action space
        output: selected action: a tuple (qubit idx 0, qubit idx 1)
                log probability of selected action
        """
        # get action probability
        _, attention_score = self.representation_network(circuit=circuit,
                                                         circuit_dgl=circuit_dgl,
                                                         physical2logical_mapping=physical2logical_mapping)
        action_prob = self.policy_network(attention_score, action_space)

        # sample action and return
        action_dist = Categorical(probs=action_prob)
        selected_action_id = action_dist.sample()
        selected_action_log_prob = action_dist.log_prob(selected_action_id)
        return action_space[selected_action_id], selected_action_log_prob

    def evaluate_action(self, circuit, circuit_dgl, physical2logical_mapping, action_space, action_id):
        """
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
                action_space: action space
                action_id: selected action id
        output: selected action: a tuple (qubit idx 0, qubit idx 1)
                log probability of selected action
        """
        # get action probability
        _, attention = self.representation_network(circuit=circuit,
                                                   circuit_dgl=circuit_dgl,
                                                   physical2logical_mapping=physical2logical_mapping)
        action_prob = self.policy_network(attention_score, action_space)

        # return statistics of given action
        action_dist = Categorical(probs=action_prob)
        action_log_prob = action_dist.log_prob(action_id)
        dist_entropy = action_dist.entropy()
        return action_log_prob, dist_entropy

    def value_forward(self, circuit, circuit_dgl, physical2logical_mapping):
        """
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
        output:
        """
        # get action probability
        register_rep, _ = self.representation_network(circuit=circuit,
                                                      circuit_dgl=circuit_dgl,
                                                      physical2logical_mapping=physical2logical_mapping)
        value = self.value_network(register_rep)
        return value
