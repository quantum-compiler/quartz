import torch
from src.model.ppo_network import PPONetwork
from src.utils.utils import update_linear_schedule


class PPOModel:
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
                 # optimization process
                 device,                            # device the model is on
                 lr,                                # learning rate
                 opti_eps,                          # eps used in adam
                 weight_decay,                      # weight decay
                 ):
        # network
        self.actor_critic = PPONetwork(num_registers=num_registers,
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
        self.actor_critic.to(device)

        # optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                          lr=lr, eps=opti_eps, weight_decay=weight_decay)

    def lr_decay(self, cur_episode, total_episodes):
        update_linear_schedule(self.optimizer, cur_episode, total_episodes, self.lr)

    def get_actions(self, circuit, circuit_dgl, physical2logical_mapping, action_space):
        """
        Compute actions and value function predictions for the given inputs.
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
                action_space: decoded action space (see utils.DecodePyActionList)
        output: value, selected action, log_probability of selected action
        """
        action, action_log_prob = \
            self.actor_critic.policy_forward(circuit=circuit,
                                             circuit_dgl=circuit_dgl,
                                             physical2logical_mapping=physical2logical_mapping,
                                             action_space=action_space)
        value = self.actor_critic.value_forward(circuit=circuit,
                                                circuit_dgl=circuit_dgl,
                                                physical2logical_mapping=physical2logical_mapping)
        return value, action, action_log_prob

    def get_values(self, circuit, circuit_dgl, physical2logical_mapping):
        """
        Get value function predictions.
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
        output: value
        """
        value = self.actor_critic.value_forward(circuit=circuit,
                                                circuit_dgl=circuit_dgl,
                                                physical2logical_mapping=physical2logical_mapping)
        return value

    def evaluate_actions(self, circuit, circuit_dgl, physical2logical_mapping,
                         action_space, action_id):
        """
        Get action log_prob / entropy and value function predictions for actor update.
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
                action_space: decoded action space (see utils.DecodePyActionList)
                action_id: selected action id
        output: value, log probabilities of the input action, distribution entropy
        """
        action_log_prob, dist_entropy = \
            self.actor_critic.evaluate_action(circuit=circuit,
                                              circuit_dgl=circuit_dgl,
                                              physical2logical_mapping=physical2logical_mapping,
                                              action_space=action_space,
                                              action_id=action_id)
        value = self.actor_critic.value_forward(circuit=circuit,
                                                circuit_dgl=circuit_dgl,
                                                physical2logical_mapping=physical2logical_mapping)
        return value, action_log_prob, dist_entropy

    def act(self, circuit, circuit_dgl, physical2logical_mapping, action_space):
        """
        Compute actions using the given inputs.
        input:  circuit, circuit_dgl, physical2logical_mapping: observation
        output:
        """
        action, _ = self.actor_critic.policy_forward(circuit=circuit,
                                                     circuit_dgl=circuit_dgl,
                                                     physical2logical_mapping=physical2logical_mapping,
                                                     action_space=action_space)
        return action
