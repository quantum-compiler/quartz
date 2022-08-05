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

    def get_actions(self, circuit_batch, physical2logical_mapping_batch, action_space_batch):
        """
        Compute actions and value function predictions for the given inputs.
        input:  circuit_batch, physical2logical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
        output: value, selected action, log_probability of selected action (each is a list)
        """
        selected_action_list, selected_action_prob_list = \
            self.actor_critic.policy_forward(circuit_batch=circuit_batch,
                                             physical2logical_mapping_batch=physical2logical_mapping_batch,
                                             action_space_batch=action_space_batch)
        value_batch = self.actor_critic.value_forward(circuit_batch=circuit_batch,
                                                      physical2logical_mapping_batch=physical2logical_mapping_batch)
        return value_batch, selected_action_list, selected_action_prob_list

    def get_values(self, circuit_batch, physical2logical_mapping_batch):
        """
        Get value function predictions.
        input:  circuit, physical2logical_mapping: list of observations
        output: a list of values
        """
        value_batch = self.actor_critic.value_forward(circuit_batch=circuit_batch,
                                                      physical2logical_mapping_batch=physical2logical_mapping_batch)
        return value_batch

    def evaluate_actions(self, circuit_batch, physical2logical_mapping_batch, action_space_batch, action_id_batch):
        """
        Get action log_prob / entropy and value function predictions for actor update.
        input:  circuit_batch, physical2logical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
                action_id_batch: list of selected actions index
        output: value, log probabilities of the input action, distribution entropy (each is a list)
        """
        action_log_prob_batch, dist_entropy_batch = \
            self.actor_critic.evaluate_action(circuit_batch=circuit_batch,
                                              physical2logical_mapping_batch=physical2logical_mapping_batch,
                                              action_space_batch=action_space_batch,
                                              action_id_batch=action_id_batch)
        value_batch = self.actor_critic.value_forward(circuit_batch=circuit_batch,
                                                      physical2logical_mapping_batch=physical2logical_mapping_batch)
        return value_batch, action_log_prob_batch, dist_entropy_batch

    def act(self, circuit_batch, physical2logical_mapping_batch, action_space_batch):
        """
        Compute actions using the given inputs.
        input:  circuit_batch, physical2logical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
        output: a list of selected actions
        """
        selected_action_list, _ = \
            self.actor_critic.policy_forward(circuit_batch=circuit_batch,
                                             physical2logical_mapping_batch=physical2logical_mapping_batch,
                                             action_space_batch=action_space_batch)
        return selected_action_list
