import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self,
                 register_embedding_dimension,      # dimension of register embedding
                 ):
        super(ValueNetwork, self).__init__()
        hidden_dim1 = 8 * math.ceil(math.log2(register_embedding_dimension / 3))
        hidden_dim2 = 8 * math.ceil(math.log2(register_embedding_dimension / 5))
        hidden_dim3 = 8 * math.ceil(math.log2(register_embedding_dimension / 11))
        self.linear1 = nn.Linear(register_embedding_dimension, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear3 = nn.Linear(hidden_dim3, 1)

    def forward(self, register_embeddings):
        """
        register_embeddings: (num registers, register_embedding_dimension)
        """
        state_embedding = torch.sum(register_embeddings, dim=1)
        hidden1 = self.linear1(state_embedding)
        hidden2 = self.linear2(hidden1)
        value = self.linear3(hidden2)
        return value


def PolicyNetworkSimple(attention_score_batch, action_space_batch):
    # calculate logit for each action
    logit_mat = attention_score_batch + attention_score_batch.transpose(1, 2)
    action_prob_batch = []
    for idx, action_space in enumerate(action_space_batch):
        logit_list = []
        for action in action_space:
            assert not action[0] == action[1]
            action_logit = logit_mat[idx][action[0]][action[1]].view(1)
            logit_list.append(action_logit)
        action_logits = torch.concat(logit_list, dim=0)
        action_prob = F.softmax(action_logits, dim=0)
        action_prob_batch.append(action_prob)
    return action_prob_batch
