import torch
from gnn import QGNN
import os
from datetime import datetime
import torch.nn as nn
from torch.distributions import Categorical
import quartz
import torch.nn.functional as F
import numpy as np
import dgl
import time
from tqdm import tqdm
import wandb
from collections import deque
import random
import sys
from Utils import masked_softmax


class ActorCritic(nn.Module):
    def __init__(self, num_gate_type, graph_embed_size, actor_hidden_size,
                 critic_hidden_size, action_dim, device):
        super(ActorCritic, self).__init__()

        self.graph_embedding = QGNN(6, num_gate_type, graph_embed_size,
                                    graph_embed_size)

        self.actor = nn.Sequential(
            nn.Linear(graph_embed_size, actor_hidden_size), nn.ReLU(),
            nn.Linear(actor_hidden_size, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(graph_embed_size, critic_hidden_size), nn.ReLU(),
            nn.Linear(critic_hidden_size, 1))

        self.device = device

    def forward(self):
        raise NotImplementedError

    def act(self, context, g):
        dgl_g = g.to_dgl_graph().to(self.device)

        # Used critic network to select node
        graph_embed = self.graph_embedding(dgl_g)

        node_vs = self.critic(graph_embed).squeeze()
        node_prob = F.softmax(node_vs, dim=-1)
        node_dist = Categorical(node_prob)
        node = node_dist.sample()

        mask = torch.zeros((context.num_xfers),
                           dtype=torch.bool).to(self.device)
        available_xfers = g.available_xfers(context=context,
                                            node=g.get_node_from_id(id=node))
        mask[available_xfers] = True
        xfer_logits = self.actor(graph_embed[node])
        xfer_probs = masked_softmax(xfer_logits, mask)
        xfer_dist = Categorical(xfer_probs)
        xfer = xfer_dist.sample()
        xfer_logprob = xfer_dist.log_prob(xfer)

        # Detach here because we use old policy to select actions
        # return node.detach(), xfer.detach(), node_logprob.detach(
        # ), xfer_logprob.detach()
        return node.detach(), xfer.detach(), xfer_logprob.detach(), mask

    # def evaluate(self, batched_dgl_gs, nodes, xfers, batched_dgl_next_gs,
    #              next_node_lists, is_terminals, masks, node_nums,
    #              next_node_nums):
    #     # start = time.time()
    #     batched_graph_embeds = self.graph_embedding(batched_dgl_gs)
    #     batched_node_vs = self.critic(batched_graph_embeds).squeeze()

    #     with torch.no_grad():
    #         batched_next_graph_embeds = self.graph_embedding(
    #             batched_dgl_next_gs)
    #         batched_next_node_vs = self.critic(batched_next_graph_embeds)

    #     # Split batched tensors into lists
    #     graph_embed_list = torch.split(batched_graph_embeds, node_nums)
    #     node_vs_list = torch.split(batched_node_vs, node_nums)
    #     next_node_vs_list = torch.split(batched_next_node_vs, next_node_nums)

    #     # t_0 = time.time()
    #     # print(f"time neural network: {t_0 - start}")

    #     values = []
    #     next_values = []

    #     for i in range(batched_dgl_gs.batch_size):
    #         value = node_vs_list[i][nodes[i]]
    #         values.append(value)

    #     # t_1 = time.time()
    #     # print(f"time get_values: {t_1 - t_0}")

    #     for i in range(batched_dgl_gs.batch_size):
    #         if is_terminals[i]:
    #             next_value = torch.tensor(0).to(self.device)
    #         else:
    #             # node_list contains "next nodes" and their neighbors
    #             # we choose the max as the next value
    #             node_list = next_node_lists[i]
    #             if list(node_list) == []:
    #                 next_value = torch.tensor(0).to(self.device)
    #             else:
    #                 next_value = torch.max(next_node_vs_list[i][node_list.to(
    #                     self.device)])
    #         next_values.append(next_value)

    #     # t_2 = time.time()
    #     # print(f"time get next values: {t_2 - t_1}")

    #     selected_node_embeds = []
    #     for i in range(batched_dgl_gs.batch_size):
    #         selected_node_embeds.append(graph_embed_list[i][nodes[i]])
    #     selected_node_embeds = torch.stack(selected_node_embeds)
    #     xfer_logits = self.actor(selected_node_embeds)
    #     xfer_probs = masked_softmax(xfer_logits, masks)
    #     xfer_dists = Categorical(xfer_probs)
    #     xfer_logprobs = xfer_dists.log_prob(
    #         torch.tensor(xfers, dtype=torch.int).to(self.device))
    #     xfer_entropy = xfer_dists.entropy().mean()

    #     values = torch.stack(values)
    #     next_values = torch.stack(next_values)

    #     # t_3 = time.time()
    #     # print(f"time get logprob: {t_3 - t_2}")
    #     # print(f"evaluation time: {time.time() - start}")

    #     return values, next_values, xfer_logprobs, xfer_entropy

    def evaluate(self, batched_dgl_gs, nodes, xfers, batched_dgl_next_gs,
                 next_node_lists, is_terminals, masks, node_nums,
                 next_node_nums):
        batched_dgl_gs = batched_dgl_gs.to(self.device)
        batched_graph_embeds = self.graph_embedding(batched_dgl_gs)
        batched_node_vs = self.critic(batched_graph_embeds).squeeze()

        # Split batched tensors into lists
        graph_embed_list = torch.split(batched_graph_embeds, node_nums)
        node_vs_list = torch.split(batched_node_vs, node_nums)

        # Get node values
        values = []
        for i in range(batched_dgl_gs.batch_size):
            value = node_vs_list[i][nodes[i]]
            values.append(value)
        values = torch.stack(values)

        # Get xfer logprobs and xfer entropy
        selected_node_embeds = []
        for i in range(batched_dgl_gs.batch_size):
            selected_node_embeds.append(graph_embed_list[i][nodes[i]])
        selected_node_embeds = torch.stack(selected_node_embeds)
        xfer_logits = self.actor(selected_node_embeds)
        xfer_probs = masked_softmax(xfer_logits, masks)
        xfer_dists = Categorical(xfer_probs)
        xfer_logprobs = xfer_dists.log_prob(
            torch.tensor(xfers, dtype=torch.int).to(self.device))
        xfer_entropy = xfer_dists.entropy().mean()

        # Get next node values
        with torch.no_grad():
            batched_dgl_next_gs = batched_dgl_next_gs.to(self.device)
            batched_next_graph_embeds = self.graph_embedding(
                batched_dgl_next_gs)
            batched_next_node_vs = self.critic(batched_next_graph_embeds)

        # Split
        next_node_vs_list = torch.split(batched_next_node_vs, next_node_nums)

        next_values = []
        for i in range(batched_dgl_gs.batch_size):
            if is_terminals[i]:
                next_value = torch.tensor(0).to(self.device)
            else:
                # node_list contains "next nodes" and their neighbors
                # we choose the max as the next value
                node_list = next_node_lists[i]
                if list(node_list) == []:
                    next_value = torch.tensor(0).to(self.device)
                else:
                    next_value = torch.max(next_node_vs_list[i][node_list.to(
                        self.device)])
            next_values.append(next_value)
        next_values = torch.stack(next_values)

        return values, next_values, xfer_logprobs, xfer_entropy