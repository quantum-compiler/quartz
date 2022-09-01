import math
import os
import time
from datetime import datetime

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN import QGNN
from IPython import embed
from torch.distributions import Categorical
from Utils import masked_softmax, temperature_softmax

from quartz import PyGraph, QuartzContext


class ActorCritic(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_gate_type,
        graph_embed_size,
        actor_hidden_size,
        critic_hidden_size,
        action_dim,
        device,
    ):
        super(ActorCritic, self).__init__()

        self.graph_embedding = QGNN(
            gnn_layers, num_gate_type, graph_embed_size, graph_embed_size
        )

        self.actor = nn.Sequential(
            nn.Linear(graph_embed_size, actor_hidden_size),
            nn.ReLU(),
            nn.Linear(actor_hidden_size, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(graph_embed_size, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1),
        )

        def _weight_init(m):
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                torch.nn.init.xavier_uniform_(m.weight, gain=gain)

        self.actor.apply(_weight_init)
        self.critic.apply(_weight_init)

        self.device = device
        self.action_dim = action_dim

    def forward(self):
        raise NotImplementedError

    def act(self, context, g, node_range):
        dgl_g = g.to_dgl_graph().to(self.device)

        graph_embed = self.graph_embedding(dgl_g)

        node_vs = self.critic(graph_embed).squeeze()

        if node_range == []:
            node_mask = torch.ones((dgl_g.number_of_nodes()), dtype=torch.bool).to(
                self.device
            )
        else:
            node_mask = torch.zeros((dgl_g.number_of_nodes()), dtype=torch.bool).to(
                self.device
            )
            node_mask[node_range] = True

        node_prob = F.softmax(node_vs, dim=-1)
        node_prob = masked_softmax(node_vs, node_mask)
        node_dist = Categorical(node_prob)
        node = node_dist.sample()

        # if node_range == []:
        #     print(node_vs)
        #     print(
        #         f'node: {node}, node value: {node_vs[node]}, max: {node_vs.max()}'
        #     )

        mask = torch.zeros((context.num_xfers), dtype=torch.bool).to(self.device)
        available_xfers = g.available_xfers_parallel(
            context=context, node=g.get_node_from_id(id=node)
        )
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

    def act_batch(self, context: QuartzContext, graphs):
        dgl_gs = [g.to_dgl_graph() for g in graphs]
        batched_dgl_gs = dgl.batch(dgl_gs).to(self.device)

        node_nums = batched_dgl_gs.batch_num_nodes().tolist()
        # ( sum(num of nodes in each graph), embed_size )
        graph_embeds = self.graph_embedding(batched_dgl_gs)
        # ( sum(num of nodes in each graph), )
        node_vss = self.critic(graph_embeds).squeeze()

        graph_embeds_list = torch.split(graph_embeds, node_nums)
        node_vs_list = torch.split(node_vss, node_nums)

        nodes = []
        node_embeds = []
        masks = []
        for i in range(len(graphs)):

            node_vs = node_vs_list[i]  # (num of nodes, )

            # Compute temperature
            # TODO: this rate should be a hyper parameter
            hit_rate = 0.9
            # Assume values of all nodes except one are 0
            temperature = 1 / math.log((node_nums[i] - 1) / (1 - hit_rate) * hit_rate)

            # node_probs = F.softmax(node_vs, dim=-1)
            node_probs = temperature_softmax(node_vs, temperature)
            node_dist = Categorical(probs=node_probs)
            node = node_dist.sample()
            nodes.append(node.item())

            node_embeds.append(graph_embeds_list[i][node])

            mask = torch.zeros(self.action_dim, dtype=torch.bool)
            available_xfers = graphs[i].available_xfers_parallel(
                context=context, node=graphs[i].get_node_from_id(id=node)
            )
            mask[available_xfers] = True
            masks.append(mask)

        node_embeds = torch.stack(node_embeds)  # (num of graph, embed_size)
        xfer_logits = self.actor(node_embeds)  # (num of graph, action_dim)

        masks = torch.stack(masks).to(self.device)
        xfer_probs = masked_softmax(xfer_logits, masks)

        xfer_dist = Categorical(probs=xfer_probs)
        xfers = xfer_dist.sample()
        xfer_logprobs = xfer_dist.log_prob(xfers)

        return nodes, xfers.tolist(), xfer_logprobs.detach(), masks

    def act_batch_beam(self, context, graphs, k):
        dgl_gs = [g.to_dgl_graph() for g in graphs]
        batched_dgl_gs = dgl.batch(dgl_gs).to(self.device)

        node_nums = batched_dgl_gs.batch_num_nodes().tolist()
        # ( sum(num of nodes in each graph), embed_size )
        graph_embeds = self.graph_embedding(batched_dgl_gs)
        # ( sum(num of nodes in each graph), )
        node_vss = self.critic(graph_embeds).squeeze()

        graph_embeds_list = torch.split(graph_embeds, node_nums)
        node_vs_list = torch.split(node_vss, node_nums)

        graphs_ = []
        nodes = []
        node_embeds = []
        masks = []
        for i in range(len(graphs)):

            node_vs = node_vs_list[i]  # (num of nodes, )

            # Compute temperature
            hit_rate = 0.9
            # Assume values of all nodes except one are 0
            temperature = 1 / math.log((node_nums[i] - 1) / (1 - hit_rate) * hit_rate)

            # node_probs = F.softmax(node_vs, dim=-1)
            node_probs = temperature_softmax(node_vs, temperature)
            # node_dist = Categorical(probs=node_probs)
            # node = node_dist.sample()
            node_k = torch.multinomial(node_probs, k)
            nodes += node_k.tolist()
            # print(node_k)

            # node_embeds += graph_embeds_list[i][node_k].tolist()

            for j in range(k):
                node_embeds.append(graph_embeds_list[i][node_k[j]])
                graphs_.append(graphs[i])
                mask = torch.zeros(self.action_dim, dtype=torch.bool)
                available_xfers = graphs[i].available_xfers_parallel(
                    context=context, node=graphs[i].get_node_from_id(id=node_k[j])
                )
                mask[available_xfers] = True
                masks.append(mask)

        # print(node_embeds)
        node_embeds = torch.stack(node_embeds)  # (num of graph, embed_size)
        xfer_logits = self.actor(node_embeds)  # (num of graph, action_dim)

        masks = torch.stack(masks).to(self.device)
        xfer_probs = masked_softmax(xfer_logits, masks)

        xfer_dist = Categorical(probs=xfer_probs)
        xfers = xfer_dist.sample()
        xfer_logprobs = xfer_dist.log_prob(xfers)

        return graphs_, nodes, xfers.tolist(), xfer_logprobs.detach(), masks

    def get_local_max_value(self, g, nodes):
        with torch.no_grad():
            dgl_g = g.to_dgl_graph().to(self.device)
            graph_embed = self.graph_embedding(dgl_g)
            values = self.critic(graph_embed).squeeze()
        return values[nodes].max()

    def evaluate(
        self,
        batched_dgl_gs,
        nodes,
        xfers,
        batched_dgl_next_gs,
        next_node_lists,
        is_nops,
        is_terminals,
        masks,
        node_nums,
        next_node_nums,
    ):

        batched_dgl_gs = batched_dgl_gs.to(self.device)
        # start = time.time()
        batched_graph_embeds = self.graph_embedding(batched_dgl_gs)
        # t_0 = time.time()
        # print(f'graph time: {t_0 - start}')

        # Split batched tensors into lists
        graph_embed_list = torch.split(batched_graph_embeds, node_nums)

        selected_node_embeds = []
        for i in range(batched_dgl_gs.batch_size):
            selected_node_embeds.append(graph_embed_list[i][nodes[i]])
        selected_node_embeds = torch.stack(selected_node_embeds)

        # Get values
        values = self.critic(selected_node_embeds).squeeze()
        # t_1 = time.time()
        # print(f'get value time: {t_1 - t_0}')

        # Get xfer logprobs and xfer entropy
        xfer_logits = self.actor(selected_node_embeds)
        xfer_probs = masked_softmax(xfer_logits, masks)
        xfer_dists = Categorical(xfer_probs)
        xfer_logprobs = xfer_dists.log_prob(
            torch.tensor(xfers, dtype=torch.int).to(self.device)
        )
        xfer_entropys = xfer_dists.entropy()
        # t_2 = time.time()
        # print(f'get xfer time: {t_2 - t_1}')

        # Get next node values
        with torch.no_grad():
            batched_dgl_next_gs = batched_dgl_next_gs.to(self.device)
            batched_next_graph_embeds = self.graph_embedding(batched_dgl_next_gs)
            batched_next_node_vs = self.critic(batched_next_graph_embeds)

        # Split
        next_node_vs_list = torch.split(batched_next_node_vs, next_node_nums)

        next_values = []
        for i in range(batched_dgl_gs.batch_size):
            if is_nops[i]:
                next_value = torch.tensor(0).to(self.device)
            elif is_terminals[i]:
                next_value = torch.tensor(0).to(self.device)
            else:
                # node_list contains "next nodes" and their neighbors
                # we choose the max as the next value
                node_list = next_node_lists[i]
                if list(node_list) == []:
                    next_value = torch.tensor(0).to(self.device)
                else:
                    next_value = torch.max(next_node_vs_list[i][node_list])
            next_values.append(next_value)
        next_values = torch.stack(next_values)
        # t_3 = time.time()
        # print(f'get next node value time: {t_3 - t_2}')

        return values, next_values, xfer_logprobs, xfer_entropys
