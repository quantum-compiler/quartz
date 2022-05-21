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
from ActorCritic import ActorCritic


class RolloutBuffer:
    def __init__(self):
        self.graphs = []
        self.nodes = []
        self.xfers = []
        self.next_graphs = []
        self.next_nodes = []
        self.xfer_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.is_start_point = []

    def clear(self):
        self.__init__()


class PPO:
    def __init__(self, num_gate_type, context, graph_embed_size,
                 actor_hidden_size, critic_hidden_size, action_dim,
                 lr_graph_embedding, lr_actor, lr_critic, gamma, K_epochs,
                 eps_clip, log_file_handle, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(num_gate_type, graph_embed_size,
                                  actor_hidden_size, critic_hidden_size,
                                  action_dim, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([{
            'params':
            self.policy.graph_embedding.parameters(),
            'lr':
            lr_graph_embedding
        }, {
            'params':
            self.policy.actor.parameters(),
            'lr':
            lr_actor
        }, {
            'params':
            self.policy.critic.parameters(),
            'lr':
            lr_critic
        }])

        self.policy_old = ActorCritic(num_gate_type, graph_embed_size,
                                      actor_hidden_size, critic_hidden_size,
                                      action_dim, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.context = context

        self.log_file_handle = log_file_handle

    def select_action(self, graph):
        # Use the old policy network to select an action
        # No gradient needed
        with torch.no_grad():
            node, xfer, xfer_logprob = self.policy_old.act(self.context, graph)

        self.buffer.graphs.append(graph)
        self.buffer.nodes.append(node)
        self.buffer.xfers.append(xfer)
        self.buffer.xfer_logprobs.append(xfer_logprob)

        return node.item(), xfer.item()

    def update(self):
        # start = time.time()

        masks = torch.zeros((len(self.buffer.nodes), self.context.num_xfers),
                            dtype=torch.bool).to(self.device)
        for i, (graph,
                node) in enumerate(zip(self.buffer.graphs, self.buffer.nodes)):
            available_xfers = graph.available_xfers(
                context=self.context, node=graph.get_node_from_id(id=node))
            masks[i][available_xfers] = True

        # t_0 = time.time()
        # print(f"mask time: {t_0 - start}")
        # print(torch.cuda.memory_summary())

        gs = [g.to_dgl_graph() for g in self.buffer.graphs]
        batched_dgl_gs = dgl.batch(gs).to(self.device)

        dgl_next_gs = [g.to_dgl_graph() for g in self.buffer.next_graphs]
        batched_dgl_next_gs = dgl.batch(dgl_next_gs).to(self.device)

        # print(torch.cuda.memory_summary())

        node_nums = batched_dgl_gs.batch_num_nodes().tolist()
        next_node_nums = batched_dgl_next_gs.batch_num_nodes().tolist()

        next_node_lists = []
        for i in range(len(self.buffer.next_graphs)):
            node_list = torch.tensor(self.buffer.next_nodes[i],
                                     dtype=torch.int64)
            src_node_ids, _, edge_ids = dgl_next_gs[i].in_edges(node_list,
                                                                form='all')
            mask = dgl_next_gs[i].edata['reversed'][edge_ids] == 0
            node_list = torch.cat((node_list, src_node_ids[mask]))
            next_node_lists.append(node_list)

        old_xfer_logprobs = torch.squeeze(
            torch.stack(self.buffer.xfer_logprobs,
                        dim=0)).detach().to(self.device)

        # print(torch.cuda.memory_summary())

        # t_0 = time.time()
        # print(f"preprocessing time: {t_0 - start}")

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # Entropy is not needed when using old policy
            # But needed in current policy
            values, next_values, xfer_logprobs, xfer_entropy = self.policy.evaluate(
                batched_dgl_gs, self.buffer.nodes, self.buffer.xfers,
                batched_dgl_next_gs, next_node_lists, self.buffer.is_terminals,
                masks, node_nums, next_node_nums)

            # if _ == 0:
            #     print(torch.cuda.memory_summary())

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(xfer_logprobs - old_xfer_logprobs.detach())

            # Finding Surrogate Loss
            rewards = torch.stack(self.buffer.rewards).to(self.device)
            advantages = rewards + next_values * self.gamma - values
            surr1 = ratios * advantages.clone().detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 +
                                self.eps_clip) * advantages.clone().detach()

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = advantages.pow(2).mean()

            wandb.log({
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'xfer_entropy': xfer_entropy
            })

            # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
            #     state_values, rewards) - 0.01 * (node_entropy + xfer_entropy)
            loss = actor_loss + 0.5 * critic_loss - 0.1 * xfer_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            torch.cuda.empty_cache()

        for i in range(len(self.buffer.graphs)):
            if self.buffer.is_start_point[i]:
                self.log_file_handle.write(
                    f'initial gate count: {self.buffer.graphs[i].gate_count}\n'
                )
            message = f"node: {self.buffer.nodes[i]}\txfer: {self.buffer.xfers[i]}\treward: {self.buffer.rewards[i]}\tvalue: {values[i]:.3f}\tnext value: {next_values[i]:.3f}"
            if self.buffer.rewards[i] > 0:
                message += "\tReduced!!!"
            elif self.buffer.rewards[i] < 0:
                message += "\tIncreased..."
            # print(message)
            self.log_file_handle.write(message + '\n')
            self.log_file_handle.flush()
            if self.buffer.is_terminals[i]:
                # print("terminated")
                self.log_file_handle.write('terminated\n')

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        # print(f"update time: {time.time() - start}")

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(
            torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage))
