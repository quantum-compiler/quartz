import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gnn import QGNN
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import TransfoXLConfig, TransfoXLModel

import quartz

device = torch.device('cpu')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

# Constants
gate_type_num = 29


def masked_softmax(logits, mask):
    if mask.sum() != 0:
        mask = torch.ones_like(mask, dtype=torch.bool) ^ mask
        logits[mask] -= 1.0e10
    return F.softmax(logits)


class ActorCritic(nn.Module):
    def __init__(self, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        transformer_config = TransfoXLConfig(
            vocab_size=0,
            cutoffs=[],
            d_model=hidden_size,
            d_embed=hidden_size,
            n_head=8,
            n_layer=5,
        )
        self.graph_embeding = QGNN(7, gate_type_num, hidden_size, hidden_size)
        # self.graph_embeding = QGNN(gate_type_num, hidden_size, hidden_size)

        self.actor_transformer = TransfoXLModel(transformer_config)
        self.actor_node_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.actor_xfer_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        self.critic_transformer = TransfoXLModel(transformer_config)
        self.critic_value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, g, context):
        dgl_g = g.to_dgl_graph().to(device)

        graph_embed = self.graph_embeding(dgl_g)
        transformer_output = self.actor_transformer(
            inputs_embeds=graph_embed.unsqueeze(0)
        )
        graph_repr = transformer_output[0].squeeze(0)

        node_logit = self.actor_node_head(graph_repr).squeeze(1)
        node_prob = F.softmax(node_logit)
        # print(f'node_prob shape: {node_prob.shape}')
        node_dist = Categorical(node_prob)
        node = node_dist.sample()
        # print(f'node: {node}')
        node_log_prob = node_dist.log_prob(node)
        node_entropy = node_dist.entropy()

        mask = torch.zeros((context.num_xfers), dtype=torch.bool).to(device)
        available_xfers = g.available_xfers(
            context=context, node=g.get_node_from_id(id=node)
        )
        mask[available_xfers] = True
        xfer_logit = self.actor_xfer_head(graph_repr)
        xfer_probs = masked_softmax(xfer_logit[node], mask)
        xfer_dist = Categorical(xfer_probs)
        xfer = xfer_dist.sample()
        xfer_log_prob = xfer_dist.log_prob(xfer)
        xfer_entropy = xfer_dist.entropy()

        transformer_output = self.critic_transformer(
            inputs_embeds=graph_embed.unsqueeze(0)
        )
        value_repr = transformer_output[0].squeeze(0)
        # print(value_repr)
        value = self.critic_value_head(value_repr).squeeze()
        value = value.sum()
        print(f'value: {value}')

        return (
            node.item(),
            node_log_prob,
            node_entropy,
            xfer.item(),
            xfer_log_prob,
            xfer_entropy,
            value,
        )

    def get_value(self, g):
        dgl_g = g.to_dgl_graph().to(device)
        graph_embed = self.graph_embeding(dgl_g)
        transformer_output = self.critic_transformer(
            inputs_embeds=graph_embed.unsqueeze(0)
        )
        value_repr = transformer_output[0].squeeze(0)
        value_repr, _ = torch.max(value_repr, dim=0)
        # print(value_repr)
        value = self.critic_value_head(value_repr).squeeze()
        return value


def get_trajectory(
    device,
    max_seq_len,
    model,
    invalid_reward,
    init_state,
    context,
    gate_count_upper_limit,
    best_gate_count,
):
    node_log_probs = []
    xfer_log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    # rollout trajectory
    seq_len = 0
    done = False
    graph = init_state

    for seq_cnt in range(max_seq_len):
        if not done:
            (
                node,
                node_log_prob,
                node_entropy,
                xfer,
                xfer_log_prob,
                xfer_entropy,
                value,
            ) = model(graph, context)
            print(f'{node}, {xfer}')
            next_graph = graph.apply_xfer(
                xfer=context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node),
            )

            if next_graph == None:
                reward = invalid_reward
                done = True
            else:
                next_gate_cnt = next_graph.gate_count
                reward = graph.gate_count - next_gate_cnt
                # Eliminate circuits with overly large gate count
                if next_gate_cnt > gate_count_upper_limit:
                    done = True
                if reward > 0:
                    print(f'positive reward! {graph.gate_count} -> {next_gate_cnt}')
                if next_gate_cnt < best_gate_count:
                    best_gate_count = next_gate_cnt
            seq_len = seq_cnt + 1

            reward = torch.tensor(reward)
            entropy += node_entropy + xfer_entropy

        else:
            break

        node_log_probs.append(node_log_prob)
        xfer_log_probs.append(xfer_log_prob)
        values.append(value)
        rewards.append(reward)
        masks.append(1 - done)

        graph = next_graph

    if next_graph == None:
        next_value = 0
    else:
        next_value = model.get_value(next_graph)

    rewards = torch.stack(rewards).to(device)
    # print(rewards)
    # print(values)
    values = torch.stack(values).to(device)
    # print(values)
    # log_probs = torch.stack(log_probs).to(device)
    node_log_probs = torch.stack(node_log_probs).to(device)
    xfer_log_probs = torch.stack(xfer_log_probs).to(device)
    masks = torch.tensor(masks).to(device)
    qs = compute_qs(next_value, rewards, masks)

    return (
        qs,
        values,
        node_log_probs,
        xfer_log_probs,
        entropy,
        seq_len,
        rewards.sum().cpu().item(),
        best_gate_count,
    )


def compute_qs(next_value, rewards, masks, gamma=0.99):
    R = next_value
    qs = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        qs.insert(0, R)
    # print(qs)
    return torch.tensor(qs).to(device)


def a2c(
    hidden_size,
    context,
    init_graph,
    episodes=20000,
    lr=1e-3,
    max_seq_len=5,
    invalid_reward=-5,
    batch_size=100,
):

    num_actions = context.num_xfers
    best_gate_count = init_graph.gate_count
    gate_count_upper_limit = best_gate_count * 1.1
    model = ActorCritic(num_actions, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: scheduler

    reward_log = []
    seq_len_log = []

    for _ in tqdm(range(episodes), mininterval=10):

        node_log_probs = torch.tensor([], dtype=torch.float).to(device)
        xfer_log_probs = torch.tensor([], dtype=torch.float).to(device)
        values = torch.tensor([], dtype=torch.float).to(device)
        qs = torch.tensor([], dtype=torch.float).to(device)
        average_seq_len = 0
        average_reward = 0
        entropy = 0

        # rollout trajectory
        for i in range(batch_size):
            (
                qs_,
                values_,
                node_log_probs_,
                xfer_log_probs_,
                entropy_,
                seq_len_,
                total_rewards_,
                best_gate_count,
            ) = get_trajectory(
                device,
                max_seq_len,
                model,
                invalid_reward,
                init_graph,
                context,
                gate_count_upper_limit,
                best_gate_count,
            )

            node_log_probs = torch.cat((node_log_probs, node_log_probs_))
            xfer_log_probs = torch.cat((xfer_log_probs, xfer_log_probs_))
            values = torch.cat((values, values_))
            qs = torch.cat((qs, qs_))
            entropy += entropy_
            average_seq_len += seq_len_
            average_reward += total_rewards_

        print(f'qs: {qs}')
        print(f'values: {values}')
        advantage = qs - values
        print(f'advantage: {advantage}')
        # print(f'log_probs: {log_probs}')
        print(f'node_log_probs: {node_log_probs}')
        print(f'xfer_log_probs: {xfer_log_probs}')

        actor_loss = -(
            (node_log_probs + xfer_log_probs) * advantage.clone().detach()
        ).mean()
        print(f'actor_loss: {actor_loss.item()}')
        critic_loss = advantage.pow(2).mean()
        print(f'critic_loss: {critic_loss.item()}')

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        print(f'entropy: {entropy.item()}')
        print(f'loss: {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            if param.grad == None:
                # print(param.shape)
                continue
            else:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()

        average_seq_len /= batch_size
        average_reward /= batch_size
        print(
            f'average sequence length: {average_seq_len}, average reward: {average_reward}, best gate count: {best_gate_count}'
        )
        seq_len_log.append(average_seq_len)
        reward_log.append(average_reward)


experiment_name = "rl_a2c_" + "pos_data_init_sample"
context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../bfs_verified_simplified.json',
    no_increase=True,
)
parser = quartz.PyQASMParser(context=context)
init_dag = parser.load_qasm(filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_graph = quartz.PyGraph(context=context, dag=init_dag)

a2c(
    hidden_size=64,
    context=context,
    init_graph=init_graph,
    episodes=200000,
    batch_size=1,
    lr=2e-3,
    max_seq_len=20,
)
