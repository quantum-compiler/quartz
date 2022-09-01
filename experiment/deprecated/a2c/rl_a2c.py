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

import quartz


def masked_softmax(logits, mask):
    """
    This method will return valid probability distribution for the particular instance if its corresponding row
    in the `mask` matrix is not a zero vector. Otherwise, a uniform distribution will be returned.
    This is just a technical workaround that allows `Categorical` class usage.
    If probs doesn't sum to one there will be an exception during sampling.
    """
    # if mask is not None:
    #     # print(logits)
    #     # print(mask)
    #     probs = F.softmax(logits, dim=-1) * mask
    #     # print(probs)
    #     probs = probs + (mask.sum(dim=-1, keepdim=True)
    #                      == 0.).to(dtype=torch.float32)
    #     Z = probs.sum(dim=-1, keepdim=True)
    #     uniform = 1 / mask.shape[-1]
    #     # probs[(Z == 0.).squeeze()] = uniform
    #     # probs[(Z != 0.).squeeze()] /= Z
    #     # print(Z)
    #     # print(probs)
    #     # return probs / Z
    #     assert len(Z.shape) == 1
    #     if Z[0] != 0.:
    #         return probs / Z
    #     print('all 0')
    #     return torch.zeros_like(probs).add(uniform)
    # else:
    #     return F.softmax(logits, dim=-1)
    mask = torch.ones_like(mask, dtype=torch.bool) ^ mask
    logits[mask] -= 1.0e10
    return F.softmax(logits)


class ActorCritic(nn.Module):
    def __init__(self, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            QGNN(gate_type_num, hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor = nn.Sequential(
            QGNN(gate_type_num, hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(self, dgl_g):
        value = self.critic(dgl_g)
        # value = value.mean()
        logits = self.actor(dgl_g)
        # print(torch.max(probs))
        # dist = Categorical(probs)
        return logits, value


def get_trajectory(
    device,
    max_seq_len,
    num_actions,
    model,
    invalid_reward,
    init_state,
    context,
    best_gate_count,
):
    log_probs = []
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
            dgl_graph = graph.to_dgl_graph().to(device)
            logits, value = model(dgl_graph)
            node = random.randint(0, dgl_graph.num_nodes() - 1)
            available_xfers = graph.available_xfers(
                context=context, node=graph.get_node_from_id(id=node)
            )
            mask = torch.zeros((num_actions), dtype=torch.bool).to(device)
            mask[available_xfers] = True
            probs = masked_softmax(logits[node], mask)
            try:
                dist = Categorical(probs)
            except ValueError as e:
                print(logits)
                print(values)
                for param in model.critic.parameters():
                    print('critic')
                    print(param.shape)
                    print(param)
                    print(f'grad = {param.grad.data}')
                    print(param.grad.data.shape)
                for param in model.actor.parameters():
                    print('actor')
                    print(param)
                    print(param.shape)
                    print(f'grad = {param.grad.data}')
                    print(sum(torch.isnan(param.grad.data)))
                    print(param.grad.data.shape)

            xfer = dist.sample()
            # action = dist.sample()
            # node = action.cpu().item() // num_actions
            # xfer = action.cpu().item() % num_actions
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
                if next_gate_cnt < best_gate_count:
                    best_gate_count = next_gate_cnt
            seq_len = seq_cnt + 1

            log_prob = dist.log_prob(xfer)
            entropy += dist.entropy().mean()
            reward = torch.tensor(reward)
            # value = value[node][0]
            value = value[node].squeeze()
            # print(f'values: {value}')

        else:
            # log_prob = torch.tensor((0)).to(device)
            # value = torch.tensor((0)).to(device)
            # reward = 0
            break

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        masks.append(1 - done)

        graph = next_graph

    # print('end')

    if next_graph == None:
        # next_value = invalid_reward
        next_value = 0
    else:
        next_dgl_graph = next_graph.to_dgl_graph().to(device)
        _, next_value = model(next_dgl_graph)
        next_value = next_value.mean()

    rewards = torch.stack(rewards).to(device)
    # print(rewards)
    # print(values)
    values = torch.stack(values).to(device)
    # print(values)
    log_probs = torch.stack(log_probs).to(device)
    masks = torch.tensor(masks).to(device)
    qs = compute_qs(next_value, rewards, masks)

    return (
        qs,
        values,
        log_probs,
        masks,
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
    invalid_reward=-10,
    batch_size=100,
):

    num_actions = context.num_xfers
    best_gate_count = init_graph.gate_count
    model = ActorCritic(num_actions, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: scheduler

    reward_log = []
    seq_len_log = []

    for _ in tqdm(range(episodes), miniters=100):

        log_probs = torch.tensor([], dtype=torch.float).to(device)
        values = torch.tensor([], dtype=torch.float).to(device)
        masks = torch.tensor([], dtype=torch.bool).to(device)
        qs = torch.tensor([], dtype=torch.float).to(device)
        average_seq_len = 0
        average_reward = 0
        entropy = 0

        # rollout trajectory
        for i in range(batch_size):
            (
                qs_,
                values_,
                log_probs_,
                masks_,
                entropy_,
                seq_len_,
                total_rewards_,
                best_gate_count,
            ) = get_trajectory(
                device,
                max_seq_len,
                num_actions,
                model,
                invalid_reward,
                init_graph,
                context,
                best_gate_count,
            )

            log_probs = torch.cat((log_probs, log_probs_))
            values = torch.cat((values, values_))
            # print(values)
            qs = torch.cat((qs, qs_))
            masks = torch.cat((masks, masks_))
            entropy += entropy_
            average_seq_len += seq_len_
            average_reward += total_rewards_

        print(f'qs: {qs}')
        print(f'values: {values}')
        advantage = qs - values
        print(f'advantage: {advantage}')
        print(f'log_probs: {log_probs}')

        actor_loss = -(log_probs * advantage.clone().detach()).mean()
        print(f'actor_loss: {actor_loss.item()}')
        critic_loss = advantage.pow(2).mean()
        print(f'critic_loss: {critic_loss.item()}')

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        print(f'entropy: {entropy.item()}')
        print(f'loss: {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        average_seq_len /= batch_size
        average_reward /= batch_size
        print(
            f'average sequence length: {average_seq_len}, average reward: {average_reward}, best gate count: {best_gate_count}'
        )
        seq_len_log.append(average_seq_len)
        reward_log.append(average_reward)


device = torch.device('cpu')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

# Constants
gate_type_num = 26

experiment_name = "rl_a2c_" + "pos_data_init_sample"

context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'], filename='../bfs_verified_simplified.json'
)
parser = quartz.PyQASMParser(context=context)

init_dag = parser.load_qasm(filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_graph = quartz.PyGraph(context=context, dag=init_dag)

a2c(64, context, init_graph, episodes=200000, batch_size=1, lr=1e-3, max_seq_len=20)
