import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
from gnn import QGNN
import quartz

# Constants
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
gate_type_num = 26


class ActorCritic(nn.Module):
    def __init__(self, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            QGNN(gate_type_num, hidden_size, hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1))

        self.actor = nn.Sequential(
            QGNN(gate_type_num, hidden_size, hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_outputs), nn.Flatten(start_dim=0),
            nn.Softmax(dim=0))

    def forward(self, dgl_g):
        value = self.critic(dgl_g).mean()
        probs = self.actor(dgl_g)
        dist = Categorical(probs)
        return dist, value


def get_trajectory(device, max_seq_len, num_actions, model, invalid_reward,
                   init_state, context):
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
            dist, value = model(dgl_graph)

            action = dist.sample()
            # action = dist.sample()
            node = action.cpu().item() // num_actions
            xfer = action.cpu().item() % num_actions
            # print(f'{node}, {xfer}')
            next_graph = graph.apply_xfer(
                xfer=context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node))

            if next_graph == None:
                reward = invalid_reward
                done = True
                seq_len = seq_cnt + 1
            else:
                reward = graph.gate_count - next_graph.gate_count

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            # value = value[node]

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
        next_value = invalid_reward
    else:
        next_dgl_graph = next_graph.to_dgl_graph().to(device)
        _, next_value = model(next_dgl_graph)

    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    masks = torch.tensor(masks, dtype=torch.bool).to(device)
    values = torch.tensor(values, dtype=torch.float).to(device)
    log_probs = torch.tensor(log_probs, dtype=torch.float).to(device)
    qs = compute_qs(next_value, rewards, masks)

    return qs, values, log_probs, masks, entropy, seq_len, rewards.sum().cpu(
    ).item()


def compute_qs(next_value, rewards, masks, gamma=0.99):
    R = next_value
    qs = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        qs.insert(0, R)
    return torch.tensor(qs).to(device)


def a2c(hidden_size,
        context,
        init_graph,
        episodes=20000,
        lr=1e-3,
        max_seq_len=5,
        invalid_reward=-10,
        batch_size=100):

    num_actions = context.num_xfers
    model = ActorCritic(num_actions, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: scheduler

    reward_log = []
    seq_len_log = []

    for _ in tqdm(range(episodes)):

        log_probs = torch.tensor([], dtype=torch.float).to(device)
        values = torch.tensor([], dtype=torch.float).to(device)
        masks = torch.tensor([], dtype=torch.bool).to(device)
        qs = torch.tensor([], dtype=torch.float).to(device)
        average_seq_len = 0
        average_reward = 0
        entropy = 0

        # rollout trajectory
        for i in range(batch_size):
            qs_, values_, log_probs_, masks_, entropy_, seq_len_, total_rewards_ = get_trajectory(
                device, max_seq_len, num_actions, model, invalid_reward,
                init_graph, context)

            log_probs = torch.cat((log_probs, log_probs_))
            values = torch.cat((values, values_))
            qs = torch.cat((qs, qs_))
            masks = torch.cat((masks, masks_))
            entropy += entropy_
            average_seq_len += seq_len_
            average_reward += total_rewards_

        advantage = qs - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        average_seq_len /= batch_size
        average_reward /= batch_size
        print(
            f'average sequence length: {average_seq_len}, average reward: {average_reward}'
        )
        seq_len_log.append(average_seq_len)
        reward_log.append(average_reward)


if __name__ == '__main__':
    experiment_name = "rl_a2c_" + "pos_data_init_sample"

    context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg'],
                                   filename='../bfs_verified_simplified.json')
    parser = quartz.PyQASMParser(context=context)

    init_dag = parser.load_qasm(
        filename="barenco_tof_3_opt_path/subst_history_39.qasm")
    init_graph = quartz.PyGraph(context=context, dag=init_dag)

    a2c(64, context, init_graph, batch_size=5)