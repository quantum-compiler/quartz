import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import logProcesses

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn import QGNN
from torch.distributions import Categorical
from tqdm import tqdm

import quartz

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# num_gate_type = 29

################################ Masked Softmax ################################


def masked_softmax(logits, mask):
    mask = torch.ones_like(mask, dtype=torch.bool) ^ mask
    logits[mask] -= 1.0e10
    return F.softmax(logits, dim=-1)


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.graphs = []
        self.nodes = []
        self.xfers = []
        self.node_logprobs = []
        self.xfer_logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.__init__()


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        gate_type_num = 29

        self.q_net = nn.Sequential(
            QGNN(6, gate_type_num, 256, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, dgl_g):
        return self.q_net(dgl_g)


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_gate_type,
        graph_embed_size,
        actor_hidden_size,
        critic_hidden_size,
        action_dim,
        critic_ckpt=None,
    ):
        super(ActorCritic, self).__init__()

        self.graph_embedding = QGNN(
            6, num_gate_type, graph_embed_size, graph_embed_size
        )
        self.actor_node = nn.Sequential(
            nn.Linear(graph_embed_size, graph_embed_size // 2),
            nn.ReLU(),
            nn.Linear(graph_embed_size // 2, 1),
        )

        self.actor_xfer = nn.Sequential(
            nn.Linear(graph_embed_size, actor_hidden_size),
            nn.ReLU(),
            nn.Linear(actor_hidden_size, action_dim),
        )

        self.critic = Critic()
        if critic_ckpt is not None:
            ckpt = torch.load(critic_ckpt)
            self.critic.load_state_dict(ckpt['state_dict'])

    def forward(self):
        raise NotImplementedError

    def act(self, context, g):
        dgl_g = g.to_dgl_graph().to(device)

        # Used critic network to select node
        graph_embed = self.graph_embedding(dgl_g)

        node_logits = self.actor_node(graph_embed).squeeze()
        node_prob = F.softmax(node_logits, dim=-1)
        node_dist = Categorical(node_prob)
        node = node_dist.sample()
        node_logprob = node_dist.log_prob(node)

        mask = torch.zeros((context.num_xfers), dtype=torch.bool).to(device)
        available_xfers = g.available_xfers(
            context=context, node=g.get_node_from_id(id=node)
        )
        mask[available_xfers] = True
        xfer_logits = self.actor_xfer(graph_embed)
        xfer_probs = masked_softmax(xfer_logits[node], mask)
        xfer_dist = Categorical(xfer_probs)
        xfer = xfer_dist.sample()
        xfer_logprob = xfer_dist.log_prob(xfer)

        # Detach here because we use old policy to select actions
        return (
            node.detach(),
            node_logprob.detach(),
            xfer.detach(),
            xfer_logprob.detach(),
        )

    # TODO
    def evaluate(self, gs, nodes, xfers, is_terminals):
        dgl_gs = [g.to_dgl_graph() for g in gs]
        batched_dgl_gs = dgl.batch(dgl_gs).to(device)
        batched_graph_embeds = self.graph_embedding(batched_dgl_gs)
        batched_node_logits = self.actor_node(batched_graph_embeds).squeeze()
        batched_xfer_logits = self.actor_xfer(batched_graph_embeds)
        # batched_node_vs = self.critic(batched_graph_embeds).squeeze()
        batched_node_vs = self.critic(batched_dgl_gs).squeeze()

        # Split batched tensors into lists
        nodes_nums = batched_dgl_gs.batch_num_nodes().tolist()
        node_logits_list = torch.split(batched_node_logits, nodes_nums)
        xfer_logits_list = torch.split(batched_xfer_logits, nodes_nums)
        node_vs_list = torch.split(batched_node_vs, nodes_nums)

        values = []
        node_logprobs = []
        node_entropy = 0
        xfer_logprobs = []
        xfer_entropy = 0

        def p(i):
            value = node_vs_list[i].max()
            node_logits = node_logits_list[i]
            node_probs = F.softmax(node_logits, dim=-1)
            node_dist = Categorical(node_probs)
            node_logprob = node_dist.log_prob(nodes[i])
            node_entropy = node_dist.entropy()

            mask = torch.zeros((context.num_xfers), dtype=torch.bool).to(device)
            available_xfers = gs[i].available_xfers(
                context=context, node=gs[i].get_node_from_id(id=nodes[i])
            )
            mask[available_xfers] = True
            xfer_probs = masked_softmax(xfer_logits_list[i][nodes[i]].clone(), mask)
            xfer_dist = Categorical(xfer_probs)
            xfer_logprob = xfer_dist.log_prob(xfers[i])
            xfer_entropy = xfer_dist.entropy()
            return value, node_logprob, node_entropy, xfer_logprob, xfer_entropy

        results = map(p, list(range(batched_dgl_gs.batch_size)))
        values = []
        node_logprobs = []
        node_entropy = 0
        xfer_logprobs = []
        xfer_entropy = 0
        for r in results:
            values.append(r[0])
            node_logprobs.append(r[1])
            node_entropy += r[2]
            xfer_logprobs.append(r[3])
            xfer_entropy += r[4]

        values = torch.stack(values)
        node_logprobs = torch.stack(node_logprobs)
        xfer_logprobs = torch.stack(xfer_logprobs)

        return values, node_logprobs, node_entropy, xfer_logprobs, xfer_entropy


class PPO:
    def __init__(
        self,
        num_gate_type,
        context,
        graph_embed_size,
        actor_hidden_size,
        critic_hidden_size,
        action_dim,
        lr_graph_embedding,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        log_file_handle,
    ):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        # self.policy = ActorCritic(
        #     num_gate_type,
        #     graph_embed_size,
        #     actor_hidden_size,
        #     critic_hidden_size,
        #     action_dim,
        #     critic_ckpt=
        #     "pretrain/outputs/2022-05-08/07-00-53/epoch=50-val_loss=0.07-best.ckpt"
        # ).to(device)

        self.policy = ActorCritic(
            num_gate_type,
            graph_embed_size,
            actor_hidden_size,
            critic_hidden_size,
            action_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            [
                {
                    'params': self.policy.graph_embedding.parameters(),
                    'lr': lr_graph_embedding,
                },
                {'params': self.policy.actor_node.parameters(), 'lr': lr_actor},
                {'params': self.policy.actor_xfer.parameters(), 'lr': lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            num_gate_type,
            graph_embed_size,
            actor_hidden_size,
            critic_hidden_size,
            action_dim,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.context = context
        self.log_file_handle = log_file_handle

    def select_action(self, graph):
        # Use the old policy network to select an action
        # No gradient needed
        with torch.no_grad():
            node, node_logprob, xfer, xfer_logprob = self.policy_old.act(
                self.context, graph
            )

        self.buffer.graphs.append(graph)
        self.buffer.nodes.append(node)
        self.buffer.xfers.append(xfer)
        self.buffer.node_logprobs.append(node_logprob)
        self.buffer.xfer_logprobs.append(xfer_logprob)

        return node.item(), xfer.item()

    def update(self):

        # TODO
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # TODO: states

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            # Entropy is not needed when using old policy
            # But needed in current policy
            (
                values,
                node_logprobs,
                node_entropy,
                xfer_logprobs,
                xfer_entropy,
            ) = self.policy.evaluate(
                self.buffer.graphs,
                self.buffer.nodes,
                self.buffer.xfers,
                self.buffer.is_terminals,
            )

            old_node_logprobs = (
                torch.squeeze(torch.stack(self.buffer.node_logprobs, dim=0))
                .detach()
                .to(device)
            )
            old_xfer_logprobs = (
                torch.squeeze(torch.stack(self.buffer.xfer_logprobs, dim=0))
                .detach()
                .to(device)
            )

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(
                node_logprobs
                + xfer_logprobs
                - old_node_logprobs.detach()
                - old_xfer_logprobs.detach()
            )

            # Finding Surrogate Loss
            rewards = torch.stack(self.buffer.rewards).to(device)

            # TODO
            advantages = rewards - values
            surr1 = ratios * advantages.clone().detach()
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages.clone().detach()
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = advantages.pow(2).mean()

            # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
            #     state_values, rewards) - 0.01 * (node_entropy + xfer_entropy)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * (node_entropy + xfer_entropy)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            self.log_file_handle.write(f"epoch: {_}\n")
            for i in range(len(self.buffer.graphs)):
                message = f"node: {self.buffer.nodes[i]}, xfer: {self.buffer.xfers[i]}, reward: {self.buffer.rewards[i]}, value: {values[i]:.3f}"
                if self.buffer.rewards[i] > 0:
                    message += ", Reduced!!!"
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

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


################################### Training ###################################

####### initialize environment hyperparameters ######

# TODO: Change this
experiment_name = "rl_ppo_" + "max_pooling"

# max timesteps in one trajectory
max_seq_len = 100
batch_size = 100
episodes = int(1e5)

# log avg reward in the interval (in num timesteps)
log_freq = 1
# save model frequency (in num timesteps)
save_model_freq = int(2e4)

#####################################################

################ PPO hyperparameters ################

K_epochs = 10  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.9  # discount factor
lr_graph_embedding = 3e-4
lr_actor = 3e-4  # learning rate for actor network
lr_critic = 1e-3  # learning rate for critic network
random_seed = 0  # set random seed if required (0 = no random seed)

invalid_reward = -1

#####################################################

# quartz initialization

context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../../bfs_verified_simplified.json',
    no_increase=True,
    include_nop=False,
)
num_gate_type = 29
parser = quartz.PyQASMParser(context=context)
# init_dag = parser.load_qasm(
#     filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_dag = parser.load_qasm(filename="../near_56.qasm")
# TODO: may need more initial graphs, from easy to hard
init_graph = quartz.PyGraph(context=context, dag=init_dag)
xfer_dim = context.num_xfers
global init_graphs
global init_graph_hashes
init_graphs = [init_graph]
init_graph_hashes = set([init_graph.hash()])
best_gate_count = init_graph.gate_count

###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = log_dir + '/' + experiment_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + experiment_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + experiment_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = (
    0  #### change this to prevent overwriting weights in same env_name folder
)

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + experiment_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
    experiment_name, random_seed, run_num_pretrained
)
print("save checkpoint path : " + checkpoint_path)

################## get trajectory ##################


def get_trajectory(ppo_agent, init_state, max_seq_len, invalid_reward):
    graph = random.choice(init_graphs)
    trajectory_best_gate_count = init_state.gate_count
    done = False
    nop_stop = False
    invalid_stop = False
    improve_stop = False
    trajectory_reward = 0

    for t in range(max_seq_len):
        if not done:
            node, xfer = ppo_agent.select_action(graph)
            next_graph = graph.apply_xfer(
                xfer=context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node),
            )

            if next_graph == None:
                reward = invalid_reward
                done = True
                invalid_stop = True
                next_graph = graph
            elif context.get_xfer_from_id(id=xfer).is_nop:
                reward = 0
                done = True
                nop_stop = True
            else:
                reward = graph.gate_count - next_graph.gate_count
                if reward > 0:
                    done = True
                    improve_stop = True
                    if next_graph.hash() not in init_graph_hashes:
                        init_graph_hashes.add(next_graph.hash())
                        init_graphs.append(next_graph)

            trajectory_reward += reward
            reward = torch.tensor(reward, dtype=torch.float)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(torch.tensor(done, dtype=torch.bool))
            graph = next_graph

            if improve_stop:
                done = False
                improve_stop = False
        else:
            break

    if graph.gate_count < trajectory_best_gate_count:
        trajectory_best_gate_count = graph.gate_count

    return trajectory_reward, trajectory_best_gate_count


############# print all hyperparameters #############

print(
    "--------------------------------------------------------------------------------------------"
)

print("running episodes : ", episodes)
print("max timesteps per trajectory : ", max_seq_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log and print frequency : " + str(log_freq) + " timesteps")

print(
    "--------------------------------------------------------------------------------------------"
)

print("xfer dimension : ", xfer_dim)

print(
    "--------------------------------------------------------------------------------------------"
)

print("Initializing a discrete action space policy")

print(
    "--------------------------------------------------------------------------------------------"
)

print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print(
    "--------------------------------------------------------------------------------------------"
)

print("optimizer learning rate graph embedding network : ", lr_graph_embedding)
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

#####################################################

print(
    "============================================================================================"
)

################# training procedure ################

# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# logging variables
log_running_reward = 0
log_running_episodes = 0

# initialize a PPO agent
ppo_agent = PPO(
    num_gate_type,
    context,
    256,
    256,
    32,  # not used
    xfer_dim,
    lr_graph_embedding,
    lr_actor,
    lr_critic,
    gamma,
    K_epochs,
    eps_clip,
    log_file_handle=log_f,
)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print(
    "============================================================================================"
)

ep = 0
i_episode = 0

# training loop
for i_episode in tqdm(range(episodes)):

    current_ep_reward = 0
    ep_best_gate_count = init_graph.gate_count

    for i in range(batch_size):
        trajectory_reward, trajectory_best_gate_count = get_trajectory(
            ppo_agent, init_graph, max_seq_len, invalid_reward
        )
        current_ep_reward += trajectory_reward
        best_gate_count = min(best_gate_count, trajectory_best_gate_count)
        ep_best_gate_count = min(ep_best_gate_count, trajectory_best_gate_count)

    # update PPO agent
    ppo_agent.update()

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    # log in logging file
    if i_episode % log_freq == 0:

        # log average reward till last episode
        log_avg_reward = log_running_reward / log_running_episodes / batch_size
        log_avg_reward = round(log_avg_reward, 4)

        message = f'episode: {i_episode}\taverage reward: {log_avg_reward}\tbest of episode: {ep_best_gate_count}\tbest: {best_gate_count} '
        log_f.write(message + '\n')
        print(message)
        log_f.flush()

        log_running_reward = 0
        log_running_episodes = 0

    # save model weights
    if i_episode % save_model_freq == 0:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("saving model at : " + checkpoint_path)
        ppo_agent.save(checkpoint_path)
        print("model saved")
        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print(
            "--------------------------------------------------------------------------------------------"
        )

log_f.close()

# print total training time
print(
    "============================================================================================"
)
end_time = datetime.now().replace(microsecond=0)
print("Started training at: ", start_time)
print("Finished training at: ", end_time)
print("Total training time  : ", end_time - start_time)
print(
    "============================================================================================"
)
