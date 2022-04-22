import torch
from gnn import QGNN
import os
import datetime
import torch.nn as nn
from torch.distributions import Categorical
import quartz
import torch.functional as F
from transformers import TransfoXLConfig, TransfoXLModel
import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# num_gate_type = 29

################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.nodes = []
        self.xfers = []
        self.states = []
        self.node_logprobs = []
        self.xfer_logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.nodes[:]
        del self.xfers[:]
        del self.states[:]
        del self.node_logprobs[:]
        del self.xfer_logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# TODO: Modify the policy network
class ActorCritic(nn.Module):
    def __init__(self, num_gate_type, hidden_size, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            5, QGNN(num_gate_type, hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim), nn.Softmax())

        self.critic = nn.Sequential(
            QGNN(5, num_gate_type, hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size))

    def forward(self):
        raise NotImplementedError

    # TODO
    def act(self, g):
        dgl_g = g.to_dgl_graph().to(device)

        # Used critic network to select node
        node_qs = self.critic(dgl_g)
        node_prob = F.softmax(node_qs)
        node_dist = Categorical(node_prob)
        node = node_dist.sample()
        node_logprob = node_dist.log_prob(node)

        xfer_probs = self.actor(dgl_g)
        xfer_dist = Categorical(xfer_probs[node])
        xfer = xfer_dist.sample()
        xfer_logprob = xfer_dist.log_prob(xfer)

        # TODO: detach here?
        return node.detach(), node_logprob.detach(), xfer.detach(
        ), xfer_logprob.detach()

    # TODO: modify this to batch
    def evaluate(self, g, node, xfer):
        dgl_g = g.to_dgl_graph().to(device)

        xfer_probs = self.actor(dgl_g)
        xfer_dist = Categorical(xfer_probs)
        xfer_logprobs = xfer_dist.log_prob(xfer)
        xfer_entropy = xfer_dist.entropy()
        state_value = self.critic(g)[node]

        return xfer_logprobs, state_value, xfer_entropy


class PPO:
    def __init__(self, hidden_size, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(hidden_size, action_dim).to(device)
        self.optimizer = torch.optim.Adam([{
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

        self.policy_old = ActorCritic(hidden_size, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, graph):

        with torch.no_grad():
            node, node_logprob, xfer, xfer_logprob = self.policy_old.act(graph)

        self.buffer.states.append(graph)
        self.buffer.nodes.append(graph)
        self.buffer.xfers.append(graph)
        self.buffer.node_logprobs.append(node_logprob)
        self.buffer.xfer_logprobs.append(xfer_logprob)

        return node.item(), xfer.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # TODO: states
        old_g = self.buffer.states
        old_node = torch.squeeze(torch.stack(self.buffer.nodes,
                                             dim=0)).detach().to(device)
        old_xfer = torch.squeeze(torch.stack(self.buffer.xfers,
                                             dim=0)).detach().to(device)
        old_node_logprobs = torch.squeeze(
            torch.stack(self.buffer.node_logprobs, dim=0)).detach().to(device)
        old_xfer_logprobs = torch.squeeze(
            torch.stack(self.buffer.xfer_logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            # TODO
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            # TODO
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_xfer_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
                state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(
            torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage))


################################### Training ###################################

####### initialize environment hyperparameters ######

# TODO: Change this
experiment_name = "rl_ppo_" + ""

# max timesteps in one episode
max_ep_len = 20
episodes = int(1e5)

# log avg reward in the interval (in num timesteps)
log_freq = 2
# save model frequency (in num timesteps)
save_model_freq = int(2e4)

#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################

update_ep_step = 4  # update policy every n episodes
K_epochs = 40  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor
lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
random_seed = 0  # set random seed if required (0 = no random seed)

#####################################################

# quartz initialization

context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg'],
                               filename='../bfs_verified_simplified.json',
                               no_increase=True)
parser = quartz.PyQASMParser(context=context)
init_dag = parser.load_qasm(
    filename="barenco_tof_3_opt_path/subst_history_39.qasm")
# TODO: may need more initial graphs, from easy to hard
init_graph = quartz.PyGraph(context=context, dag=init_dag)
xfer_dim = context.num_xfers

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
log_f_name = log_dir + '/PPO_' + experiment_name + "_log_" + str(
    run_num) + ".csv"

print("current logging run number for " + experiment_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + experiment_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
    experiment_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################

############# print all hyperparameters #############

print(
    "--------------------------------------------------------------------------------------------"
)

print("running episodes : ", episodes)
print("max timesteps per episode : ", max_ep_len)

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

print("PPO update frequency : " + str(update_ep_step) + " episodes")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print(
    "--------------------------------------------------------------------------------------------"
)

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

# initialize a PPO agent
ppo_agent = PPO(64, xfer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print(
    "============================================================================================"
)

# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# logging variables
log_running_reward = 0
log_running_episodes = 0

ep = 0
i_episode = 0

# training loop
while i_episode <= episodes:

    current_ep_reward = 0

    # TODO: use get trajectory
    for t in range(1, max_ep_len + 1):

        # select action with policy
        action = ppo_agent.select_action(state)

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        current_ep_reward += reward

        # break; if the episode is over
        if done:
            break

    # update PPO agent
    if i_episode % update_ep_step == 0:
        ppo_agent.update()

    # log in logging file
    if i_episode % log_freq == 0:

        # log average reward till last episode
        log_avg_reward = log_running_reward / log_running_episodes
        log_avg_reward = round(log_avg_reward, 4)

        message = f'{i_episode}, {log_avg_reward}'
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
        print("Elapsed Time  : ",
              datetime.now().replace(microsecond=0) - start_time)
        print(
            "--------------------------------------------------------------------------------------------"
        )

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

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