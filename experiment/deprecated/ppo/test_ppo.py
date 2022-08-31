import os
import sys
import time
from datetime import datetime

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn import QGNN
from qiskit import QuantumCircuit
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
        self.next_graphs = []
        self.next_nodes = []
        self.xfer_logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.__init__()


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_gate_type,
        graph_embed_size,
        actor_hidden_size,
        critic_hidden_size,
        action_dim,
    ):
        super(ActorCritic, self).__init__()

        self.graph_embedding = QGNN(
            6, num_gate_type, graph_embed_size, graph_embed_size
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

    def forward(self):
        raise NotImplementedError

    def act(self, context, g):
        dgl_g = g.to_dgl_graph().to(device)

        # Used critic network to select node
        graph_embed = self.graph_embedding(dgl_g)

        node_vs = self.critic(graph_embed).squeeze()
        node_prob = F.softmax(node_vs, dim=-1)
        node_dist = Categorical(node_prob)
        node = node_dist.sample()

        mask = torch.zeros((context.num_xfers), dtype=torch.bool).to(device)
        available_xfers = g.available_xfers(
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
        return node.detach(), xfer.detach(), xfer_logprob.detach()


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

        self.policy = ActorCritic(
            num_gate_type,
            graph_embed_size,
            actor_hidden_size,
            critic_hidden_size,
            action_dim,
        ).to(device)

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
            node, xfer, xfer_logprob = self.policy_old.act(self.context, graph)

        self.buffer.graphs.append(graph)
        self.buffer.nodes.append(node)
        self.buffer.xfers.append(xfer)
        self.buffer.xfer_logprobs.append(xfer_logprob)

        return node.item(), xfer.item()

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

################## get trajectory ##################


def get_trajectory(ppo_agent, init_state, max_seq_len, invalid_reward, log_f):
    graph = init_state
    done = False
    nop_stop = False
    trajectory_reward = 0
    trajectory_len = 0
    trajectory_best_gate_count = init_state.gate_count

    for t in range(max_seq_len):
        if not done:
            node, xfer = ppo_agent.select_action(graph)
            next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(
                xfer=context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node),
            )

            if next_graph == None:
                reward = invalid_reward
                done = True
                next_graph = graph
            elif context.get_xfer_from_id(id=xfer).is_nop:
                reward = 0
                done = True
                nop_stop = True
                next_nodes = [node]
            else:
                reward = (graph.gate_count - next_graph.gate_count) * 3

            message = f'step: {t}, node: {node}, xfer: {xfer}, reward: {reward}'
            if reward > 0:
                message += ', Reduced!'
            elif reward < 0:
                message += ', Increased.'
            message += '\n'
            log_f.write(message)

            trajectory_reward += reward
            reward = torch.tensor(reward, dtype=torch.float)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(torch.tensor(done, dtype=torch.bool))
            ppo_agent.buffer.next_graphs.append(next_graph)
            ppo_agent.buffer.next_nodes.append(next_nodes)
            graph = next_graph
        else:
            trajectory_len = t
            break

    trajectory_best_gate_count = min(graph.gate_count, trajectory_best_gate_count)

    return trajectory_reward, trajectory_best_gate_count, trajectory_len, graph


if __name__ == '__main__':
    ####### initialize environment hyperparameters ######

    experiment_name = "rl_ppo_test"

    # max timesteps in one trajectory
    max_seq_len = 1000
    batch_size = 256
    episodes = 1

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

    # logging file
    log_f = open(log_f_name, "w+")

    ################ PPO hyperparameters ################

    K_epochs = 20  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.9  # discount factor
    lr_graph_embedding = 3e-4  # learning rate for graph embedding network
    lr_actor = 3e-4  # learning rate for actor network
    lr_critic = 1e-3  # learning rate for critic network
    random_seed = 0  # set random seed if required (0 = no random seed)

    invalid_reward = -1

    #####################################################

    # quartz initialization
    assert len(sys.argv) > 1
    qasm_fn = sys.argv[1]

    print(
        "--------------------------------------------------------------------------------------------"
    )

    print(qasm_fn)
    log_f.write(f'{qasm_fn}\n')

    context = quartz.QuartzContext(
        gate_set=['h', 'cx', 't', 'tdg', 'x'],
        filename='../../bfs_verified_simplified.json',
        no_increase=False,
    )
    num_gate_type = 29
    parser = quartz.PyQASMParser(context=context)
    # init_dag = parser.load_qasm(
    #     filename="barenco_tof_3_opt_path/subst_history_39.qasm")
    # init_dag = parser.load_qasm(
    #     filename="../barenco_tof_3_opt_path/subst_history_39.qasm")
    # init_dag = parser.load_qasm(
    #     filename=
    #     "../t_tdg_h_cx_toffoli_flip_dataset/barenco_tof_4_after_toffoli_flip.qasm"
    # )
    init_dag = parser.load_qasm(filename=qasm_fn)
    init_graph = quartz.PyGraph(context=context, dag=init_dag)
    xfer_dim = context.num_xfers
    best_gate_cnt = init_graph.gate_count
    best_circ = init_graph

    ############# print all hyperparameters #############

    print(
        "--------------------------------------------------------------------------------------------"
    )

    print("max timesteps per trajectory : ", max_seq_len)
    print("batch size: ", batch_size)

    print(
        "--------------------------------------------------------------------------------------------"
    )

    print("xfer dimension : ", xfer_dim)

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

    # logging variables
    log_running_reward = 0
    log_running_episodes = 0

    # initialize a PPO agent
    ppo_agent = PPO(
        num_gate_type,
        context,
        128,
        256,
        128,
        xfer_dim,
        lr_graph_embedding,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        log_file_handle=log_f,
    )
    ppo_agent.load(
        'PPO_preTrained/rl_ppo_local_multi_init_states_with_increase/PPO_rl_ppo_local_multi_init_states_with_increase_0_6.pth'
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print(
        "============================================================================================"
    )

    ep = 0
    i_episode = 0

    current_ep_reward = 0
    ep_best_gate_cnt = init_graph.gate_count
    ep_seq_len = 0

    tqdm_bar = tqdm(range(batch_size))
    for i in tqdm_bar:
        log_f.write(f'trajectory {i}\n')
        t_reward, t_best_gate_cnt, t_seq_len, last_graph = get_trajectory(
            ppo_agent, init_graph, max_seq_len, invalid_reward, log_f
        )

        current_ep_reward += t_reward
        if best_gate_cnt > t_best_gate_cnt:
            best_gate_cnt = t_best_gate_cnt
            best_circ = last_graph
            qiskit_circ = QuantumCircuit.from_qasm_str(best_circ.to_qasm_str())
            qiskit_circ.draw(output='mpl').savefig('best.png')
        ep_best_gate_cnt = min(ep_best_gate_cnt, t_best_gate_cnt)
        ep_seq_len += t_seq_len
        tqdm_bar.set_description(f'best: {best_gate_cnt}')

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    # log average reward till last episode
    log_avg_reward = log_running_reward / log_running_episodes / batch_size
    log_avg_reward = round(log_avg_reward, 4)
    log_avg_seq_len = ep_seq_len / batch_size

    message = f'avg reward: {log_avg_reward}\tavg seq len: {log_avg_seq_len}\tbest of ep: {ep_best_gate_cnt}\tbest: {best_gate_cnt} '
    log_f.write(message + '\n')
    print(message)
    log_f.flush()

    log_running_reward = 0
    log_running_episodes = 0

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
