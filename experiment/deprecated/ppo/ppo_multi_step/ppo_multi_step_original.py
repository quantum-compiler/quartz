import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import wandb
from PPO import PPO
from prompt_toolkit import PromptSession
from torch.distributions import Categorical
from tqdm import tqdm
from Utils import get_trajectory

import quartz

wandb.init(project='ppo_multi_step')
os.environ['OMP_SCHEDULE'] = 'dynamic'

# set device to cpu or cuda
device = torch.device('cpu')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:3')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

################################### Training ###################################

####### initialize environment hyperparameters ######

# TODO: Change this in new experiments
experiment_name = "ppo_multi_sep"

# max timesteps in one trajectory
max_seq_len = 300
batch_size = 256
max_init_states = 255
episodes = int(1e5)

# save model frequency (in num timesteps)
save_model_freq = 20

################ PPO hyperparameters ################

K_epochs = 40  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.995  # discount factor
lr_graph_embedding = 3e-4  # learning rate for graph embedding network
lr_actor = 3e-4  # learning rate for actor network
lr_critic = 5e-4  # learning rate for critic network
random_seed = 0  # set random seed if required (0 = no random seed)
entropy_coefficient = 0.02
gnn_layers = 8

invalid_reward = -1

#####################################################

# quartz initialization

context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../../../bfs_verified_simplified.json',
    no_increase=False,
)
num_gate_type = 29
parser = quartz.PyQASMParser(context=context)
# init_dag = parser.load_qasm(
#     filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_dag = parser.load_qasm(filename="../../near_56.qasm")
init_circ = quartz.PyGraph(context=context, dag=init_dag)
xfer_dim = context.num_xfers

global init_graphs
global new_init_states
global new_init_state_hash_set
global ground_truth_minimum
global circ_dataset

# new_init = {}
# new_init['graphs'] = []
# new_init['values'] = []
# new_init['hash_set'] = set([])
# new_init['num'] = 0
# new_init['cnt_statics'] = {}
# for i in range(38, 58):
#     new_init['cnt_statics'][str(i)] = 0
new_init = {}
new_init['graphs'] = {}
new_init['graphs'][init_circ.gate_count] = [init_circ]
new_init['hash_set'] = set([])
new_init['cnt_statics'] = {}
new_init['num'] = 1
for i in range(38, 58):
    new_init['cnt_statics'][str(i)] = 0
best_gate_cnt = init_circ.gate_count
ground_truth_minimum = 38  # TODO: change this if use other circuits

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

################### checkpointing ###################

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + experiment_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
    experiment_name, random_seed, run_num
)
print("save checkpoint path : " + checkpoint_path)

############# print all hyperparameters #############

print(
    "--------------------------------------------------------------------------------------------"
)

print("running episodes : ", episodes)
print("max timesteps per trajectory : ", max_seq_len)
print("batch size: ", batch_size)
print("model saving frequency : " + str(save_model_freq) + " episodes")
print("xfer dimension : ", xfer_dim)
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("optimizer learning rate graph embeddng: ", lr_graph_embedding)
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
print(f"entropy coefficient : {entropy_coefficient}")
print(f"GNN layers: {gnn_layers}")
# print(f"circuits used in training: {circ_names}")

if random_seed:
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

# initialize a PPO agent
ppo_agent = PPO(
    num_gate_type,
    context,
    gnn_layers,
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
    entropy_coefficient,
    log_f,
    device,
)
# ppo_agent.load(
#     'PPO_preTrained/rl_ppo_local_multi_init_states_include_increase/PPO_rl_ppo_local_multi_init_states_include_increase_0_0.pth'
# )

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print(
    "============================================================================================"
)

# training loop
for i_episode in tqdm(range(episodes)):

    start = time.time()

    current_ep_reward = 0
    ep_best_gate_cnt = init_circ.gate_count
    ep_seq_len = 0
    ep_best_reward = 0
    total_possible_reward = 0
    ep_batch_size = 0

    ep_init_graphs = [init_circ]
    keys = list(new_init['graphs'].keys())
    keys = torch.tensor(keys)
    key_logits = 1 / keys.pow(2)
    dist = Categorical(logits=key_logits)
    for i in range(max_init_states - 1):
        sampled_gate_num = keys[dist.sample()].item()
        # print(sampled_gate_num)
        ep_init_graphs.append(random.choice(new_init['graphs'][sampled_gate_num]))

    # if new_init['num'] < max_init_states - 1:
    #     ep_init_graphs = [init_circ] + new_init['graphs']
    # else:
    #     logits = torch.tensor(new_init['values'])
    #     dist = Categorical(logits=logits)
    #     sample_idxs = dist.sample(torch.Size([max_init_states - 1]))
    #     ep_init_graphs = [init_circ]
    #     for idx in sample_idxs:
    #         ep_init_graphs.append(new_init["graphs"][idx])

    for i in range(batch_size):

        init_graph = ep_init_graphs[i % len(ep_init_graphs)]
        total_possible_reward += (init_graph.gate_count - ground_truth_minimum) * 4

        t_reward, t_best_gate_cnt, t_seq_len, intermediate_graphs = get_trajectory(
            ppo_agent, context, init_graph, max_seq_len, invalid_reward
        )

        current_ep_reward += t_reward
        best_gate_cnt = min(best_gate_cnt, t_best_gate_cnt)
        ep_best_gate_cnt = min(ep_best_gate_cnt, t_best_gate_cnt)
        ep_best_reward = max(ep_best_reward, t_reward)
        ep_seq_len += t_seq_len

        # Sample from intermediate_graphs
        # for g in intermediate_graphs:
        #     g_hash = g.hash()
        #     if g_hash not in new_init['hash_set']:
        #         if random.random() < 1:
        #             new_init['hash_set'].add(g_hash)
        #             new_init['graphs'].append(g)
        #             new_init['values'].append(1 / math.pow(g.gate_count, 2))
        #             new_init['num'] += 1
        #             new_init['cnt_statics'][str(g.gate_count)] += 1

        for g in intermediate_graphs:
            g_hash = g.hash()
            if g_hash not in new_init['hash_set']:
                new_init['hash_set'].add(g_hash)
                if g.gate_count not in new_init['graphs']:
                    new_init['graphs'][g.gate_count] = [g]
                else:
                    new_init['graphs'][g.gate_count].append(g)

                new_init['cnt_statics'][str(g.gate_count)] += 1
                new_init['num'] += 1

    if ep_batch_size == 0:
        ep_batch_size = batch_size

    torch.cuda.empty_cache()

    t_0 = time.time()
    print(f'get trajectory time: {t_0 - start}')

    # update PPO agent
    ppo_agent.update()

    t_1 = time.time()
    print(f'update time: {t_1 - t_0}')

    reward_realization_rate = current_ep_reward / total_possible_reward
    avg_reward = current_ep_reward / ep_batch_size
    avg_reward = round(avg_reward, 4)
    avg_seq_len = ep_seq_len / ep_batch_size
    new_init_num = new_init['num']

    message = f'ep: {i_episode}\trealize: {reward_realization_rate * 100:.2f}%\tavg_r: {avg_reward:.4f}\tbest_r: {ep_best_reward}\tavg_seq_len: {avg_seq_len:.2f}\tep_best_cnt: {ep_best_gate_cnt}\tbest_cnt: {best_gate_cnt}\tinit_state_num: {new_init_num}'
    log_f.write(message + '\n')
    print(message)
    log_f.flush()

    wandb.log(
        {
            'episode': i_episode,
            'batch_size': ep_batch_size,
            'rewrad_realization_rate': reward_realization_rate,
            'avg_reward': avg_reward,
            'avg_seq_len': avg_seq_len,
            'ep_best': ep_best_gate_cnt,
            'ep_best_reward': ep_best_reward,
            'best': best_gate_cnt,
            'sampled_new_state_num': new_init_num,
        }
    )

    wandb.log(new_init['cnt_statics'])

    # save model weights
    if i_episode % save_model_freq == 0 and i_episode != 0:
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
