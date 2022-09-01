import copy
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import wandb
from PPO import PPO
from quartz.core import PyGraph
from torch.distributions import Categorical
from tqdm import tqdm
from Trajectory import get_trajectory_batch

import quartz

wandb.init(project='ppo_t_count', mode='online')

os.environ['OMP_SCHEDULE'] = 'dynamic'

# set device to cpu or cuda
device_get_trajectory = torch.device('cuda:0')
device_update = torch.device('cuda:3')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:3')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

################################### Training ###################################

####### initialize environment hyperparameters ######

# TODO: Change this in new experiments
experiment_name = "rl_ppo_fine_tune"

# max timesteps in one trajectory
max_seq_len = 128
batch_size = 256
episodes = int(1e5)
save_model_freq = 20

################ PPO hyperparameters ################

K_epochs = 20  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.95  # discount factor
lr_graph_embedding = 3e-4  # learning rate for graph embedding network
lr_actor = 3e-4  # learning rate for actor network
lr_critic = 5e-4  # learning rate for critic network
random_seed = 0  # set random seed if required (0 = no random seed)
entropy_coefficient = 0.02
gnn_layers = 6
mini_batch_size = 256
invalid_reward = -1

#####################################################

# quartz initialization

context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg', 'x'],
    filename='../../ecc_set/t_tdg.json.ecc',
    no_increase=False,
)
# context = quartz.QuartzContext(gate_set=['h', 'cx', 'rz', 'add', 'x'],
#                                filename='../../ecc_set/nam.json.ecc',
#                                no_increase=False)
# context = quartz.QuartzContext(gate_set=['h', 'cx', 'x', 'rz', 'add'],
#                                filename='../../../Nam_complete_ECC_set.json',
#                                no_increase=False)
num_gate_type = 29
parser = quartz.PyQASMParser(context=context)
xfer_dim = context.num_xfers

global circ_info
global circ_dataset
global circ_names

# circ_names = [
#     'barenco_tof_3', 'mod5_4', 'vbe_adder_3', 'gf2^4_mult', 'qcla_com_7'
# ]
# circ_names = ['barenco_tof_3']  # , 'mod5_4']
# circ_names = ['mod5_4']  # , 'mod5_4']
# circ_names = ['qcla_mod_7']
# circ_names = ['csla_mux_3']
# circ_names = ['gf2^4_mult']
# circ_names = ['gf2^4_mult', 'gf2^4_mult_0', 'gf2^4_mult_1', 'gf2^4_mult_2']
# circ_names = ['tof_3']
# circ_names = ['rc_adder_6']
# circ_names = ['qcla_com_7']
circ_names = ['adder_8']
# circ_names = ['gf2^10_mult']
# circ_names = ["all"]

circ_dataset = {}
for circ_name in circ_names:
    circ_dataset[circ_name] = {}

    if circ_name == 'barenco_tof_3':
        # init_circ = PyGraph.from_qasm(context=context,
        #                               filename="../../near_56_rz.qasm")
        init_circ = PyGraph.from_qasm(context=context, filename="../../near_56.qasm")
    elif circ_name == "all":
        init_circ = PyGraph.from_qasm(context=context, filename="../../all.qasm")
    else:
        # init_dag = parser.load_qasm(
        #     # filename=
        #     # f"../../t_tdg_h_cx_toffoli_flip_dataset/{circ_name}_after_toffoli_flip.qasm"
        #     filename=
        #     f"../../t_tdg_h_x_cx_rm/{circ_name}.t_tdg.rotation_merging.qasm")
        # init_circ = quartz.PyGraph(context=context, dag=init_dag)
        # init_circ = PyGraph.from_qasm(
        #     context=context,
        #     filename=
        #     f"../../t_tdg_h_x_cx_rm/{circ_name}.t_tdg.rotation_merging.qasm")
        init_circ = PyGraph.from_qasm(
            context=context, filename=f"../../t_tdg_rm_circs/{circ_name}.qasm"
        )

    circ_dataset[circ_name]['init_circ'] = init_circ
    circ_dataset[circ_name]['circs'] = {}
    circ_dataset[circ_name]['circs'][init_circ.t_count] = [init_circ]
    circ_dataset[circ_name]['hash_set'] = set([init_circ.hash()])
    circ_dataset[circ_name]['num'] = 1
    circ_dataset[circ_name]['best'] = init_circ.t_count
    circ_dataset[circ_name]['best_circ'] = init_circ

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

################### save circuits ###################

global circ_directory

circ_directory = "PPO_circs"
if not os.path.exists(circ_directory):
    os.makedirs(circ_directory)

circ_directory = circ_directory + '/' + str(run_num) + '/'
if not os.path.exists(circ_directory):
    os.makedirs(circ_directory)

for circ_name in circ_names:
    circ_path = f"{circ_directory}{circ_name}_best.qasm"
    circ_dataset[circ_name]['best_circ'].to_qasm(filename=circ_path)

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
print(f"Mini batch size: {mini_batch_size}")
print(f"circuits used in training: {circ_names}")

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
    mini_batch_size,
    log_f,
    device_get_trajectory,
    device_update,
)
# ppo_agent.load('pretrained_model.pth')

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print(
    "============================================================================================"
)

# training loop
for i_episode in tqdm(range(episodes)):

    start = time.time()

    # Samplers for each kind of circuit
    samplers = {}
    for circ_name in circ_names:
        keys = list(circ_dataset[circ_name]['circs'].keys())
        values = torch.tensor(keys, dtype=torch.float)
        values = 1 / values.pow(4)
        dist = Categorical(logits=values)
        samplers[circ_name] = (keys, dist)

    sampled_init_circs = []
    # Put original circuits into list
    for circ_name in circ_names:
        circ_gate_cnt = circ_dataset[circ_name]['init_circ'].t_count
        circ = circ_dataset[circ_name]['init_circ']
        sampled_init_circs.append((circ_name, circ, circ_gate_cnt))

    # Sample the rest
    for i in range(batch_size - len(circ_names)):
        sampled_circ_name = random.choice(circ_names)
        sampled_gate_cnt_idx = samplers[sampled_circ_name][1].sample().item()
        sampled_gate_cnt = samplers[sampled_circ_name][0][sampled_gate_cnt_idx]
        sampled_circ = random.choice(
            circ_dataset[sampled_circ_name]['circs'][sampled_gate_cnt]
        )
        sampled_circ_gate_cnt = sampled_gate_cnt
        sampled_init_circs.append(
            (sampled_circ_name, sampled_circ, sampled_circ_gate_cnt)
        )

    intermediate_circs, trajectory_infos = get_trajectory_batch(
        ppo_agent, context, sampled_init_circs, max_seq_len, invalid_reward
    )

    t_0 = time.time()
    print(f'get trajectory time: {t_0 - start}')

    # update PPO agent
    ppo_agent.update_mini_batch()

    t_1 = time.time()
    print(f'update time: {t_1 - t_0}')

    for circ_name, circs in intermediate_circs.items():
        for circ, hash in zip(circs['circ'], circs['hash']):
            if hash not in circ_dataset[circ_name]['hash_set']:
                circ_dataset[circ_name]['hash_set'].add(hash)
                circ_dataset[circ_name]['num'] += 1
                if circ.t_count not in circ_dataset[circ_name]['circs']:
                    circ_dataset[circ_name]['circs'][circ.t_count] = [circ]
                else:
                    circ_dataset[circ_name]['circs'][circ.t_count].append(circ)

    # Update best
    for circ_name in trajectory_infos:
        if (
            trajectory_infos[circ_name]['best_gate_cnt']
            < circ_dataset[circ_name]['best']
        ):
            circ_dataset[circ_name]['best'] = trajectory_infos[circ_name][
                'best_gate_cnt'
            ]
            circ_dataset[circ_name]['best_circ'] = trajectory_infos[circ_name][
                'best_circ'
            ]
            circ_path = f"{circ_directory}{circ_name}_best.qasm"
            circ_dataset[circ_name]['best_circ'].to_qasm(filename=circ_path)

    # Wandb log & print & log
    for circ_name, info in trajectory_infos.items():
        info_dict = {}

        useless_keys = ['reward', 'possible_reward', 'seq_len', 'num', 'best_circ']

        message = f'circ_name: {circ_name}\n'
        for key_name, value in info.items():
            if key_name not in useless_keys:
                info_dict[f'{circ_name}_{key_name}'] = value
                message += f'{key_name}: {value}\n'

        message += f"best: {circ_dataset[circ_name]['best']}\n"
        info_dict[f'{circ_name}_best'] = circ_dataset[circ_name]['best']

        message += f"init state count: {circ_dataset[circ_name]['num']}\n"
        info_dict[f'{circ_name}_init_state_cnt'] = circ_dataset[circ_name]['num']
        message += '\n'

        wandb.log(info_dict)
        print(message)
        log_f.write(message)
        log_f.flush()

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
