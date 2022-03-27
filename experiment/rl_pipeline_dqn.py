import quartz
import matplotlib.pyplot as plt
import rl_dqn
import json

experiment_name = "rl_dqn_" + "pos_data_init_sample_"

quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../bfs_verified_simplified.json')
parser = quartz.PyQASMParser(context=quartz_context)

init_dag = parser.load_qasm(
    filename="barenco_tof_3_opt_path/subst_history_39.qasm")
init_graph = quartz.PyGraph(context=quartz_context, dag=init_dag)
init_dgl_graph = init_graph.to_dgl_graph()

# Get valid xfer dict
# all_nodes = init_graph.all_nodes()
# i = 0
# valid_xfer_dict = {}
# for node in all_nodes:
#     valid_xfer_dict[i] = init_graph.available_xfers(context=quartz_context,
#                                                     node=node)
#     print(f'{i}: {valid_xfer_dict[i]}')
#     i += 1

# with open('valid_xfer_dict.json', 'w') as f:
#     json.dump(valid_xfer_dict, f)

with open('valid_xfer_dict.json', 'r') as f:
    valid_xfer_dict = json.load(f)

# RL training
seq_lens, correct_cnts, rewards = rl_dqn.train(
    lr=5e-3,
    gamma=0.999,
    replay_times=20,
    a_size=quartz_context.num_xfers,
    episodes=1000,
    epsilon=0.5,
    epsilon_decay=0.0003,
    train_epoch=10,
    max_seq_len=30,
    batch_size=20,
    context=quartz_context,
    init_graph=init_graph,
    target_update_interval=5,
    log_fn=f"log/{experiment_name}_log.txt",
    valid_xfer_dict=valid_xfer_dict,
    use_cuda=True,
    pos_data_init=True,
    pos_data_sampling=True,
    pos_data_sampling_rate=0.1)

fig, ax = plt.subplots()
ax.plot(seq_lens)
plt.title("sequence length - training epochs")
plt.savefig(f'figures/{experiment_name}_seqlen.png')

fig, ax = plt.subplots()
ax.plot(correct_cnts)

fig, ax = plt.subplots()
ax.plot(correct_cnts)
plt.title("correct counts - training epochs")
plt.savefig(f'figures/{experiment_name}_corrcnt.png')

fig, ax = plt.subplots()
ax.plot(rewards)
plt.title("rewards - training epochs")
plt.savefig(f'figures/{experiment_name}_rewards.png')


def find_number(fn, n):
    with open(fn, 'r') as f:
        for l in f:
            if l[:2] == str(n):
                print(f"{n} found!")
                return
    print(f"{n} not found!")


find_number(f"log/{experiment_name}_log.txt", 56)
