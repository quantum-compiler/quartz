import copy
import json
import random
from collections import deque

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pos_data import PosRewardData
from tqdm import tqdm

import quartz

# Constant
gate_type_num = 26

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class QConv(nn.Module):
    def __init__(self, in_feat, inter_dim, out_feat):
        super(QConv, self).__init__()
        self.linear2 = nn.Linear(in_feat + inter_dim, out_feat)
        self.linear1 = nn.Linear(in_feat + 3, inter_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)

    def message_func(self, edges):
        # print(f'node h {edges.src["h"].shape}')
        # print(f'node w {edges.data["w"].shape}')
        return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}

    def reduce_func(self, nodes):
        # print(f'node m {nodes.mailbox["m"].shape}')
        tmp = self.linear1(nodes.mailbox['m'])
        tmp = F.leaky_relu(tmp)
        h = torch.mean(tmp, dim=1)
        # h = torch.max(tmp, dim=1).values
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        # g.edata['w'] = w #self.embed(torch.unsqueeze(w,1))
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        h_linear = self.linear2(h_total)
        h_relu = F.relu(h_linear)
        # h_norm = torch.unsqueeze(torch.linalg.norm(h_relu, dim=1), dim=1)
        # h_normed = torch.divide(h_relu, h_norm)
        # return h_normed
        return h_relu


class QGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, inter_dim):
        super(QGNN, self).__init__()
        self.conv1 = QConv(in_feats, inter_dim, h_feats)
        self.conv2 = QConv(h_feats, inter_dim, h_feats)
        self.conv3 = QConv(h_feats, inter_dim, h_feats)
        self.conv4 = QConv(h_feats, inter_dim, h_feats)
        self.conv5 = QConv(h_feats, inter_dim, h_feats)
        # self.attn = nn.MultiheadAttention(embed_dim=h_feats, num_heads=1)
        self.linear1 = nn.Linear(h_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, num_classes)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)
        self.embedding = nn.Embedding(in_feats, in_feats)

    def forward(self, g):
        # print(g.ndata['gate_type'])
        # print(self.embedding)
        g.ndata['h'] = self.embedding(g.ndata['gate_type'])
        w = torch.cat(
            [
                torch.unsqueeze(g.edata['src_idx'], 1),
                torch.unsqueeze(g.edata['dst_idx'], 1),
                torch.unsqueeze(g.edata['reversed'], 1),
            ],
            dim=1,
        )
        g.edata['w'] = w
        h = self.conv1(g, g.ndata['h'])
        h = self.conv2(g, h)
        h = self.conv3(g, h)
        h = self.conv4(g, h)
        h = self.conv5(g, h)
        # h, _ = self.attn(h, h, h)
        # print(h.shape)
        # print(f'h: {h}')
        # batched_attn_output = []
        # num_nodes = g.batch_num_nodes()
        # b, e = 0, 0
        # for i in range(len(num_nodes)):
        #     if i != 0:
        #         b += num_nodes[i - 1]
        #     e += num_nodes[i]
        #     attn_output, attn_output_weight = self.attn(h[b:e], h[b:e], h[b:e])
        #     # print(f'h[b:e]: {h[b:e]}')
        #     # print(f'attn_output shape: {attn_output.shape}')
        #     # print(f'attn_output: {attn_output}')
        #     # print(f'attn_output_weight shape: {attn_output_weight.shape}')
        #     # print(f'attn_output_weight: {attn_output_weight}')
        #     batched_attn_output.append(attn_output)
        # h = torch.cat(batched_attn_output)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h


#################### Agent ####################


class QAgent:
    def __init__(
        self,
        *,
        lr,
        lr_decay_interval,
        lr_decay_rate,
        gamma,
        a_size,
        context,
        f=None,
        pretrained_model=None,
    ):
        torch.manual_seed(42)
        if pretrained_model != None:
            self.q_net = copy.deepcopy(pretrained_model).to(device)
        else:
            self.q_net = QGNN(gate_type_num, 64, a_size, 64).to(device)

        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.loss_fn = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, lr_decay_interval, lr_decay_rate
        )
        self.a_size = a_size
        self.gamma = gamma
        self.context = context
        self.f = f

    def select_a(self, g, dgl_g, e):
        a_size = self.a_size

        if random.random() < e:
            node = np.random.randint(0, dgl_g.num_nodes())
            A = np.random.randint(0, a_size)
            xfers = g.available_xfers(
                context=self.context, node=g.get_node_from_id(id=node)
            )
            if xfers != []:
                A = random.choice(xfers)
                self.print_log(f'randomly select node {node} and xfer {A}')
            else:
                A = np.random.randint(0, a_size)
                self.print_log(
                    f'randomly select node {node} and xfer {A} (unavailable)'
                )
            # self.print_log(f'randomly select node {node} and xfer {A}')
            A = torch.tensor(A)
            node = torch.tensor(node)

        else:
            dgl_g = dgl_g.to(device)
            with torch.no_grad():
                pred = self.q_net(dgl_g)
            Qs, As = torch.max(pred, dim=1)
            Q, node = torch.max(Qs, dim=0)
            A = As[node]
            self.print_log(f'select node {node} and xfer {A}')

        return node, A

    def train(self, data, batch_size):
        losses = 0
        pred_rs = []
        target_rs = []

        ss = []
        nodes = []
        actions = []
        rs = []
        s_nexts = []
        none_masks = []
        for i in range(batch_size):
            s, node, a, r, s_next = data.get_data()
            ss.append(s)
            nodes.append(node)
            actions.append(a)
            rs.append(r)
            none_masks.append(s_next == None)
            if s_next != None:
                s_nexts.append(s_next)

            # if s_next == None:
            #     target_r = torch.tensor(-5.0)
            #     if self.cuda:
            #         target_r = target_r.cuda()
            # else:
            #     if self.cuda:
            #         s_next = s_next.to('cuda:0')
            #     q_next = self.target_net(s_next).detach()
            #     # q_next_r = q_next[node][a]
            #     q_next_r = torch.max(q_next)
            #     # print(q_next_r)
            #     # print(q_next.shape)
            #     target_r = r + self.gamma * q_next_r

            # target_rs.append(target_r)

        batched_ss = dgl.batch(ss)
        batched_ss = batched_ss.to(device)
        batched_preds = self.q_net(batched_ss)
        num_nodes = batched_ss.batch_num_nodes()
        b, e = 0, 0
        for i in range(batch_size):
            if i != 0:
                b += num_nodes[i - 1]
            e += num_nodes[i]

            pred_rs.append(batched_preds[b:e][nodes[i]][actions[i]])

        if s_nexts != []:
            batched_nexts = dgl.batch(s_nexts)
            batched_nexts = batched_nexts.to(device)
            batched_qs = self.target_net(batched_nexts)
            num_nodes = batched_nexts.batch_num_nodes()
        idx, b, e = 0, 0, 0
        for i in range(batch_size):
            if not none_masks[i]:
                if idx != 0:
                    b += num_nodes[idx - 1]
                e += num_nodes[idx]
                idx += 1

                q_next = torch.max(batched_qs[b:e])
                print(f'q_next: {q_next}')
                target_r = rs[i] + self.gamma * q_next
            else:
                target_r = torch.tensor(-5.0)
                target_r = target_r.to(device)
            target_rs.append(target_r)

        loss = self.loss_fn(
            torch.stack(pred_rs), torch.stack(target_rs).clone().detach()
        )
        self.print_log(torch.stack(pred_rs))
        self.print_log(torch.stack(target_rs))
        self.print_log(loss)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def print_log(self, s):
        s = str(s)
        print(s)
        if self.f != None:
            print(s, file=self.f)

    def __del__(self):
        if self.f != None:
            self.f.close()


#################### Replay Buffer ####################


class QData:
    def __init__(self):
        self.data = deque(maxlen=20000)
        self.hash_map = {}

    # def add_data(self, d):
    #     if d[4] == None:
    #         d_0_hash = d[0].hash()
    #         if d_0_hash not in self.hash_map:
    #             self.hash_map[d_0_hash] = d[0].to_dgl_graph()
    #         self.data.append((d_0_hash, d[1], d[2], d[3], None))
    #     else:
    #         d_0_hash = d[0].hash()
    #         d_4_hash = d[4].hash()
    #         if d_0_hash not in self.hash_map:
    #             self.hash_map[d_0_hash] = d[0].to_dgl_graph()
    #         if d_4_hash not in self.hash_map:
    #             self.hash_map[d_4_hash] = d[4].to_dgl_graph()
    #         self.data.append((d_0_hash, d[1], d[2], d[3], d_4_hash))

    def add_data(self, d):
        self.data.append(d)

    # def add_data_dgl(self, d):
    #     if d[5] == None:
    #         if d[1] not in self.hash_map:
    #             self.hash_map[d[1]] = d[0]
    #         self.data.append((d[1], d[2], d[3], d[4], None))

    #     else:
    #         if d[1] not in self.hash_map:
    #             self.hash_map[d[1]] = d[0]
    #         if d[6] not in self.hash_map:
    #             self.hash_map[d[6]] = d[5]
    #         self.data.append((d[1], d[2], d[3], d[4], d[6]))

    def add_data_dgl(self, d):
        self.data.append((d[0], d[2], d[3], d[4], d[5]))

    # def get_data(self):
    #     s = random.choice(self.data)
    #     # print(s)
    #     if s[4] == None:
    #         return self.hash_map[s[0]], s[1], s[2], s[3], None
    #     return self.hash_map[s[0]], s[1], s[2], s[3], self.hash_map[s[4]]

    def get_data(self):
        s = random.choice(self.data)
        return s


#################### RL training ####################


def train(
    *,
    lr,
    lr_decay_interval,
    lr_decay_rate,
    gamma,
    a_size,
    replay_times,
    episodes,
    epsilon,
    epsilon_decay,
    train_epoch,
    context,
    init_graph,
    max_seq_len,
    batch_size,
    target_update_interval=1,
    log_fn='',
    pretrained_model=None,
    pos_data_init=False,
    pos_data_sampling=False,
    **kwargs,
):

    # Prepare for log
    if log_fn != '':
        f = open(log_fn, "w")
    else:
        f = None

    def print_log(s):
        s = str(s)
        print(s)
        if f != None:
            print(s, file=f)

    if pretrained_model == None:
        agent = QAgent(
            lr=lr,
            lr_decay_interval=lr_decay_interval,
            lr_decay_rate=lr_decay_rate,
            gamma=gamma,
            a_size=a_size,
            context=context,
            f=f,
        )
    else:
        agent = QAgent(
            lr=lr,
            lr_decay_interval=lr_decay_interval,
            lr_decay_rate=lr_decay_rate,
            gamma=gamma,
            a_size=a_size,
            context=context,
            f=f,
            pretrained_model=pretrained_model,
        )
    data = QData()
    pos_data = PosRewardData()
    pos_data.load_data()
    if pos_data_init:
        for d in pos_data.all_data():
            data.add_data_dgl(d)

    if pos_data_sampling:
        assert 'pos_data_sampling_rate' in kwargs
        pos_sampling_times = int(kwargs['pos_data_sampling_rate'] * replay_times)

    average_seq_lens = []
    correct_cnts = []
    average_rewards = []
    training_loss = []

    for i in tqdm(range(episodes)):
        rewards = 0
        losses = 0
        average_seq_len = 0
        if pos_data_sampling:
            for j in range(pos_sampling_times):
                d = pos_data.sample()
                data.add_data_dgl(d)

        for j in range(replay_times):

            count = 0
            end = False
            g = init_graph
            while count < max_seq_len and not end:
                dgl_g = g.to_dgl_graph()
                count += 1
                node, A = agent.select_a(g, dgl_g, epsilon)
                # print(A)
                new_g = g.apply_xfer(
                    xfer=context.get_xfer_from_id(id=A), node=g.all_nodes()[node]
                )

                if new_g == None:
                    end = True
                    data.add_data((dgl_g, node, A, torch.tensor(-5), None))
                    print_log("end")

                else:
                    reward = g.gate_count - new_g.gate_count

                    data.add_data(
                        (dgl_g, node, A, torch.tensor(reward), new_g.to_dgl_graph())
                    )

                    g = new_g
                    rewards += reward
                    if reward > 0:
                        print_log("Reduced!")
                    print_log(g.gate_count)

            average_seq_len += count
        average_seq_len /= replay_times
        average_seq_lens.append(average_seq_len)
        average_rewards.append(rewards / replay_times)
        print_log(f"average sequence length: {average_seq_len}")
        print_log(f"average total reward: {rewards / replay_times}")

        for j in range(train_epoch):
            loss = agent.train(data, batch_size)
            losses += loss
        print_log(f'Training loss: {losses}')
        training_loss.append(losses)
        agent.scheduler.step()

        if epsilon > 0.05:
            epsilon -= epsilon_decay

        if i % target_update_interval == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        with torch.no_grad():
            pred = agent.q_net(init_graph.to_dgl_graph().to(device))
            _, As = torch.max(pred, dim=1)
            print_log(As)
            if 'valid_xfer_dict' in kwargs:
                correct_cnt = 0
                valid_xfer_dict = kwargs['valid_xfer_dict']
                for k in range(len(As)):
                    a = As[k]
                    if a in valid_xfer_dict[str(k)]:
                        correct_cnt += 1
                print_log(f'correct count is {correct_cnt}')
                correct_cnts.append(correct_cnt)

        print_log(f'end of episode {i}')

    if f != None:
        f.close()

    if 'valid_xfer_dict' in kwargs:
        return average_seq_lens, correct_cnts, average_rewards, training_loss

    return average_seq_lens, average_rewards, training_loss


if __name__ == '__main__':

    experiment_name = "rl_dqn_" + "debug"

    quartz_context = quartz.QuartzContext(
        gate_set=['h', 'cx', 't', 'tdg'], filename='../bfs_verified_simplified.json'
    )
    parser = quartz.PyQASMParser(context=quartz_context)

    init_dag = parser.load_qasm(filename="barenco_tof_3_opt_path/subst_history_39.qasm")
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
        fvalid_xfer_dict = json.load(f)

    # RL training
    seq_lens, correct_cnts, rewards, losses = train(
        lr=2e-3,
        lr_decay_interval=100,
        lr_decay_rate=0.999,
        gamma=0.999,
        replay_times=10,
        a_size=quartz_context.num_xfers,
        episodes=2000,
        epsilon=0.5,
        epsilon_decay=0.0001,
        train_epoch=30,
        max_seq_len=80,
        batch_size=100,
        context=quartz_context,
        init_graph=init_graph,
        target_update_interval=100,
        log_fn=f"log/{experiment_name}_log.txt",
        valid_xfer_dict=valid_xfer_dict,
        pos_data_init=False,
        pos_data_sampling=False,
        pos_data_sampling_rate=0.1,
    )

    fig, ax = plt.subplots()
    ax.plot(seq_lens)
    plt.title("sequence length - training epochs")
    plt.savefig(f'figures/{experiment_name}_seqlen.png')

    fig, ax = plt.subplots()
    ax.plot(correct_cnts)
    plt.title("correct counts - training epochs")
    plt.savefig(f'figures/{experiment_name}_corrcnt.png')

    fig, ax = plt.subplots()
    ax.plot(rewards)
    plt.title("rewards - training epochs")
    plt.savefig(f'figures/{experiment_name}_rewards.png')

    fig, ax = plt.subplots()
    ax.plot(losses)
    plt.title("losses - training epochs")
    plt.savefig(f'figures/{experiment_name}_loss.png')

    def find_number(fn, n):

        with open(fn, 'r') as f:
            for l in f:
                if l[:2] == str(n):
                    print(f"{n} found!")
                    return
        print(f"{n} not found!")

    find_number(f"log/{experiment_name}_log.txt", 52)
