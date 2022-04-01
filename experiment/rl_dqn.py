import numpy as np
import torch
import copy
from gnn import QGNN
import random
from collections import deque
from tqdm import tqdm
from pos_data import PosRewardData

# Constant
gate_type_num = 26


class QAgent:
    def __init__(self,
                 *,
                 lr,
                 gamma,
                 a_size,
                 context,
                 f=None,
                 pretrained_model=None,
                 use_cuda=False):
        torch.manual_seed(42)
        if pretrained_model != None:
            self.q_net = copy.deepcopy(pretrained_model)
        else:
            self.q_net = QGNN(gate_type_num, 64, a_size, 64)
        self.target_net = copy.deepcopy(self.q_net)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 100, 0.90)
        self.a_size = a_size
        self.gamma = gamma
        self.context = context
        self.f = f
        if use_cuda:
            self.use_cuda()
            self.cuda = True
        else:
            self.cuda = False

    def use_cuda(self):
        self.q_net.cuda()
        self.target_net.cuda()
        self.cuda = True

    def select_a(self, g, dgl_g, e):
        a_size = self.a_size

        if random.random() < e:
            node = np.random.randint(0, dgl_g.num_nodes())
            A = np.random.randint(0, a_size)
            xfers = g.available_xfers(context=self.context,
                                      node=g.get_node_from_id(id=node))
            if xfers != []:
                A = random.choice(xfers)
                self.print_log(f'randomly select node {node} and xfer {A}')
            else:
                A = np.random.randint(0, a_size)
                self.print_log(
                    f'randomly select node {node} and xfer {A} (unavailable)')
            # self.print_log(f'randomly select node {node} and xfer {A}')
            A = torch.tensor(A)
            node = torch.tensor(node)

        else:
            if self.cuda:
                dgl_g = dgl_g.to('cuda:0')
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
        for i in range(batch_size):
            s, node, a, r, s_next = data.get_data()

            if self.cuda:
                s = s.to('cuda:0')
            pred = self.q_net(s)
            pred_r = pred[node][a]
            #s_a = s_as.gather(1, a)

            if s_next == None:
                target_r = torch.tensor(-5.0)
                if self.cuda:
                    target_r = target_r.cuda()
            else:
                if self.cuda:
                    s_next = s_next.to('cuda:0')
                q_next = self.target_net(s_next).detach()
                # q_next_r = q_next[node][a]
                q_next_r = torch.max(q_next)
                # print(q_next_r)
                # print(q_next.shape)
                target_r = r + self.gamma * q_next_r

            pred_rs.append(pred_r)
            target_rs.append(target_r)
        loss = self.loss_fn(torch.stack(pred_rs), torch.stack(target_rs))
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


class QData:
    def __init__(self):
        self.data = deque(maxlen=10000)
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


# RL training
def train(*,
          lr,
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
          use_cuda=False,
          pos_data_init=False,
          pos_data_sampling=False,
          **kwargs):

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
        agent = QAgent(lr=lr, gamma=gamma, a_size=a_size, context=context, f=f)
    else:
        agent = QAgent(lr=lr,
                       gamma=gamma,
                       a_size=a_size,
                       context=context,
                       f=f,
                       pretrained_model=pretrained_model)
    if use_cuda:
        agent.use_cuda()

    data = QData()
    pos_data = PosRewardData()
    pos_data.load_data()
    if pos_data_init:
        for d in pos_data.all_data():
            data.add_data_dgl(d)

    if pos_data_sampling:
        assert 'pos_data_sampling_rate' in kwargs
        pos_sampling_times = int(kwargs['pos_data_sampling_rate'] *
                                 replay_times)

    average_seq_lens = []
    correct_cnts = []
    average_rewards = []

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
                new_g = g.apply_xfer(xfer=context.get_xfer_from_id(id=A),
                                     node=g.all_nodes()[node])

                if new_g == None:
                    end = True
                    data.add_data((dgl_g, node, A, torch.tensor(-5), None))
                    print_log("end")

                else:
                    reward = g.gate_count - new_g.gate_count

                    data.add_data((dgl_g, node, A, torch.tensor(reward),
                                   new_g.to_dgl_graph()))

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
            agent.scheduler.step()

        if epsilon > 0.05:
            epsilon -= epsilon_decay

        if i % target_update_interval == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        with torch.no_grad():
            if use_cuda:
                pred = agent.q_net(init_graph.to_dgl_graph().to('cuda:0'))
            else:
                pred = agent.q_net(init_graph.to_dgl_graph())
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
        return average_seq_lens, correct_cnts, average_rewards

    return average_seq_lens, average_rewards
