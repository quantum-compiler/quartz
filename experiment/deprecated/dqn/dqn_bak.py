import heapq
import json
import math
import os
import random
import sys
import warnings
from collections import deque, namedtuple
from typing import List, Tuple

import dgl
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from IPython import embed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

import quartz

sys.path.append(os.path.join(os.getcwd(), '..'))
from pretrain.pretrain import PretrainNet


def topk_2d(
    x: torch.Tensor, k: int, as_tuple: bool = False
) -> Tuple[torch.Tensor, torch.tensor | Tuple[torch.tensor, torch.tensor]]:
    x_f = x.flatten()
    v_topk, i_f_topk = torch.topk(x_f, k)
    i_f_topk = i_f_topk.tolist()
    i_topk = [(i_f // x.shape[1], i_f % x.shape[1]) for i_f in i_f_topk]
    if not as_tuple:
        i_topk = torch.tensor(i_topk, device=x.device)
    else:
        i_topk_sep = list(zip(*i_topk))
        i_topk = (
            torch.tensor(i_topk_sep[0], device=x.device),
            torch.tensor(i_topk_sep[1], device=x.device),
        )
    return v_topk, i_topk


class SpecialMinHeap:
    def __init__(self, max_size=math.inf):
        self.arr = []
        self.inserted_hash = set()
        self.max_size = max_size

    def __len__(self):
        return len(self.arr)

    def push(self, item):
        popped = None
        if item[1].hash() not in self.inserted_hash:
            self.inserted_hash.add(item[1].hash())
            if len(self.arr) >= self.max_size:
                popped = self.pop()
            heapq.heappush(self.arr, item)
        return popped

    def pop(self):
        popped = heapq.heappop(self.arr)
        self.inserted_hash.remove(popped[1].hash())
        return popped

    def sample(self, k: int = 1):
        # arr: [ (num_gate_reduced, graph) ]
        assert len(self.arr) > 0 and isinstance(self.arr[0], tuple)
        num_gate_reduced_list, graph_list = zip(*self.arr)
        num_gate_reduced_list = list(num_gate_reduced_list)
        graph_list = list(graph_list)
        # generate sample distribution
        sample_counts = torch.tensor(num_gate_reduced_list)
        sample_counts = sample_counts - sample_counts.min()  # [0, 0, 1, 2, ...]
        sample_counts = sample_counts * 10 + 1
        sampled = random.sample(self.arr, k=k, counts=sample_counts.tolist())
        return sampled


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
        return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}

    def reduce_func(self, nodes):
        tmp = self.linear1(nodes.mailbox['m'])
        tmp = F.leaky_relu(tmp)
        h = torch.mean(tmp, dim=1)
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        h_linear = self.linear2(h_total)
        h_relu = F.relu(h_linear)
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
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h


Experience = namedtuple(
    'Experience',
    field_names=[
        'state',
        'action_node',
        'action_xfer',
        'reward',
        'next_state',
        'game_over',
    ],
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.prios = torch.Tensor([]).to(device)
        self.max_prio = torch.Tensor([1e5]).to(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, exp: Experience, prio: torch.tensor = None):
        self.buffer.append(exp)
        if prio is None:
            prio = self.max_prio
        prio.reshape(
            1,
        )
        self.prios = torch.cat([self.prios, prio])
        if self.prios.shape[0] > self.capacity:
            self.prios = self.prios[1:]

    def update_prios(self, indices: List[int], prios: torch.Tensor):
        self.prios[indices] = prios

    def sample_and_del(self, sample_size: int = 100) -> Tuple:
        sample_size = min(sample_size, len(self.buffer))
        indices = torch.multinomial(self.prios, sample_size, replacement=False)
        indices_list = indices.tolist()
        # states, action_nodes, action_xfers, rewards, next_states, game_overs = \
        #     list(zip(*(self.buffer[idx] for idx in indices_list)))
        exps = [self.buffer[idx] for idx in indices_list]
        print(f'before: {len(self.buffer)}')
        self.buffer = [
            self.buffer[idx]
            for idx in range(len(self.buffer))
            if idx not in indices_list
        ]
        print(f'after: {len(self.buffer)}')
        return exps
        # return (
        #     states, action_nodes, action_xfers, rewards, next_states, game_overs
        # )


class RLDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 100):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        """
        Samples many items from buffer, but only yield one each time
        """
        exps = self.buffer.sample_and_del(self.sample_size)
        states, action_nodes, action_xfers, rewards, next_states, game_overs = zip(
            *exps
        )
        for i in range(len(states)):
            yield (
                states[i],
                action_nodes[i],
                action_xfers[i],
                rewards[i],
                next_states[i],
                game_overs[i],
            )


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 1):
        self.size = size

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.size


class Environment:
    def __init__(
        self,
        init_graph: quartz.PyGraph,
        quartz_context: quartz.QuartzContext,
        max_steps_per_episode: int = 100,
        nop_policy: list = ['c', 0.0],
        game_over_when_better: bool = True,
    ):
        self.init_graph = init_graph
        self.quartz_context = quartz_context
        self.num_xfers = quartz_context.num_xfers
        self.max_steps_per_episode = max_steps_per_episode
        self.nop_policy = nop_policy
        self.game_over_when_better = game_over_when_better

        self.state = None
        self.cur_step = 0
        self.exp_seq = [self._get_firtst_exp()]

        self.reset()

    def reset(self):
        self.state: quartz.PyGraph = self.init_graph
        self.cur_step = 0
        self.exp_seq = [self._get_firtst_exp()]

    def set_init_state(self, graph: quartz.PyGraph):
        self.init_graph = graph
        self.state = graph
        self.cur_step = 0
        self.exp_seq = [self._get_firtst_exp()]

    def _get_firtst_exp(self) -> Experience:
        return Experience(
            None,
            0,
            0,
            0,
            self.init_graph,
            False,
        )

    def get_action_space(self) -> Tuple[List[int], List[int]]:
        return (list(range(self.state.gate_count)), list(range(self.num_xfers)))

    def available_xfers(self, action_node: int) -> int:
        xfers = self.state.available_xfers(
            context=self.quartz_context,
            node=self.state.get_node_from_id(id=action_node),
        )
        return xfers

    def act(self, action_node: int, action_xfer: int) -> Experience:
        cur_state = self.state
        next_state: quartz.PyGraph = self.state.apply_xfer(
            xfer=self.quartz_context.get_xfer_from_id(id=action_xfer),
            node=cur_state.all_nodes()[action_node],
        )
        if next_state is None:
            game_over = True
            this_step_reward = -2  # TODO  neg reward when the xfer is invalid
        else:
            game_over = False
            this_step_reward = cur_state.gate_count - next_state.gate_count

        # handle NOP
        if self.quartz_context.xfer_id_is_nop(xfer_id=action_xfer):
            # if action_xfer == self.quartz_context.num_xfers:
            game_over = self.nop_policy[0] == 's'
            this_step_reward = self.nop_policy[1]

        self.cur_step += 1
        self.state = next_state
        if self.cur_step >= self.max_steps_per_episode:
            game_over = True
            # TODO  whether we need to change this_step_reward here?

        if self.game_over_when_better and this_step_reward > 0:
            game_over = True  # TODO  only reduce gate count once in each episode
            # TODO  note this case: 58 -> 80 -> 78

        exp = Experience(
            cur_state, action_node, action_xfer, this_step_reward, next_state, game_over
        )
        self.exp_seq.append(exp)
        return exp

    def back_step(self):
        self.state = self.exp_seq[-1].state
        self.cur_step -= 1
        self.exp_seq.pop()


class Agent:
    def __init__(
        self,
        env: Environment,
        device: torch.device,
    ):
        self.env = env
        self.device = device

        self.choices = [('s', 0)]

    @torch.no_grad()
    def _get_action(
        self,
        q_net: nn.Module,
        eps: float,
        topk: int = 1,
        append_choice: bool = True,
    ) -> Tuple[torch.tensor]:
        if np.random.random() < (1 - eps):
            # greedy
            cur_state: dgl.DGLGraph = self.env.state.to_dgl_graph().to(self.device)
            # (num_nodes, num_actions)
            q_values = q_net(cur_state)
            topk_q_values, topk_actions = topk_2d(q_values, topk, as_tuple=False)
            # only append the first q_value; should not be used when topk > 1
            if append_choice:
                self.choices.append(('q', eps, topk_q_values[0].item()))
            # return values will be added into data buffer, which latter feeds the dataset
            # they cannot be CUDA tensors, or CUDA error will occur in collate_fn
            return topk_actions, topk_q_values
        else:
            # random
            # TODO  whether we need to include invalid xfers in action space?
            # TODO  distinguish invalid xfers from xfers leading to gate count increase?
            node_space, xfer_space = self.env.get_action_space()
            node = np.random.choice(node_space)
            if np.random.random() < 1:
                av_xfers = self.env.available_xfers(node)
                if len(av_xfers) > 0:
                    xfer_space = av_xfers
            xfer = np.random.choice(xfer_space)
            self.choices.append(('r', eps, 0))
            return torch.tensor([[node, xfer]]), torch.Tensor([0])

    def play_step(self, q_net: nn.Module, eps: float) -> Experience:
        action, _ = self._get_action(q_net, eps)  # [ [node, xfer] ]
        exp = self.env.act(action[0, 0].item(), action[0, 1].item())
        return exp

    def clear_choices(self):
        self.choices = [('s', 0, 0)]


class DQNMod(pl.LightningModule):
    def __init__(
        self,
        init_graph_qasm_str: str,
        gate_type_num: int = 26,
        gate_set: List = ['h', 'cx', 't', 'tdg'],
        ecc_file: str = 'bfs_verified_simplified.json',
        no_increase: bool = True,
        include_nop: bool = False,
        lr: float = 1e-3,
        batch_size: int = 128,
        eps_init: float = 0.5,
        eps_decay: float = 0.0001,
        eps_min: float = 0.05,
        gamma: float = 0.9,
        episode_length: int = 30,  # TODO  check these hparams
        replaybuf_size: int = 10_000,
        warm_start_steps: int = 512,
        target_update_interval: int = 100,
        seq_out_dir: str = 'out_graphs',
        pretrained_weight_path: str = None,
        restore_weight_after_better: bool = False,
        nop_policy: list = ['c', 0.0],
        sample_init: bool = False,
        restart_from_best: bool = False,
        game_over_when_better: bool = True,
        strict_better: bool = True,
        clear_buf_after_better: bool = False,
        init_state_buf_size: int = 512,
        agent_episode: bool = True,
        qgnn_h_feats: int = 64,
        qgnn_inter_dim: int = 64,
        test_topk: int = 3,
        mode: str = 'unknown',
    ):
        super().__init__()
        self.save_hyperparameters()

        quartz_context = quartz.QuartzContext(
            gate_set=gate_set,
            filename=ecc_file,
            # TODO  we need to include xfers that lead to gate increase when training?
            # we may exclude them when generating the dataset for pre-training
            # TODO  to make the task easier, we exclude those xfers currently
            no_increase=no_increase,
            include_nop=include_nop,
        )
        self.quartz_context = quartz_context
        self.num_xfers = quartz_context.num_xfers
        parser = quartz.PyQASMParser(context=quartz_context)
        init_dag = parser.load_qasm_str(init_graph_qasm_str)
        init_graph = quartz.PyGraph(context=quartz_context, dag=init_dag)
        self.init_graph = init_graph

        self.q_net = QGNN(
            self.hparams.gate_type_num,
            self.hparams.qgnn_h_feats,
            quartz_context.num_xfers,
            self.hparams.qgnn_inter_dim,
        )
        self.target_net = QGNN(
            self.hparams.gate_type_num,
            self.hparams.qgnn_h_feats,
            quartz_context.num_xfers,
            self.hparams.qgnn_inter_dim,
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.loss_fn = nn.MSELoss()

        self.env = Environment(
            init_graph=init_graph,
            quartz_context=quartz_context,
            max_steps_per_episode=episode_length,
            nop_policy=nop_policy,
            game_over_when_better=game_over_when_better,
        )
        # we will set device for agent in on_after_batch_transfer latter
        self.agent = Agent(self.env, self.device)
        self.buffer = ReplayBuffer(replaybuf_size)
        self.init_state_buffer = SpecialMinHeap(max_size=init_state_buf_size)
        self.init_state_buffer.push((0, init_graph))
        self.best_graph = init_graph

        self.eps = eps_init
        self.episode_reward = 0
        self.total_reward = 0
        self.num_out_graphs = 0
        self.pretrained_q_net = QGNN(
            self.hparams.gate_type_num,
            self.hparams.qgnn_h_feats,
            quartz_context.num_xfers,
            self.hparams.qgnn_inter_dim,
        )

        self.load_pretrained_weight()
        if mode != 'test':
            self.populate(self.hparams.warm_start_steps)

    def load_pretrained_weight(self):
        if self.hparams.pretrained_weight_path is not None:
            ckpt_path = self.hparams.pretrained_weight_path
            assert os.path.exists(ckpt_path)
            pretrained_net = PretrainNet.load_from_checkpoint(ckpt_path)
            self.pretrained_q_net = pretrained_net.q_net
            self.q_net.load_state_dict(self.pretrained_q_net.state_dict())
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _restore_pretrained_weight(self):
        if self.pretrained_q_net is not None:
            self.q_net.load_state_dict(self.pretrained_q_net.state_dict())
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _output_seq(self, exp_seq: list = None, choices: list = None):
        # output sequence
        self.num_out_graphs += 1
        out_dir = os.path.join(
            self.hparams.seq_out_dir,
            'out_graphs',
            f'{self.num_out_graphs}',
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if exp_seq is None:
            exp_seq = self.env.exp_seq
        if choices is None:
            choices = self.agent.choices
        for i_step, exp in enumerate(exp_seq):
            out_path = os.path.join(
                out_dir,
                f'{i_step}_{exp.next_state.gate_count}_{exp.action_node}_{exp.action_xfer}_'
                f'{choices[i_step][0]}_{choices[i_step][1]:.3f}_{choices[i_step][2]:.3f}.qasm',
            )
            qasm_str = exp.next_state.to_qasm_str()
            with open(out_path, 'w') as f:
                print(qasm_str, file=f)

    def agent_step(self, eps: float) -> Tuple[Experience, int]:
        exp = self.agent.play_step(self.q_net, eps)
        self.buffer.append(exp)
        env_cur_step = self.env.cur_step

        if (
            exp.next_state
            and exp.next_state.gate_count < self.best_graph.gate_count
            and (not self.hparams.strict_better or self.agent.choices[-1][0] == 'q')
        ):
            # better graph found, add it to init_state_buffer
            self.init_state_buffer.push(
                (self.init_graph.gate_count - exp.next_state.gate_count, exp.next_state)
            )
        elif exp.next_state and (
            np.random.random() < 0.10 or len(self.init_state_buffer) < 10
        ):
            self.init_state_buffer.push(
                (self.init_graph.gate_count - exp.next_state.gate_count, exp.next_state)
            )

        if exp.next_state and exp.next_state.gate_count < self.best_graph.gate_count:
            # a better graph is found
            print(
                f'\n!!! Better graph with gate_count {exp.next_state.gate_count} found!'
                f' method: {self.agent.choices[-1][0]}, eps: {self.agent.choices[-1][1]: .3f}, q: {self.agent.choices[-1][2]: .3f}'
            )
            if not self.hparams.strict_better or self.agent.choices[-1][0] == 'q':
                print(f'Best graph updated to {exp.next_state.gate_count} .')
                self.best_graph = exp.next_state
                self._output_seq()

        if exp.game_over:
            # TODO  if a better graph is found, clear the buffer and populate it with the new graph?
            # reset
            if self.hparams.sample_init:
                # sample a init state
                init_state_pair = self.init_state_buffer.sample(k=1)[0]
                self.env.set_init_state(init_state_pair[1])
            elif self.hparams.restart_from_best:
                # start from the best graph
                self.env.set_init_state(self.best_graph)
            else:
                self.env.reset()

            self.agent.clear_choices()
            if self.hparams.clear_buf_after_better:
                # clear and re-populate
                self.buffer.buffer.clear()
                # may meet gate count reduction again, which introduces recursions
                self.populate(self.hparams.warm_start_steps)
            if self.hparams.restore_weight_after_better:
                self._restore_pretrained_weight()
        # end if
        return exp, env_cur_step

    def agent_episode(self, eps: float) -> Tuple[Experience, int]:
        while True:
            exp, env_cur_step = self.agent_step(eps)
            if exp.game_over is True:
                break
        return exp, env_cur_step

    def populate(self, steps: int = 1000):
        """
        Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in tqdm(range(steps), desc='Populating the buffer'):
            self.agent_step(1.0)

    def _compute_loss(self, batch) -> torch.Tensor:
        (
            states,
            action_nodes,
            action_xfers,
            rewards,
            next_states,
            game_overs,
            ids,
        ) = batch
        cur_num_nodes = states.batch_num_nodes().tolist()
        next_num_nodes = next_states.batch_num_nodes().tolist()

        # ( sum(num of nodes), num_xfers )
        pred_q_values = self.q_net(states)
        if max(next_num_nodes) > 0:
            with torch.no_grad():
                target_next_q_values = self.target_net(next_states)
            # ( sum(num of nodes), )
            target_next_max_q_values_all_nodes, _ = torch.max(
                target_next_q_values, dim=1
            )
        # compute max q values for next states
        target_next_max_q_values = []
        r_next_start, r_next_end = 0, 0
        r_cur_start, r_cur_end = 0, 0
        presum_cur_num_nodes = 0
        for i_batch in range(len(cur_num_nodes)):
            r_next_end += next_num_nodes[i_batch]
            r_next = slice(r_next_start, r_next_end)
            r_cur_end += cur_num_nodes[i_batch]
            r_cur = slice(r_cur_start, r_cur_end)

            target_next_max_q_values.append(
                torch.max(target_next_max_q_values_all_nodes[r_next])
                if not game_overs[i_batch]
                else torch.tensor(0.0).to(self.device)
            )  # each elem has size [] (0-dim tensor)
            action_nodes[i_batch] += presum_cur_num_nodes
            presum_cur_num_nodes += cur_num_nodes[i_batch]

            r_next_start = r_next_end
            r_cur_start = r_cur_end
        # (batch_size, )
        target_next_max_q_values = torch.stack(target_next_max_q_values)
        # pred_Q = reward_of_action + gamma * target_next_max_Q
        acted_pred_q_values = pred_q_values[action_nodes, action_xfers]
        target_max_q_values = rewards + self.hparams.gamma * target_next_max_q_values
        loss = self.loss_fn(acted_pred_q_values, target_max_q_values)

        self.log_dict(
            {
                f'mean_batch_reward': rewards.mean(),
                f'mean_target_next_max_Q': target_next_max_q_values.mean(),
                f'mean_target_max_Q': target_max_q_values.mean(),
                f'mean_pred_Q': acted_pred_q_values.mean(),
                f'max_batch_reward': rewards.max(),
                f'max_target_next_max_Q': target_next_max_q_values.max(),
                f'max_target_max_Q': target_max_q_values.max(),
                f'max_pred_Q': acted_pred_q_values.max(),
            },
            on_step=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        """
        1. Carries out a single step to add one experience to the replay buffer.
        2. GD on the q_net with a batch of data

        Args:
            batch: (states, action_nodes, action_xfers, rewards, next_states, game_overs)
        """
        # play one step
        self.eps = max(self.eps - self.hparams.eps_decay, self.hparams.eps_min)
        # TODO  for _ in range(self.hparams.batch_size // 10):
        if self.hparams.agent_episode:
            exp, env_cur_step = self.agent_episode(self.eps)
        else:
            exp, env_cur_step = self.agent_step(self.eps)
        self.episode_reward += exp.reward
        self.log(f'episode_reward', self.episode_reward, on_step=True)
        # GD with sampled data
        loss = self._compute_loss(batch)

        if exp.game_over:
            self.total_reward += self.episode_reward
            self.episode_reward = 0

        if (
            self.global_step
            and self.global_step % self.hparams.target_update_interval == 0
        ):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.log(f'train_loss', loss)
        self.log(f'step_reward', exp.reward, on_step=True, prog_bar=True)
        self.log(f'total_reward', self.total_reward, on_step=True)
        self.log(f'eps', self.eps, on_step=True)
        self.log(f'env_step', env_cur_step, on_step=True, prog_bar=True)
        self.log(f'best_gc', self.best_graph.gate_count, on_step=True, prog_bar=True)
        self.log(f'init', self.env.init_graph.gate_count, on_step=True, prog_bar=True)
        self.log(f'|buf|', len(self.buffer.buffer), on_step=True, prog_bar=True)
        self.log(
            f'|init_state_buf|',
            len(self.init_state_buffer),
            on_step=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.q_net.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer

    def __dataloader(self) -> torch.utils.data.DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        default_collate = torch.utils.data.dataloader.default_collate

        def collate_fn(batch):
            """
            Args:
                batch: (states, action_nodes, action_xfers, rewards, next_states, game_overs)
                    states: [ quartz.PyGraph ]
                    action_nodes: [ action_node: int ]
                    action_xfers: [ action_xfer: int ]
                    rewards: [ reward: float ]
                    next_states: [ quartz.PyGraph ]
                    game_overs: [ bool ]

            Return: batched data
                batch: (states, action_nodes, action_xfers, rewards, next_states, game_overs)
                    b_states: dgl.DGLGraph (batched_graph)
                    b_action_nodes: torch.tensor
                    b_action_xfers: torch.tensor
                    b_rewards: torch.Tensor
                    b_next_states: dgl.DGLGraph (batched_graph)
                    b_game_overs: torch.tensor of
            """
            (
                states,
                action_nodes,
                action_xfers,
                rewards,
                next_states,
                game_overs,
                ids,
            ) = list(zip(*batch))
            states = [state.to_dgl_graph() for state in states]
            next_states = [
                next_state.to_dgl_graph() if next_state is not None else dgl.DGLGraph()
                for next_state in next_states
            ]
            b_states = dgl.batch(states)
            b_next_states = dgl.batch(next_states)
            b_action_nodes = torch.tensor(action_nodes)
            b_action_xfers = torch.tensor(action_xfers)
            b_rewards = torch.Tensor(rewards)
            b_game_overs = torch.tensor(game_overs, dtype=torch.bool)
            b_ids = default_collate(ids)
            return (
                b_states,
                b_action_nodes,
                b_action_xfers,
                b_rewards,
                b_next_states,
                b_game_overs,
            )

        # Ref: https://pytorch.org/docs/master/notes/randomness.html#dataloader
        g = torch.Generator()
        g.manual_seed(0)

        dataset = RLDataset(self.buffer, 10 * self.hparams.batch_size)
        # TODO  not sure if it can avoid duplicate data when using DDP
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            # shuffle=self.training, # this would be overwritten by PL
            collate_fn=collate_fn,
            generator=g,
        )
        return dataloader

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.__dataloader()

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        self.agent.device = self.device
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = DummyDataset()
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
        )
        return dataloader

    def beam_search(self):
        assert self.env.state == self.init_graph

        cur_graph = self.init_graph
        cur_hash = cur_graph.hash()
        q = [(cur_graph, 0, 0, cur_hash)]
        visited = set()
        visited.add(cur_hash)
        hash_2_graph = {}  # hash -> graph
        hash_2_exp = {cur_hash: (None, 0, 0, 0)}

        best_graph, best_gc, best_hash = cur_graph, cur_graph.gate_count, cur_hash
        print(f'Start BFS')
        with tqdm(
            total=self.init_graph.gate_count,
            desc='num of gates reduced',
            bar_format='{desc}: {n}/{total} |{bar}| {elapsed} {postfix}',
        ) as pbar:

            while len(q) > 0:
                cur_graph, m_cur_q, cur_depth, cur_hash = heapq.heappop(q)
                hash_2_graph[cur_hash] = cur_graph
                if cur_graph.gate_count < best_gc:
                    pbar.update(best_gc - cur_graph.gate_count)
                    best_graph, best_gc, best_hash = (
                        cur_graph,
                        cur_graph.gate_count,
                        cur_hash,
                    )
                    print(
                        f'Better graph with gc {best_gc} is found!'
                        f' Q: {hash_2_exp[cur_hash][3]}'
                    )

                if cur_depth >= self.hparams.episode_length:
                    continue

                cur_dgl_graph: dgl.DGLGraph = cur_graph.to_dgl_graph().to(self.device)
                # (num_nodes, num_actions)
                q_values = self.q_net(cur_dgl_graph)
                topk_q_values, topk_actions = topk_2d(
                    q_values, self.hparams.test_topk, as_tuple=False
                )
                for q_value, action in zip(topk_q_values, topk_actions):
                    action_node = action[0].item()
                    action_xfer = action[1].item()
                    next_graph = cur_graph.apply_xfer(
                        xfer=self.quartz_context.get_xfer_from_id(id=action_xfer),
                        node=cur_graph.all_nodes()[action_node],
                    )
                    if next_graph is not None:
                        next_hash = next_graph.hash()
                        if next_hash not in visited:
                            visited.add(next_hash)
                            hash_2_exp[next_hash] = (
                                cur_hash,
                                action_node,
                                action_xfer,
                                q_value,
                            )
                            heapq.heappush(
                                q, (next_graph, -q_value, cur_depth + 1, next_hash)
                            )
                            if next_graph.gate_count < cur_graph.gate_count:
                                pass

                pbar.set_postfix(
                    {
                        'cur_gate_cnt': cur_graph.gate_count,
                        'best_gate_cnt': best_gc,
                        '|q|': len(q),
                        '|visited|': len(visited),
                    }
                )
                pbar.refresh()
                # end for
            # end while
        # end with
        """save the path of finding the best graph"""
        out_dir = os.path.join(
            self.hparams.seq_out_dir,
            'out_graphs',
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(f'saving the path to {out_dir} ...')
        output_list = []
        cur_hash = best_hash
        while cur_hash is not None:
            cur_graph = hash_2_graph[cur_hash]
            to_cur_exp = hash_2_exp[cur_hash]
            pre_hash, action_node, action_xfer, q_value = to_cur_exp
            out_name = (
                f'{cur_graph.gate_count}_{action_node}_{action_xfer}_{q_value:.3f}.qasm'
            )
            out_qasm = cur_graph.to_qasm_str()
            output_list = [(out_name, out_qasm)] + output_list
            cur_hash = pre_hash
        for i_step, (out_name, out_qasm) in enumerate(output_list):
            out_path = os.path.join(out_dir, f'{i_step}_{out_name}')
            with open(out_path, 'w') as f:
                print(out_qasm, file=f)

    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> int:
        self.beam_search()


def seed_all(seed: int):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_wandb(
    enable: bool = True,
    offline: bool = False,
    project: str = 'Quartz-DQN',
    task: str = 'train',
    entity: str = '',
):
    if enable is False:
        return None
    wandb_logger = WandbLogger(
        entity=entity,
        offline=offline,
        project=project,
        group=task,
    )
    return wandb_logger


def train(cfg):
    wandb_logger = init_wandb(
        enable=cfg.wandb.en,
        offline=cfg.wandb.offline,
        task='train',
        entity=cfg.wandb.entity,
    )
    ckpt_callback_list = [
        ModelCheckpoint(
            monitor='train_loss',  # TODO  other values?
            dirpath=output_dir,
            filename='{epoch}-{train_loss:.2f}-best',
            save_top_k=3,
            save_last=True,
            mode='min',
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=1000_0000,
        gpus=cfg.gpus,
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=ckpt_callback_list,
        sync_batchnorm=True,
        strategy=DDPStrategy(find_unused_parameters=True),
        track_grad_norm=2,
        # detect_anomaly=True,
        # gradient_clip_val=cfg.task.optimizer.clip_value,
        # gradient_clip_algorithm=cfg.task.optimizer.clip_algo,
        # val_check_interval=cfg.val_check_interval,
    )
    if cfg.load_pretrained or not cfg.resume:
        ckpt_path = None
    else:
        ckpt_path = cfg.ckpt_path
    trainer.fit(dqnmod, ckpt_path=ckpt_path)


def test(cfg):
    wandb_logger = init_wandb(
        enable=cfg.wandb.en,
        offline=cfg.wandb.offline,
        task='test',
        entity=cfg.wandb.entity,
    )
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        logger=wandb_logger,
    )
    if cfg.resume is True:
        ckpt_path = cfg.ckpt_path
        assert os.path.exists(ckpt_path)
    else:
        ckpt_path = None
        print(f'Warning: Test from scratch!', file=sys.stderr)
    trainer.test(dqnmod, ckpt_path=ckpt_path)


# global vars
dqnmod: DQNMod = None


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    global quartz_context
    global dqnmod
    global output_dir

    output_dir = os.path.abspath(os.curdir)  # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd())  # set working dir to the original one

    seed_all(cfg.seed)

    # warnings.filterwarnings("ignore", message='DGLWarning: Recommend creating graphs')
    warnings.simplefilter('ignore')

    with open(cfg.init_graph_path) as f:
        init_graph_qasm_str = f.read()

    dqnmod = DQNMod(
        init_graph_qasm_str=init_graph_qasm_str,
        gate_set=cfg.gate_set,
        ecc_file=cfg.ecc_file,
        no_increase=cfg.no_increase,
        include_nop=cfg.include_nop,
        seq_out_dir=output_dir,
        pretrained_weight_path=cfg.pretrained_weight if cfg.load_pretrained else None,
        batch_size=cfg.batch_size,
        warm_start_steps=cfg.batch_size * 2,
        gamma=cfg.gamma,
        replaybuf_size=cfg.replaybuf_size,
        init_state_buf_size=cfg.init_state_buf_size,
        restore_weight_after_better=cfg.restore_weight_after_better,
        target_update_interval=cfg.target_update_interval,
        nop_policy=cfg.nop_policy,
        sample_init=cfg.sample_init,
        restart_from_best=cfg.restart_from_best,
        game_over_when_better=cfg.game_over_when_better,
        clear_buf_after_better=cfg.clear_buf_after_better,
        agent_episode=cfg.agent_episode,
        qgnn_h_feats=cfg.qgnn_h_feats,
        qgnn_inter_dim=cfg.qgnn_inter_dim,
        episode_length=cfg.episode_length,
        test_topk=cfg.test_topk,
        mode=cfg.mode,
    )

    # TODO  how to resume RL training? how to save the state and buffer?

    if cfg.mode == 'train':
        train(cfg)
    elif cfg.mode == 'test':
        test(cfg)
    else:
        raise ValueError(f'Invalid mode: {cfg.mode}')


if __name__ == '__main__':
    main()
