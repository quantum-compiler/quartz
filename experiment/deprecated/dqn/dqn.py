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
import wandb
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
    def __init__(
        self, capacity: int = 100, device: str = 'cpu', prioritized: bool = False
    ):
        self.capacity = capacity
        self.device = device
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.prios = torch.Tensor([]).to(device)
        self.max_prio = torch.Tensor([1e5]).to(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def to(self, device: str):
        self.device = device
        self.prios = self.prios.to(device)
        self.max_prio = self.max_prio.to(device)

    def append(self, exp: Experience, prio: torch.tensor = None) -> None:
        self.buffer.append(exp)
        if self.prioritized:
            """manually append priority"""
            if prio is None:
                prio = self.max_prio
            prio = prio.reshape(
                1,
            ).to(self.device)
            self.prios = torch.cat([self.prios, prio])
            if self.prios.shape[0] > self.capacity:
                self.prios = self.prios[1:]

    def update_prios(
        self, indices: List[int] | torch.Tensor, prios: torch.Tensor
    ) -> None:
        self.prios[indices] = prios

    def sample(
        self, sample_size: int = 100
    ) -> Tuple[List[Experience], torch.Tensor, torch.Tensor, torch.tensor]:
        sample_size = min(sample_size, len(self.buffer))
        if self.prioritized:
            sum_prios = self.prios.sum()
            normed_prios = self.prios / sum_prios
            indices = torch.multinomial(self.prios, sample_size, replacement=False)
            indices_list = indices.tolist()
            exps = [self.buffer[idx] for idx in indices_list]
            return exps, indices, normed_prios[indices], self.prios.min() / sum_prios
        else:
            indices = torch.multinomial(
                torch.ones(len(self)), sample_size, replacement=False
            )
            indices_list = indices.tolist()
            exps = [self.buffer[idx] for idx in indices_list]
            return exps, indices, None, None

    def __str__(self):
        return self.buffer.__str__()


class InitStateBuffer:
    def __init__(self, capacity: int = 100, init_state: quartz.PyGraph = None):
        self.capacity = capacity
        assert init_state is not None
        self.init_state = init_state
        self.q = deque(maxlen=capacity)

    def append(self, item):
        self.q.append(item)

    def __len__(self):
        return len(self.q) + 1

    def sample(self) -> quartz.PyGraph:
        idx = np.random.choice(len(self))
        if idx == len(self.q):
            return self.init_state
        else:
            return self.q[idx]

    def __str__(self):
        return self.q.__str__() + f'\ninit_state: {self.init_state}'


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
        max_additional_gates: int = 10,
    ):
        self.input_init_graph = init_graph
        self.init_graph = init_graph
        self.quartz_context = quartz_context
        self.num_xfers = quartz_context.num_xfers
        self.max_steps_per_episode = max_steps_per_episode
        self.nop_policy = nop_policy
        self.game_over_when_better = game_over_when_better
        self.max_additional_gates = max_additional_gates

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
            if (
                self.cur_step + 1 >= self.max_steps_per_episode
                or next_state.gate_count
                > self.input_init_graph.gate_count + self.max_additional_gates
            ):
                game_over = True

        # handle NOP
        if self.quartz_context.xfer_id_is_nop(xfer_id=action_xfer):
            game_over = self.nop_policy[0] == 's'
            this_step_reward = self.nop_policy[1]

        if self.game_over_when_better and this_step_reward > 0:
            game_over = True  # TODO  only reduce gate count once in each episode
            # TODO  note this case: 58 -> 80 -> 78

        self.cur_step += 1
        self.state = next_state

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

    def to(self, device: str):
        self.device = device

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
        mode: str = 'unknown',
        test_topk: int = 3,
        output_dir: str = 'output_dir',
        pretrained_weight_path: str = None,
        # envs
        init_graph_qasm_str: str = '',
        gate_set: List = ['h', 'cx', 't', 'tdg'],
        gate_type_num: int = 26,
        ecc_file: str = 'bfs_verified_simplified.json',
        no_increase: bool = True,
        include_nop: bool = False,
        nop_policy: list = ['c', 0.0],
        gamma: float = 0.9,
        episode_length: int = 60,
        max_additional_gates: int = 10,
        # module
        lr: float = 1e-3,
        scheduler: str = None,
        batch_size: int = 128,
        eps_init: float = 0.5,
        eps_decay: float = 0.0001,
        eps_min: float = 0.05,
        target_update_interval: int = 100,
        agent_play_interval: int = 4,
        warm_start_steps: int = 512,
        replaybuf_size: int = 10_000,
        prioritized_buffer: bool = True,
        prio_alpha: float = 0.2,
        prio_init_beta: float = 0.6,
        double_dqn: bool = False,
        # how to play
        agent_episode: bool = True,
        strict_better: bool = True,
        restart_from_best: bool = False,
        sample_init: bool = False,
        init_state_buf_size: int = 256,
        # network
        qgnn_h_feats: int = 64,
        qgnn_inter_dim: int = 64,
        # deprecated
        restore_weight_after_better: bool = False,
        game_over_when_better: bool = False,
        clear_buf_after_better: bool = False,
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
        assert len(init_graph_qasm_str) > 0
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
        self.loss_fn = nn.MSELoss(
            reduction='none' if self.hparams.prioritized_buffer else 'mean'
        )

        self.env = Environment(
            init_graph=init_graph,
            quartz_context=quartz_context,
            max_steps_per_episode=episode_length,
            nop_policy=nop_policy,
            game_over_when_better=game_over_when_better,
            max_additional_gates=max_additional_gates,
        )
        # we will set device for agent in on_after_batch_transfer latter
        self.agent = Agent(self.env, self.device)
        self.buffer = ReplayBuffer(
            replaybuf_size, self.device, self.hparams.prioritized_buffer
        )
        self.init_state_buffer = InitStateBuffer(
            capacity=init_state_buf_size, init_state=init_graph
        )
        self.best_graph = init_graph
        self.episode_best_gc = 0x7FFFFFFF

        self.eps = eps_init
        self.prio_beta = prio_init_beta
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

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        self.agent.to(self.device)
        self.buffer.to(self.device)
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def _output_seq(self, exp_seq: list = None, choices: list = None):
        # output sequence
        self.num_out_graphs += 1
        out_dir = os.path.join(
            self.hparams.output_dir,
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

    def save_ckpt(self, name: str):
        self.trainer.save_checkpoint(
            filepath=os.path.join(self.hparams.output_dir, f'{name}.ckpt')
        )

    def agent_step(self, eps: float) -> Tuple[Experience, int]:
        exp = self.agent.play_step(self.q_net, eps)
        self.buffer.append(exp)
        env_cur_step = self.env.cur_step

        if self.hparams.sample_init and exp.next_state and np.random.random() < 1:
            """fill init state buffer"""
            self.init_state_buffer.append(exp.next_state)

        """maintain the best graph info"""
        if exp.next_state and exp.next_state.gate_count < self.episode_best_gc:
            self.episode_best_gc = exp.next_state.gate_count
            if exp.next_state.gate_count < self.best_graph.gate_count:
                # a better graph is found
                print(
                    f'\nBetter graph with gate_count {exp.next_state.gate_count} found!'
                    f' method: {self.agent.choices[-1][0]}, eps: {self.agent.choices[-1][1]: .3f}, q: {self.agent.choices[-1][2]: .3f}'
                )
                if not self.hparams.strict_better or self.agent.choices[-1][0] == 'q':
                    info = f'!!! Best graph updated to {exp.next_state.gate_count} .'
                    print(info)
                    self.best_graph = exp.next_state
                    self._output_seq()
                    self.save_ckpt(
                        f'best_{exp.next_state.gate_count}_step_{self.global_step}'
                    )
                    wandb.alert(
                        title='Better graph is found!',
                        text=info,
                        level=wandb.AlertLevel.INFO,
                        wait_duration=0,
                    )
            # end if
        # end if

        if exp.game_over:
            """reset env"""
            self.episode_best_gc = 0x7FFFFFFF
            if self.hparams.sample_init:
                # sample a init state
                init_state = self.init_state_buffer.sample()
                self.env.set_init_state(init_state)
            elif self.hparams.restart_from_best:
                # start from the best graph
                self.env.set_init_state(self.best_graph)
            else:  # default setting: just return to the init state
                self.env.reset()

            self.agent.clear_choices()

            # below are deprecated
            if self.hparams.clear_buf_after_better:
                # clear and re-populate
                self.buffer.buffer.clear()
                # may meet gate count reduction again, which introduces recursions
                self.populate(self.hparams.warm_start_steps)
            if self.hparams.restore_weight_after_better:
                self._restore_pretrained_weight()
        # end if
        return exp, env_cur_step

    def agent_episode(self, eps: float) -> Tuple[Experience, int, float]:
        acc_reward = 0
        self.episode_best_gc = 0x7FFFFFFF
        while True:
            exp, env_cur_step = self.agent_step(eps)
            acc_reward += exp.reward
            if exp.game_over is True:
                break
        return exp, env_cur_step, acc_reward

    def populate(self, steps: int = 1000):
        """
        Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.
        Args:
            steps: number of random steps to populate the buffer
        """
        for i in tqdm(range(steps), desc='Populating the buffer'):
            self.agent_step(1.0)

    def _compute_loss(self) -> torch.Tensor:
        exps, indices, normed_prios, min_normed_prio = self.buffer.sample(
            self.hparams.batch_size
        )
        """prepare batched data"""
        states, action_nodes, action_xfers, rewards, next_states, game_overs = zip(
            *exps
        )
        b_state = dgl.batch([state.to_dgl_graph() for state in states]).to(self.device)
        b_next_state = dgl.batch(
            [
                state.to_dgl_graph() if state is not None else dgl.DGLGraph()
                for state in next_states
            ]
        ).to(self.device)
        cur_num_nodes = b_state.batch_num_nodes().tolist()
        next_num_nodes = b_next_state.batch_num_nodes().tolist()
        rewards = torch.Tensor(rewards).to(self.device)
        action_nodes = list(action_nodes)
        """predict Q values"""
        # ( sum(cur_num_nodes), num_xfers )
        pred_q_values = self.q_net(b_state)
        """predict max next Q values"""
        if max(next_num_nodes) > 0:
            with torch.no_grad():
                # ( sum(next_num_nodes), num_actions)
                next_q_values_tnet = self.target_net(b_next_state)
                if self.hparams.double_dqn:
                    next_q_values_qnet = self.q_net(b_next_state)
            if self.hparams.double_dqn:
                pass
            else:
                # ( sum(next_num_nodes), )
                next_max_q_values_tnet, _ = torch.max(next_q_values_tnet, dim=1)
        # compute max q values for next states
        target_next_q_values = []
        r_next_start, r_next_end = 0, 0
        r_cur_start, r_cur_end = 0, 0
        presum_cur_num_nodes = 0
        for i_batch in range(len(cur_num_nodes)):
            r_next_end += next_num_nodes[i_batch]
            r_next = slice(r_next_start, r_next_end)
            r_cur_end += cur_num_nodes[i_batch]
            r_cur = slice(r_cur_start, r_cur_end)
            # NOTE: zero if game over
            if game_overs[i_batch]:
                target_next_q = torch.tensor(0.0).to(self.device)
            else:
                # each elem has size [] (0-dim tensor)
                if self.hparams.double_dqn:
                    """double DQN"""
                    # Ref: https://arxiv.org/pdf/1509.06461.pdf
                    max_q, opt_action = topk_2d(next_q_values_qnet[r_next, :], k=1)
                    opt_action = opt_action[0]
                    target_next_q = next_q_values_tnet[r_next, :][
                        opt_action[0], opt_action[1]
                    ]
                else:
                    target_next_q = torch.max(next_max_q_values_tnet[r_next])
            target_next_q_values.append(target_next_q)
            action_nodes[i_batch] += presum_cur_num_nodes
            presum_cur_num_nodes += cur_num_nodes[i_batch]

            r_next_start = r_next_end
            r_cur_start = r_cur_end
        # (batch_size, )
        target_next_q_values = torch.stack(target_next_q_values)
        acted_pred_q_values = pred_q_values[action_nodes, action_xfers]
        target_q_values = rewards + self.hparams.gamma * target_next_q_values
        if self.hparams.prioritized_buffer:
            # Ref: https://github.com/Curt-Park/rainbow-is-all-you-need
            """compute element-wise loss and weighted loss"""
            # pred_Q = reward_of_action + gamma * target_next_max_Q
            elementwise_loss = self.loss_fn(acted_pred_q_values, target_q_values)
            loss_weights = (len(self.buffer) * normed_prios) ** (-self.prio_beta)
            max_loss_weights = (len(self.buffer) * min_normed_prio) ** (-self.prio_beta)
            normed_loss_weights = loss_weights / max_loss_weights
            loss = torch.mean(normed_loss_weights * elementwise_loss)
            """compute priorities and update them to the buffer"""
            prios = (elementwise_loss.detach() + 1e-6) ** self.hparams.prio_alpha
            self.buffer.update_prios(indices, prios)
        else:
            loss = self.loss_fn(acted_pred_q_values, target_q_values)

        self.log_dict(
            {
                f'mean_batch_reward': rewards.mean(),
                f'mean_target_next_max_Q': target_next_q_values.mean(),
                f'mean_target_max_Q': target_q_values.mean(),
                f'mean_pred_Q': acted_pred_q_values.mean(),
                f'max_batch_reward': rewards.max(),
                f'max_target_next_max_Q': target_next_q_values.max(),
                f'max_target_max_Q': target_q_values.max(),
                f'max_pred_Q': acted_pred_q_values.max(),
            },
            on_step=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        """
        1. Carries out an episode to add some experiences to the replay buffer.
        2. GD on the q_net with a batch of sampled data

        Args:
            batch: dummy data, not used
        """
        """sample a batch of data from the buffer to update the network"""
        # schedule prio_beta
        self.prio_beta = (
            self.hparams.prio_init_beta
            + (1.0 - self.hparams.prio_init_beta) * self.global_step / 1e5
        )
        loss = self._compute_loss()

        """play an episode"""
        self.eps = max(self.eps - self.hparams.eps_decay, self.hparams.eps_min)
        # if self.hparams.agent_episode:
        if self.global_step % self.hparams.agent_play_interval == 0:
            last_exp, tot_steps, acc_reward = self.agent_episode(self.eps)
            self.log_dict(
                {
                    f'eps': self.eps,
                    f'episode_steps': tot_steps,
                    f'episode_reward': acc_reward,
                    f'prio_beta': self.prio_beta,
                    f'lr': self.optimizers().param_groups[0]['lr'],
                },
                on_step=True,
            )

        if (
            self.global_step
            and self.global_step % self.hparams.target_update_interval == 0
        ):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.log(f'train_loss', loss)
        self.log_dict(
            {
                f'best_gc': self.best_graph.gate_count,
                f'init': self.env.init_graph.gate_count,
                f'|buf|': len(self.buffer.buffer),
            },
            on_step=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.q_net.parameters(),
            lr=self.hparams.lr,
        )
        if self.hparams.scheduler == 'reduce':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=10000,
                factor=0.2,
                threshold=0.005,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss',
                    'strict': False,  # consider resuming from checkpoint
                },
            }
        else:
            return optimizer

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = DummyDataset(size=self.hparams.target_update_interval)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
        )
        return dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = DummyDataset(size=1)
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
            self.hparams.output_dir,
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
        mode=cfg.mode,
        test_topk=cfg.test_topk,
        output_dir=output_dir,
        pretrained_weight_path=cfg.pretrained_weight if cfg.load_pretrained else None,
        init_graph_qasm_str=init_graph_qasm_str,
        gate_set=cfg.gate_set,
        ecc_file=cfg.ecc_file,
        no_increase=cfg.no_increase,
        include_nop=cfg.include_nop,
        nop_policy=cfg.nop_policy,
        gamma=cfg.gamma,
        episode_length=cfg.episode_length,
        max_additional_gates=cfg.max_additional_gates,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
        batch_size=cfg.batch_size,
        target_update_interval=cfg.target_update_interval,
        agent_play_interval=cfg.agent_play_interval,
        warm_start_steps=cfg.batch_size * 2,
        replaybuf_size=cfg.replaybuf_size,
        prioritized_buffer=cfg.prioritized_buffer,
        prio_alpha=cfg.prio_alpha,
        prio_init_beta=cfg.prio_init_beta,
        double_dqn=cfg.double_dqn,
        agent_episode=cfg.agent_episode,
        strict_better=cfg.strict_better,
        restart_from_best=cfg.restart_from_best,
        sample_init=cfg.sample_init,
        init_state_buf_size=cfg.init_state_buf_size,
        qgnn_h_feats=cfg.qgnn_h_feats,
        qgnn_inter_dim=cfg.qgnn_inter_dim,
    )

    # TODO  how to resume RL training perfectly? how to save the state and buffer?

    if cfg.mode == 'train':
        train(cfg)
    elif cfg.mode == 'test':
        test(cfg)
    else:
        raise ValueError(f'Invalid mode: {cfg.mode}')


if __name__ == '__main__':
    main()
