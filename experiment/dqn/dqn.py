import os
import sys
import json
import math
import random
import warnings
from collections import deque, namedtuple
import heapq
from typing import List, Tuple
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import dgl

from tqdm import tqdm
import icecream as ic
from IPython import embed

import quartz

sys.path.append(os.path.join(os.getcwd(), '..'))
from pretrain.pretrain import PretrainNet

class SpecialMinHeap:
    def __init__(self, max_size = math.inf):
        self.arr= []
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
        sample_counts = sample_counts - sample_counts.min() # [0, 0, 1, 2, ...]
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
        w = torch.cat([
            torch.unsqueeze(g.edata['src_idx'], 1),
            torch.unsqueeze(g.edata['dst_idx'], 1),
            torch.unsqueeze(g.edata['reversed'], 1)
        ], dim=1)
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
    field_names=['state', 'action_node', 'action_xfer', 'reward', 'next_state', 'game_over'],
)

class ReplayBuffer:

    def __init__(self, capacity: int = 100):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def append(self, exp: Experience):
        self.buffer.append(exp)
    
    def sample(self, sample_size: int = 100) -> Tuple:
        indices = np.random.choice(
            len(self.buffer), min(sample_size, len(self.buffer)), replace=False)
        states, action_nodes, action_xfers, rewards, next_states, game_overs = \
            list(zip(*(self.buffer[idx] for idx in indices)))
        
        return (
            states, action_nodes, action_xfers, rewards, next_states, game_overs
        )

class RLDataset(torch.utils.data.dataset.IterableDataset):
    
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 100):
        self.buffer = buffer
        self.sample_size = sample_size
    
    def __iter__(self) -> Tuple:
        """
        Samples many items from buffer, but only yield one each time
        """
        states, action_nodes, action_xfers, rewards, next_states, game_overs = \
            self.buffer.sample(self.sample_size)
        for i in range(len(states)):
            yield (
                states[i], action_nodes[i], action_xfers[i], rewards[i], next_states[i], game_overs[i]
            )

class Environment:

    def __init__(self,
        init_graph: quartz.PyGraph,
        quartz_context: quartz.QuartzContext,
        max_steps_per_episode: int = 100,
        nop_policy: list = ['c', 0.0],
    ):
        self.init_graph = init_graph
        self.quartz_context = quartz_context
        self.num_xfers = quartz_context.num_xfers
        self.max_steps_per_episode = max_steps_per_episode
        self.nop_policy = nop_policy
        
        self.state = None
        self.cur_step = 0
        self.exp_seq = [ self._get_firtst_exp() ]

        self.reset()

    def reset(self):
        self.state: quartz.PyGraph = self.init_graph
        self.cur_step = 0
        self.exp_seq = [ self._get_firtst_exp() ]

    def set_init_state(self, graph: quartz.PyGraph):
        self.init_graph = graph
        self.state = graph
        self.cur_step = 0
        self.exp_seq = [ self._get_firtst_exp() ]

    def _get_firtst_exp(self) -> Experience:
        return Experience(
            None, 0, 0, 0, self.init_graph, False,
        )

    def get_action_space(self) -> Tuple[List[int], List[int]]:
        return (
            list(range(self.state.gate_count)),
            list(range(self.num_xfers))
        )
    
    def available_xfers(self, action_node: int) -> int:
        xfers = self.state.available_xfers(
            context=self.quartz_context,
            node=self.state.get_node_from_id(id=action_node)
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
            this_step_reward = -2 # TODO  neg reward when the xfer is invalid
        else:
            game_over = False
            this_step_reward = cur_state.gate_count - next_state.gate_count
        
        # handle NOP
        if self.quartz_context.xfer_id_is_nop(xfer_id=action_xfer):
        # if action_xfer == self.quartz_context.num_xfers:
            game_over = (self.nop_policy[0] == 's')
            this_step_reward = self.nop_policy[1]
        
        self.cur_step += 1
        self.state = next_state
        if self.cur_step >= self.max_steps_per_episode:
            game_over = True
            # TODO  whether we need to change this_step_reward here?
        
        if this_step_reward > 0:
            game_over = True # TODO  only apply one xfer each episode
            # TODO  note this case: 58 -> 80 -> 78

        exp = Experience(cur_state, action_node, action_xfer, this_step_reward, next_state, game_over)
        self.exp_seq.append(exp);
        return exp

class Agent:

    def __init__(self,
        env: Environment,
        device: torch.device,
    ):
        self.env = env
        self.device = device

        self.choices = [ ('s', 0) ]

    @torch.no_grad()
    def _get_action(self, q_net: nn.Module, eps: float) -> Tuple[int, int]:
        if np.random.random() < (1 - eps):
            # greedy
            cur_state: dgl.graph = self.env.state.to_dgl_graph().to(self.device)
            # (num_nodes, num_actions)
            q_values = q_net(cur_state)
            max_q_values, optimal_xfers = torch.max(q_values, dim=1)
            q_value, optimal_node = torch.max(max_q_values, dim=0)
            optimal_xfer = optimal_xfers[optimal_node]
            # return values will be added into data buffer, which latter feeds the dataset
            # they cannot be cuda tensors, or CUDA error will occur in collate_fn
            self.choices.append( ('q', eps) )
            return optimal_node.item(), optimal_xfer.item()
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
            self.choices.append( ('r', eps) )
            return node, xfer

    @torch.no_grad()
    def play_step(self, q_net: nn.Module, eps: float) -> Experience:
        action_node, action_xfer = self._get_action(q_net, eps)
        exp = self.env.act(action_node, action_xfer)        
        return exp

    def clear_choices(self):
        self.choices = [ ('s', 0) ]


class DQNMod(pl.LightningModule):

    def __init__(self,
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
        episode_length: int = 100, # TODO  check these hparams
        replaybuf_size: int = 10_000,
        warm_start_steps: int = 512,
        target_update_interval: int = 100,
        seq_out_dir: str = 'out_graphs',
        pretrained_weight_path: str = None,
        restore_weight_after_better: bool = False,
        nop_policy: list = ['c', 0.0],
        restart_from_best: bool = False,
        clear_buf_after_better: bool = False,
        init_state_buf_size: int = 512,
        agent_episode: bool = True,
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
        self.num_xfers = quartz_context.num_xfers
        parser = quartz.PyQASMParser(context=quartz_context)
        init_dag = parser.load_qasm_str(init_graph_qasm_str)
        init_graph = quartz.PyGraph(context=quartz_context, dag=init_dag)
        self.init_graph = init_graph

        self.q_net = QGNN(self.hparams.gate_type_num, 64, quartz_context.num_xfers, 64)
        self.target_net = QGNN(self.hparams.gate_type_num, 64, quartz_context.num_xfers, 64)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.loss_fn = nn.MSELoss()

        self.env = Environment(
            init_graph=init_graph,
            quartz_context=quartz_context,
            max_steps_per_episode=episode_length,
            nop_policy=nop_policy,
        )
        # we will set device for agent in on_after_batch_transfer latter
        self.agent = Agent(self.env, self.device)
        self.buffer = ReplayBuffer(replaybuf_size)
        self.init_state_buffer = SpecialMinHeap(max_size=init_state_buf_size)
        self.init_state_buffer.push((init_graph.gate_count, init_graph))
        self.best_graph = init_graph

        self.eps = eps_init
        self.episode_reward = 0
        self.total_reward = 0
        self.num_out_graphs = 0
        self.pretrained_q_net = None

        self.load_pretrained_weight()
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

    def _output_seq(self):
        # output sequence
        self.num_out_graphs += 1
        out_dir = os.path.join(
            self.hparams.seq_out_dir,
            'out_graphs',
            f'{self.num_out_graphs}',
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i_step, exp in enumerate(self.env.exp_seq):
            out_path = os.path.join(
                out_dir,
                f'{i_step}_{exp.next_state.gate_count}_{exp.action_node}_{exp.action_xfer}_'
                f'{self.agent.choices[i_step][0]}_{self.agent.choices[i_step][1]:.3f}.qasm',
            ) # TODO  wrong filename
            qasm_str = exp.next_state.to_qasm_str()
            with open(out_path, 'w') as f:
                print(qasm_str, file=f)

    def agent_step(self, eps: float) -> Tuple[Experience, int]:
        exp = self.agent.play_step(self.q_net, eps)
        self.buffer.append(exp)
        env_cur_step = self.env.cur_step

        if exp.game_over and exp.next_state and exp.next_state.gate_count < self.best_graph.gate_count:
            # better graph found, add it to init_state_buffer
            self.init_state_buffer.push((self.init_graph.gate_count - exp.next_state.gate_count, exp.next_state))
        elif exp.next_state:
            if np.random.random() < 0.10 or len(self.init_state_buffer) < 10:
                self.init_state_buffer.push((self.init_graph.gate_count - exp.next_state.gate_count, exp.next_state))

        if exp.game_over:
            # TODO  if a better graph is found, clear the buffer and populate it with the new graph?
            if exp.next_state and \
                exp.next_state.gate_count < self.best_graph.gate_count:
                # a better graph is found
                self.best_graph = exp.next_state
                print(
                    f'\n!!! Better graph with gate_count {exp.next_state.gate_count} found!'
                    f' method: {self.agent.choices[-1][0]}, eps: {self.agent.choices[-1][1]: .3f}'
                )
                self._output_seq()
                # reset
                if self.hparams.restart_from_best:
                    # start from the best graph
                    self.env.set_init_state(exp.next_state)
                else:
                    # sample a init state
                    init_state_pair = self.init_state_buffer.sample(k=1)[0]
                    self.env.set_init_state(init_state_pair[1])
                
                self.agent.clear_choices()
                if self.hparams.clear_buf_after_better:
                    # clear and re-populate
                    self.buffer.buffer.clear()
                    # may meet gate count reduction again, which introduces recursions
                    self.populate(self.hparams.warm_start_steps)
                if self.hparams.restore_weight_after_better:
                    self._restore_pretrained_weight()
            else:
                self.agent.clear_choices()
                if self.hparams.restart_from_best:
                    self.env.reset()
                else:
                    # sample a init state
                    init_state_pair = self.init_state_buffer.sample(k=1)[0]
                    self.env.set_init_state(init_state_pair[1])
        # end if
        return exp, env_cur_step

    def agent_episode(self, eps: float) -> Tuple[Experience, int]:
        while True:
            exp, env_cur_step = self.agent_step(eps)
            if exp.game_over == True:
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
        states, action_nodes, action_xfers, rewards, next_states, _ = batch
        cur_num_nodes = states.batch_num_nodes().tolist()
        next_num_nodes = next_states.batch_num_nodes().tolist()

        # ( sum(num of nodes), num_xfers )
        pred_q_values = self.q_net(states)
        if max(next_num_nodes) > 0:
            target_next_q_values = self.target_net(next_states)
            # ( sum(num of nodes), )
            target_next_max_q_values_all_nodes, _ = torch.max(target_next_q_values, dim=-1)
        # pad neg rewards for empty graphs
        target_next_max_q_values = []
        next_num_nodes_fixed = []
        r_start, r_end = 0, 0
        for i_batch, next_num_node in enumerate(next_num_nodes):
            r_end += next_num_node
            r = slice(r_start, r_end)
            target_next_max_q_values.append(
                torch.max(target_next_max_q_values_all_nodes[r])
                if next_num_node > 0 else
                torch.Tensor([-2]) # TODO  neg reward
            )
            next_num_nodes_fixed.append(max(1, next_num_node))
            r_start = r_end
        # (batch_size, )
        target_next_max_q_values = torch.Tensor(
            target_next_max_q_values
        ).to(self.device)
        # pred_Q = reward_of_action + gamma * target_next_max_Q
        acted_pred_q_values = pred_q_values[action_nodes, action_xfers]
        target_max_q_values = rewards + self.hparams.gamma * target_next_max_q_values
        
        loss = self.loss_fn(acted_pred_q_values, target_max_q_values)

        self.log_dict({
            f'mean_batch_reward': rewards.mean(),
            f'mean_target_next_max_Q': target_next_max_q_values.mean(),
            f'mean_target_max_Q': target_max_q_values.mean(),
            f'mean_pred_Q': acted_pred_q_values.mean(),

            f'max_batch_reward': rewards.max(),
            f'max_target_next_max_Q': target_next_max_q_values.max(),
            f'max_target_max_Q': target_max_q_values.max(),
            f'max_pred_Q': acted_pred_q_values.max(),
        }, on_step=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        1. Carries out a single step to add one experience to the replay buffer.
        2. GD on the q_net with a batch of data

        Args:
            batch: (states, action_nodes, action_xfers, rewards, next_states, game_overs)
        """
        # play one step
        self.eps = max(
            self.eps - self.hparams.eps_decay,
            self.hparams.eps_min
        )
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

        if self.global_step and self.global_step % self.hparams.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.log(f'train_loss', loss)
        self.log(f'step_reward', exp.reward, on_step=True, prog_bar=True)
        self.log(f'total_reward', self.total_reward, on_step=True)
        self.log(f'eps', self.eps, on_step=True)
        self.log(f'env_step', env_cur_step, on_step=True, prog_bar=True)
        self.log(f'best_gc', self.best_graph.gate_count, on_step=True, prog_bar=True)
        self.log(f'init', self.env.init_graph.gate_count, on_step=True, prog_bar=True)
        self.log(f'|buf|', len(self.buffer.buffer), on_step=True, prog_bar=True)
        self.log(f'|init_state_buf|', len(self.init_state_buffer), on_step=True, prog_bar=True)

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
                    b_states: dgl.graph (batched_graph)
                    b_action_nodes: torch.tensor
                    b_action_xfers: torch.tensor
                    b_rewards: torch.Tensor
                    b_next_states: dgl.graph (batched_graph)
                    b_game_overs: torch.tensor of 
            """
            states, action_nodes, action_xfers, rewards, next_states, game_overs = \
                list(zip(*batch))
            states = [state.to_dgl_graph() for state in states]
            next_states = [
                next_state.to_dgl_graph()
                if next_state is not None else dgl.DGLGraph()
                for next_state in next_states
            ]
            b_states = dgl.batch(states)
            b_next_states = dgl.batch(next_states)
            b_action_nodes = torch.tensor(action_nodes)
            b_action_xfers = torch.tensor(action_xfers)
            b_rewards = torch.Tensor(rewards)
            b_game_overs = torch.tensor(game_overs, dtype=torch.bool)
            return (
                b_states, b_action_nodes, b_action_xfers, b_rewards, b_next_states, b_game_overs
            )
        
        # Ref: https://pytorch.org/docs/master/notes/randomness.html#dataloader
        g = torch.Generator()
        g.manual_seed(0)
        
        dataset = RLDataset(self.buffer, self.hparams.episode_length * self.hparams.batch_size)
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

    # TODO  no val or test

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
    entity: str = ''
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
        enable=cfg.wandb.en, offline=cfg.wandb.offline,
        task='train', entity=cfg.wandb.entity
    )
    ckpt_callback_list = [
        ModelCheckpoint(
            monitor='train_loss', # TODO  other values?
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
        enable=cfg.wandb.en, offline=cfg.wandb.offline,
        task='test', entity=cfg.wandb.entity
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

    output_dir = os.path.abspath(os.curdir) # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd()) # set working dir to the original one

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
        restart_from_best=cfg.restart_from_best,
        clear_buf_after_better=cfg.clear_buf_after_better,
        agent_episode=cfg.agent_episode,
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
