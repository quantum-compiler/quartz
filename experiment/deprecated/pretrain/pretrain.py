import argparse
import json
import os
import random
import sys
import time
from math import gamma

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

quartz_context: quartz.QuartzContext = None
output_dir: str = None


def seed_all(seed: int):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class QGNNPretrainDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = 'dataset',
        graph_file: str = 'graph.json',
        reward_file: str = 'reward.json',
        include_nop: bool = False,
        split_file: str = 'split.json',
        batch_size: int = 128,
        use_max_gate_count: bool = False,
        gamma: float = 0.99,
        num_workers: int = 4,
    ):
        super().__init__()
        self.hash2graphs = {}
        self.split_info = {}
        self.num_xfers = 0
        self.max_gate_count = 0
        self.batch_size = batch_size
        self.use_max_gate_count = use_max_gate_count
        self.gamma = gamma
        self.num_workers = num_workers

        # load graphs and rewards
        with open(os.path.join(dataset_dir, graph_file)) as f:
            hash2graphs: dict = json.load(f)  # hash -> (graph_qasm, gate_count)
        with open(os.path.join(dataset_dir, reward_file)) as f:
            rewards: dict = json.load(f)  # hash -> { node_id: { xfer_id: reward } }

        split_file_path = os.path.join(dataset_dir, split_file)

        def gen_save_split():
            # generate and save split info
            graph_keys = list(rewards.keys())
            random.shuffle(graph_keys)
            num_train = int(0.7 * len(graph_keys))
            num_val = int(0.1 * len(graph_keys))
            num_test = int(0.2 * len(graph_keys))
            self.split_info = {
                'train': graph_keys[:num_train],
                'val': graph_keys[num_train : num_train + num_val],
                'test': graph_keys[num_train + num_val :],
            }
            with open(split_file_path, 'w') as f:
                json.dump(self.split_info, fp=f, indent=2)

        if not os.path.exists(split_file_path):
            gen_save_split()
        else:
            with open(split_file_path) as f:
                self.split_info = json.load(f)
            graphs_in_split_info = sum(
                [len(hashes) for mode, hashes in self.split_info.items()]
            )
            if graphs_in_split_info != len(rewards):
                gen_save_split()

        with open(os.path.join(output_dir, 'split.json'), 'w') as f:
            json.dump(self.split_info, fp=f, indent=2)

        self.num_xfers = quartz_context.num_xfers
        parser = quartz.PyQASMParser(context=quartz_context)

        # TODO  speed up by parallelism
        for (g_hash, xfers) in tqdm(rewards.items()):
            if include_nop is False:
                # filter no-op away
                xfers = {
                    node_id: {
                        xfer_id: reward
                        for xfer_id, reward in xfer_dict.items()
                        if int(xfer_id) < quartz_context.num_xfers  # filter nop
                    }
                    for node_id, xfer_dict in xfers.items()
                }
            graph_qasm, gate_count = hash2graphs[g_hash]
            pydag = parser.load_qasm_str(graph_qasm)
            pygraph = quartz.PyGraph(context=quartz_context, dag=pydag)
            self.hash2graphs[g_hash] = (pygraph, xfers, gate_count)
            self.max_gate_count = max(self.max_gate_count, gate_count)

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('val')

    def test_dataloader(self):
        return self._get_dataloader('test')

    def _get_dataloader(self, mode: str):
        default_collate = torch.utils.data.dataloader.default_collate

        def collate_fn(batch):
            # batch [ (dgl_graph, reward_mat, mask_mat) ]
            # s_time = time.time_ns()
            dgl_graphs, reward_mats, mask_mats = list(zip(*batch))
            batched_graphs = dgl.batch(dgl_graphs)
            if self.use_max_gate_count:
                reward_mats = default_collate(reward_mats)
                mask_mats = default_collate(mask_mats)
            else:
                reward_mats = torch.cat(reward_mats, dim=0)
                mask_mats = torch.cat(mask_mats, dim=0)
            # e_time = time.time_ns()
            # print(f'duration: { (e_time - s_time) / 1e6 } ms')
            return (batched_graphs, reward_mats, mask_mats)

        # Ref: https://pytorch.org/docs/master/notes/randomness.html#dataloader
        g = torch.Generator()
        g.manual_seed(0)
        dataset = QGNNPretrainDS(
            hash2graphs=self.hash2graphs,
            hashes=self.split_info[mode],
            num_xfers=self.num_xfers,
            max_gate_count=self.max_gate_count,
            use_max_gate_count=self.use_max_gate_count,
            gamma=self.gamma,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=(mode == 'train'),
            collate_fn=collate_fn,
            generator=g,
        )
        return dataloader


class QGNNPretrainDS(torch.utils.data.Dataset):
    def __init__(
        self,
        hash2graphs: dict,
        hashes: list,
        num_xfers: int,
        max_gate_count: int,
        use_max_gate_count: bool = False,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.num_xfers = num_xfers
        self.max_gate_count = max_gate_count
        self.use_max_gate_count = use_max_gate_count
        self.gamma = gamma
        self.graph_hash_list = [(*hash2graphs[g_hash], g_hash) for g_hash in hashes]

    def __len__(self):
        return len(self.graph_hash_list)

    def __getitem__(self, idx):
        """
        Return: (dgl_graph, reward_mat, mask_mat)
        """
        # s_time = time.time_ns()
        pygraph, reward_dict, gate_count, g_hash = self.graph_hash_list[idx]
        gate_count_to_use = (
            self.max_gate_count if self.use_max_gate_count else gate_count
        )
        # { node_id: { xfer_id: reward } }
        # (num_nodes, num_xfers)
        reward_mat = torch.zeros(gate_count_to_use, self.num_xfers)
        mask_mat = torch.zeros(gate_count_to_use, self.num_xfers, dtype=torch.bool)
        zero_reward_mat = torch.zeros(
            gate_count_to_use, self.num_xfers, dtype=torch.bool
        )
        neg_reward_mat = torch.ones(gate_count_to_use, self.num_xfers, dtype=torch.bool)

        num_pos_rewards = 0
        for (node_id, xfers) in reward_dict.items():
            for (xfer_id, reward) in xfers.items():
                if isinstance(reward, list):
                    reward = [pair[0] * self.gamma ** pair[1] for pair in reward]
                    reward = max(reward)
                node_id = int(node_id)
                xfer_id = int(xfer_id)  # TODO  why json dump/load int as str
                reward_mat[node_id][xfer_id] = float(reward)
                # use all of the positive rewards
                mask_mat[node_id][xfer_id] = reward > 0
                zero_reward_mat[node_id][xfer_id] = reward == 0
                neg_reward_mat[node_id][xfer_id] = False  # reward < 0
                num_pos_rewards += reward > 0

        # TODO  num_pos_rewards should have a min value
        num_pos_rewards = max(num_pos_rewards, 10)
        # (?, 2)
        zero_reward_indices = zero_reward_mat.nonzero()
        neg_reward_indices = neg_reward_mat.nonzero()
        # set negtive reward values
        reward_mat[neg_reward_indices[:, 0], neg_reward_indices[:, 1]] = -2

        # use part of the zero or negative rewards
        # we select them randomly here so they are different for each epochs
        zero_reward_indices = zero_reward_indices[
            torch.randperm(zero_reward_indices.shape[0])[:num_pos_rewards]
        ]  # (num_pos_rewards, 2)
        neg_reward_indices = neg_reward_indices[
            torch.randperm(neg_reward_indices.shape[0])[:num_pos_rewards]
        ]
        # set mask to select
        mask_mat[zero_reward_indices[:, 0], zero_reward_indices[:, 1]] = True
        mask_mat[neg_reward_indices[:, 0], neg_reward_indices[:, 1]] = True

        # e_time = time.time_ns()
        # print(f'duration: { (e_time - s_time) / 1e6 } ms')
        return (pygraph.to_dgl_graph(), reward_mat, mask_mat)


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


class PretrainNet(pl.LightningModule):
    def __init__(
        self,
        num_xfers: int,
        acc_interval: int = 32,
        scheduler: str = '',
        qgnn_h_feats: int = 64,
        qgnn_inter_dim: int = 64,
        acc_topk: int = 3,
        use_mask: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        gate_type_num = 26
        self.q_net = QGNN(
            in_feats=gate_type_num,
            h_feats=qgnn_h_feats,
            num_classes=num_xfers,
            inter_dim=qgnn_inter_dim,
        )
        self.loss_fn = nn.MSELoss(reduction='sum')

    def _common_step(self, batch, mode: str = 'unknwon'):
        """
        Args: batch contains:
            batched_graph: batch_num_nodes: sum(batched_graph.batch_num_nodes())
            gt_rewards: (bs, batch_num_nodes, num_xfers)
            masks: (bs, batch_num_nodes, num_xfers)
        """
        batched_graph, gt_rewards, masks = batch
        num_nodes = batched_graph.batch_num_nodes()
        num_nodes = [int(n) for n in num_nodes]
        # out: ( sum(num of nodes), num_xfers )
        out = self.q_net(batched_graph)

        loss = self._compute_loss(out, gt_rewards, masks, mode)
        if mode in ['val', 'test'] or self.global_step % self.hparams.acc_interval == 0:
            batch_acc = self._compute_acc(
                out, gt_rewards, num_nodes, topk=self.hparams.acc_topk, prefix=mode
            )
            print(f'{mode}: batch_acc {batch_acc} at globalstep {self.global_step}')

        # log some info
        r_start, r_end = 0, 0
        for i_batch, n_nodes in enumerate(num_nodes):
            if i_batch >= 1:
                break
            r_end += n_nodes
            r = slice(r_start, r_end)

            selected_indices = masks[r].nonzero()
            selected_rewards = gt_rewards[
                selected_indices[:, 0], selected_indices[:, 1]
            ]

            self.log(f'{mode}_num_nodes_{i_batch}', float(n_nodes), on_step=True)
            self.log(
                f'{mode}_num_unmasked_label_{i_batch}',
                float(selected_indices.shape[0]),
                on_step=True,
            )
            self.log(
                f'{mode}_pos_label_{i_batch}',
                float((selected_rewards > 0).sum()),
                on_step=True,
            )
            self.log(
                f'{mode}_zero_label_{i_batch}',
                float((selected_rewards == 0).sum()),
                on_step=True,
            )
            self.log(
                f'{mode}_neg_label_{i_batch}',
                float((selected_rewards < 0).sum()),
                on_step=True,
            )

            self.log(
                f'{mode}_max_reward_{i_batch}',
                torch.max(selected_rewards),
                on_step=True,
            )
            self.log(
                f'{mode}_min_reward_{i_batch}',
                torch.min(selected_rewards),
                on_step=True,
            )
            self.log(
                f'{mode}_mean_reward_{i_batch}',
                torch.mean(selected_rewards),
                on_step=True,
            )

            r_start = r_end

        # print learning rates
        if mode == 'train':
            optimizer = self.optimizers()
            self.log(
                f'lr',
                optimizer.param_groups[0]['lr'],
                prog_bar=True,
                on_step=True,
            )

        return loss

    def _compute_loss(self, out, gt_rewards, masks, prefix: str = ''):
        if self.hparams.use_mask:
            pred = out * masks
            label = gt_rewards * masks

            loss = self.loss_fn(pred, label)
            loss = loss / masks.sum()  # manually apply mean reduction
        else:
            loss = self.loss_fn(out, gt_rewards)  # sum reduce
            mse_loss = loss / (out.shape[0] * out.shape[1])
            # self.log(f'{prefix}_mse_loss', mse_loss)
            if loss > 0.01:
                loss = mse_loss

        self.log(f'{prefix}_loss', loss)

        return loss

    def _compute_acc(
        self, pred, gt, num_nodes: list, topk: int, prefix: str = ''
    ) -> torch.Tensor:
        def topk_2d(x: torch.Tensor, k: int, as_tuple: bool = False):
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

        # end def
        # s_time = time.time_ns()
        batch_right_action_count = torch.zeros(len(num_nodes), device='cpu')
        batch_pred_action_count = torch.zeros(len(num_nodes), device='cpu')
        best_pred_in_topk_gt = 0
        best_gt_in_topk_pred = 0
        valid_pred_actions = 0
        valid_best_pred_actions = 0
        r_start, r_end = 0, 0
        for i_batch, n_nodes in enumerate(num_nodes):
            """compute optimal actions for each graph"""
            r_end += n_nodes
            r = slice(r_start, r_end)
            i_pred, i_gt = pred[r], gt[r]

            gt_q_topk, gt_act_topk = topk_2d(i_gt, k=topk, as_tuple=False)
            pred_q_topk, pred_act_topk = topk_2d(i_pred, k=topk, as_tuple=False)

            # deal with the case that multiple elements has the same value
            gt_lower_bound, _ = torch.min(gt_q_topk, dim=0)
            gt_act_topk = (i_gt >= gt_lower_bound).nonzero()
            pred_lower_bound, _ = torch.min(pred_q_topk, dim=0)
            pred_act_topk = (i_pred >= pred_lower_bound).nonzero()

            """how well the top k actions matched"""
            num_right = sum([act in gt_act_topk for act in pred_act_topk])
            batch_right_action_count[i_batch] = num_right
            batch_pred_action_count[i_batch] = pred_act_topk.shape[0]

            """whether the best predicted action lies in the top k ground truth"""
            pred_q_best, _ = torch.max(pred_q_topk, dim=0)
            pred_act_best = (i_pred >= pred_q_best).nonzero()
            best_pred_in_topk_gt += (
                sum([act in gt_act_topk for act in pred_act_best]) > 0
            )

            """whether the best ground truth action lies in the top k predicted actions"""
            gt_q_best, _ = torch.max(gt_q_topk, dim=0)
            gt_act_best = (i_gt >= gt_q_best).nonzero()
            best_gt_in_topk_pred += (
                sum([act in pred_act_topk for act in gt_act_best]) > 0
            )

            """ratio of valid actions in top k predicted ones"""
            pred_topk_act_gt_q_values = i_gt[pred_act_topk[:, 0], pred_act_topk[:, 1]]
            valid_pred_actions += torch.sum(pred_topk_act_gt_q_values > -2, dim=0)

            pred_best_act_gt_q_values = i_gt[pred_act_best[:, 0], pred_act_best[:, 1]]
            valid_best_pred_actions += (
                torch.sum(pred_best_act_gt_q_values > -2, dim=0) > 0
            )

            r_start = r_end
        # end for
        num_pred_topk_actions = torch.sum(batch_pred_action_count, dim=0)
        topk_pred_in_topk_gt_acc = (
            torch.sum(batch_right_action_count, dim=0) / num_pred_topk_actions
        )
        best_pred_in_topk_gt_acc = best_pred_in_topk_gt / len(num_nodes)
        best_gt_in_topk_pred_acc = best_gt_in_topk_pred / len(num_nodes)
        valid_in_topk_pred_acc = valid_pred_actions / num_pred_topk_actions
        valid_of_best_pred_acc = valid_best_pred_actions / len(num_nodes)
        # e_time = time.time_ns()
        # print(f'duration: { (e_time - s_time) / 1e6 } ms')
        self.log_dict(
            {
                f'{prefix}_top{topk}_pred_in_top{topk}_gt_acc': topk_pred_in_topk_gt_acc,
                f'{prefix}_best_pred_in_top{topk}_gt_acc': best_pred_in_topk_gt_acc,
                f'{prefix}_best_gt_in_top{topk}_pred_acc': best_gt_in_topk_pred_acc,
                f'{prefix}_top{topk}_pred_valid_acc': valid_in_topk_pred_acc,
                f'{prefix}_best_pred_valid_acc': valid_of_best_pred_acc,
            },
            on_step=True,
        )
        return topk_pred_in_topk_gt_acc

    def _compute_acc_droped(
        self,
        pred,
        gt,
        num_nodes,
        xfer_topk: int = 1,
        node_topk: int = 1,
        prefix: str = '',
    ):
        # s_time = time.time_ns()
        def get_opt_xfers(q_values):
            n_nodes: int = q_values.shape[0]
            q_topk, _ = torch.topk(q_values, k=xfer_topk, dim=1)
            min_q_th, _ = torch.min(q_topk, dim=1)
            min_q_th = torch.unsqueeze(min_q_th, dim=1)
            opt_nodes, opt_xfers = torch.nonzero(q_values >= min_q_th, as_tuple=True)
            opt_xfers_per_node = [
                opt_xfers[opt_nodes == node_id] for node_id in range(n_nodes)
            ]
            return opt_xfers_per_node

        batch_action_right = []
        batch_xfer_acc_per_node = []
        r_start, r_end = 0, 0
        for i_batch, n_nodes in enumerate(num_nodes):
            """it takes more than 20 ms/it , which is very slow for large batch_size"""
            r_end += n_nodes
            r = slice(r_start, r_end)
            i_pred, i_gt = pred[r], gt[r]

            gt_q_topk_per_node, gt_xfer_topk_per_node = torch.topk(
                i_gt, k=xfer_topk, dim=1
            )
            gt_min_q_th_per_node, _ = torch.min(gt_q_topk_per_node, dim=1)
            gt_min_q_th_per_node = torch.unsqueeze(gt_min_q_th_per_node, dim=1)
            gt_opt_nodes, gt_opt_xfers = torch.nonzero(
                i_gt >= gt_min_q_th_per_node, as_tuple=True
            )
            gt_opt_xfers_per_node = [
                gt_opt_xfers[gt_opt_nodes == node_id] for node_id in range(n_nodes)
            ]

            pred_q_topk_per_node, pred_xfer_topk_per_node = torch.topk(
                i_pred, k=xfer_topk, dim=1
            )
            pred_min_q_th_per_node, _ = torch.min(pred_q_topk_per_node, dim=1)
            pred_min_q_th_per_node = torch.unsqueeze(pred_min_q_th_per_node, dim=1)
            pred_opt_nodes, pred_opt_xfers = torch.nonzero(
                i_pred >= pred_min_q_th_per_node, as_tuple=True
            )
            pred_opt_xfers_per_node = [
                pred_opt_xfers[pred_opt_nodes == node_id] for node_id in range(n_nodes)
            ]

            """compute acc of action (node_id, xfer_id) for the whole graph"""
            gt_q_max_per_node, _idxs1 = torch.max(gt_q_topk_per_node, dim=1)
            gt_xfer_max_per_node = torch.tensor(
                [gt_xfer_topk_per_node[i, xfer_id] for i, xfer_id in enumerate(_idxs1)]
            ).to(self.device)
            gt_q_max, gt_node_max = torch.max(gt_q_max_per_node, dim=0)
            gt_xfer_max = gt_xfer_max_per_node[gt_node_max]

            pred_q_max_per_node, _idxs2 = torch.max(pred_q_topk_per_node, dim=1)
            pred_xfer_max_per_node = torch.tensor(
                [
                    pred_xfer_topk_per_node[i, xfer_id]
                    for i, xfer_id in enumerate(_idxs2)
                ]
            ).to(self.device)
            pred_q_max, pred_node_max = torch.max(pred_q_max_per_node, dim=0)
            pred_xfer_max = pred_xfer_max_per_node[pred_node_max]

            batch_action_right.append(
                gt_node_max == pred_node_max and gt_xfer_max == pred_xfer_max
            )

            """compute acc of selecting node for the whole graph"""
            # TODO  min or max?
            gt_q_mean_topk_per_node = torch.mean(gt_q_topk_per_node, dim=1).flatten()
            pred_q_mean_topk_per_node = torch.mean(
                pred_q_topk_per_node, dim=1
            ).flatten()
            gt_opt_topk_q, gt_opt_topk_nodes = torch.topk(
                gt_q_mean_topk_per_node, k=node_topk, dim=0
            )
            pred_opt_topk_q, pred_opt_topk_nodes = torch.topk(
                pred_q_mean_topk_per_node, k=node_topk, dim=0
            )
            node_acc = torch.mean(
                torch.tensor(
                    [node_id in gt_opt_topk_nodes for node_id in pred_opt_topk_nodes],
                    device='cpu',
                ),
                dim=0,
            )

            """compute acc of selecting xfer per node"""
            for node_id in range(n_nodes):
                gt_opt_xfers: torch.tensor = gt_opt_xfers_per_node[node_id]
                pred_opt_xfers: torch.tensor = pred_opt_xfers_per_node[node_id]
                node_acc = torch.mean(
                    torch.tensor(
                        [xfer in gt_opt_xfers for xfer in pred_opt_xfers], device='cpu'
                    ),
                    dim=0,
                )  # float32
                batch_xfer_acc_per_node.append(node_acc)
            # end for
            r_start = r_end
        # end for
        batch_action_right = torch.mean(torch.tensor(batch_action_right), device='cpu')
        # mean xfer acc for all nodes in the batch
        batch_xfer_acc_per_node = torch.Tensor(batch_xfer_acc_per_node, device='cpu')
        batch_xfer_mean_acc = torch.mean(batch_xfer_acc_per_node, dim=0)
        self.log_dict(
            {
                f'{prefix}_batch_xfer_mean_acc': batch_xfer_mean_acc,
                f'{prefix}_batch_action_right': batch_action_right,
            },
            on_step=True,
        )

        # e_time = time.time_ns()
        # print(f'duration: { (e_time - s_time) / 1e6 } ms')
        return batch_xfer_mean_acc

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=list(self.q_net.parameters()),
            lr=1e-3,
        )
        if self.hparams.scheduler == 'reduce':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=128,
                factor=0.2,
                threshold=0.001,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss',  # TODO
                    'strict': False,  # TODO  when resuming from checkpoint
                },
            }
        else:
            return optimizer


def init_wandb(
    enable: bool = True,
    offline: bool = False,
    project: str = 'QGNN-Pretrain',
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


model: pl.LightningModule = None
datamodule: pl.LightningDataModule = None


def train(cfg):
    wandb_logger = init_wandb(
        enable=cfg.wandb.en,
        offline=cfg.wandb.offline,
        task='train',
        entity=cfg.wandb.entity,
    )
    ckpt_callback_list = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=output_dir,
            filename='{epoch}-{val_loss:.2f}-best',
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
        strategy=DDPStrategy(find_unused_parameters=False),
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        # gradient_clip_val=cfg.task.optimizer.clip_value,
        # gradient_clip_algorithm=cfg.task.optimizer.clip_algo,
        # val_check_interval=cfg.val_check_interval,
        # plugins=DDPPlugin(find_unused_parameters=False),
    )
    if cfg.resume is True:
        ckpt_path = cfg.ckpt_path
        assert os.path.exists(ckpt_path)
    else:
        ckpt_path = None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


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
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    global model
    global datamodule
    global output_dir
    global quartz_context

    output_dir = os.path.abspath(os.curdir)  # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd())  # set working dir to the original one
    seed_all(cfg.seed)

    # only use this context to convert qasm to graphs
    quartz_context = quartz.QuartzContext(
        gate_set=cfg.gate_set,
        filename=cfg.ecc_file,
        # we need to include xfers that lead to gate increase when training
        # we may exclude them when generating the dataset for pre-training
        no_increase=cfg.no_increase,
        include_nop=cfg.include_nop,  # TODO
    )

    datamodule = QGNNPretrainDM(
        dataset_dir=cfg.dataset_dir,
        graph_file=cfg.graph_file,
        reward_file=cfg.reward_file,
        include_nop=cfg.include_nop,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    PLModel = PretrainNet

    model = PLModel(
        num_xfers=quartz_context.num_xfers,
        acc_interval=cfg.acc_interval,
        scheduler=cfg.scheduler,
        qgnn_h_feats=cfg.qgnn_h_feats,
        qgnn_inter_dim=cfg.qgnn_inter_dim,
        acc_topk=cfg.acc_topk,
        use_mask=cfg.use_mask,
    )

    # if cfg.mode == 'train':
    #     if cfg.resume:
    #         assert os.path.exists(cfg.ckpt_path)
    #         # TODO  do not pass num_xfers
    #         model = PLModel.load_from_checkpoint(cfg.ckpt_path, datamodule.num_xfers)
    #     else:
    #         model = PLModel(datamodule.num_xfers)  # train from scratch
    # elif cfg.mode == 'test':
    #     if len(cfg.ckpt_path) > 0:
    #         assert os.path.exists(cfg.ckpt_path)
    #         # TODO  do not pass num_xfers
    #         model = PLModel.load_from_checkpoint(cfg.ckpt_path, datamodule.num_xfers)
    #     else:
    #         model = PLModel(datamodule.num_xfers)  # test from scratch
    # else:
    #     raise ValueError(f'Invalid mode: {cfg.mode}')

    if cfg.mode == 'train':
        train(cfg)
    elif cfg.mode == 'test':
        test(cfg)
    else:
        raise ValueError(f'Invalid mode: {cfg.mode}')


if __name__ == '__main__':
    main()
