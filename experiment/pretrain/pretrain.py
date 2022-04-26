
import os
import json
import random
import torch
import pytorch_lightning as pl

import quartz


# class QGNNPretrainDM(pl.LightningDataModule):

#     def __init__(self,
#         dataset_dir: str = 'dataset',
#         graph_file: str = 'graph.json',
#         reward_file: str = 'reward.json',
#         gate_set: list = ['h', 'cx', 't', 'tdg'],
#         ecc_file: str = 'bfs_verified_simplified.json',
#         split_file: str = 'split.json',
#     ):
#         super().__init__()
#         self.dataset_dir = dataset_dir
#         self.graph_file = graph_file
#         self.reward_file = reward_file
#         self.gate_set = gate_set
#         self.ecc_file = ecc_file
#         self.split_file = split_file

#     def train_dataloader(self):
#         return self._get_dataloader('train')

#     def val_dataloader(self):
#         return self._get_dataloader('val')

#     def test_dataloader(self):
#         return self._get_dataloader('test')
    
#     def _get_dataloader(self, mode: str):
#         return QGNNPretrainDS(
#             dataset_dir=self.dataset_dir,
#             graph_file=self.graph_file,
#             reward_file=self.reward_file,
#             gate_set=self.gate_set,
#             ecc_file=self.ecc_file,
#             split_file=self.split_file,
#             mode=mode
#         )

class QGNNPretrainDM(pl.LightningDataModule):

    def __init__(self,
        dataset_dir: str = 'dataset',
        graph_file: str = 'graph.json',
        reward_file: str = 'reward.json',
        gate_set: list = ['h', 'cx', 't', 'tdg'],
        ecc_file: str = 'bfs_verified_simplified.json',
        split_file: str = 'split.json',
        mode: str = 'train',
        batch_size: int = 4
    ):
        super().__init__()
        self.hash2graphs = {}
        self.split_info = {}
        self.num_xfers = 0
        self.max_gate_count = 0
        self.batch_size = batch_size

        # load graphs and rewards
        with open(os.path.join(dataset_dir, graph_file)) as f:
            hash2graphs: dict = json.load(f) # hash -> (graph_qasm, gate_count)
        with open(os.path.join(dataset_dir, reward_file)) as f:
            rewards: dict = json.load(f) # hash -> { node_id: { xfer_id: reward } }
        
        split_file_path = os.path.join(dataset_dir, split_file)
        if not os.path.exists(split_file_path):
            # generate and save split info
            graph_keys = list(rewards.keys())
            random.shuffle(graph_keys)
            num_train = int(0.7 * len(graph_keys))
            num_val = int(0.1 * len(graph_keys))
            num_test = int(0.2 * len(graph_keys))
            self.split_info = {
                'train': graph_keys[ : num_train],
                'val': graph_keys[num_train : num_train + num_val],
                'test': graph_keys[num_train + num_val : ],
            }
            with open(split_file_path, 'w') as f:
                json.dump(self.split_info, fp=f, indent=2)
        else:
            with open(split_file_path) as f:
                self.split_info = json.load(f)

        # only use this context to convert qasm to graphs
        quartz_context = quartz.QuartzContext(
            gate_set=gate_set,
            filename=ecc_file,
            # no_increase=no_increase, # TODO  no need to specify
            include_nop=False, # TODO
        )
        self.num_xfers = quartz_context.num_xfers
        parser = quartz.PyQASMParser(context=quartz_context)

        for (g_hash, (qasm_str, gate_count)) in hash2graphs.items():
            pydag = parser.load_qasm_str(qasm_str)
            pygraph = quartz.PyGraph(context=quartz_context, dag=pydag)
            self.hash2graphs[g_hash] = (pygraph, rewards[g_hash], gate_count)
            self.max_gate_count = max(self.max_gate_count, gate_count)


    def train_dataloader(self):
        return self._get_dataloader(self.split_info['train'])

    def val_dataloader(self):
        return self._get_dataloader(self.split_info['val'])

    def test_dataloader(self):
        return self._get_dataloader(self.split_info['test'])
    
    def _get_dataloader(self, mode: str):
        default_collate = torch.utils.data.dataloader.default_collate
        # Ref: https://pytorch.org/docs/master/notes/randomness.html#dataloader
        g = torch.Generator()
        g.manual_seed(0)
        dataset = QGNNPretrainDS(
            hash2dglgraphs=self.hash2dglgraphs,
            hashes=self.split_info[mode],
            num_xfers=self.num_xfers,
            max_gate_count=self.max_gate_count,
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=8,
            batch_size=self.batch_size,
            shuffle=(mode == 'train'),
            generator=g,
        )
        return dataloader

class QGNNPretrainDS(torch.utils.data.Dataset):

    def __init__(self,
        hash2dglgraphs: dict,
        hashes: list,
        num_xfers: int,
        max_gate_count: int,
    ):
        super().__init__()
        self.num_xfers = num_xfers
        self.max_gate_count = max_gate_count
        self.hash2dglgraphs = [
            (*hash2dglgraphs[g_hash], g_hash)
            for g_hash in hashes
        ]
    
    def __len__(self):
        return len(self.hash2dglgraphs)

    def __getitem__(self, idx):
        '''
        return (dgl_graph, reward_mat, mask_mat)
        '''
        pygraph, reward_dict, gate_count, g_hash = self.hash2dglgraphs[idx]
        # { node_id: { xfer_id: reward } }
        # (num_nodes, num_xfers)
        reward_mat = torch.zeros(self.max_gate_count, self.num_xfers)
        mask_mat = torch.zeros(self.max_gate_count, self.num_xfers, dtype=torch.bool)
        zero_reward_mat = torch.zeros(self.max_gate_count, self.num_xfers, dtype=torch.bool)
        neg_reward_mat = torch.ones(self.max_gate_count, self.num_xfers, dtype=torch.bool)

        num_pos_rewards = 0
        for (node_id, xfers) in reward_dict.items():
            for (xfer_id, reward) in xfers.items():
                reward_mat[node_id][xfer_id] = reward
                # use all of the positive rewards
                mask_mat[node_id][xfer_id] = reward > 0
                zero_reward_mat[node_id][xfer_id] = reward == 0
                neg_reward_mat[node_id][xfer_id] = False
                num_pos_rewards += reward > 0

        zero_reward_indices = zero_reward_mat.nonzero()
        neg_reward_indices = neg_reward_mat.nonzero()
        # use part of the zero or negative rewards
        zero_reward_indices = zero_reward_indices[
            torch.randperm(zero_reward_indices.shape[0])[:num_pos_rewards]
        ]
        neg_reward_indices = neg_reward_indices[
            torch.randperm(neg_reward_indices.shape[0])[:num_pos_rewards]
        ]

        mask_mat[zero_reward_indices] = True
        mask_mat[neg_reward_indices] = True

        return (pygraph.to_dgl_graph(), reward_mat, mask_mat)


        




        

