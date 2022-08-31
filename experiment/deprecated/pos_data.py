import json
import random

from dgl import load_graphs, save_graphs


class PosRewardData:
    def __init__(self, maxlen=500) -> None:
        self.from_graphs = []
        self.from_graph_hashes = []
        self.node_ids = []
        self.xfer_ids = []
        self.rewards = []
        self.to_graphs = []
        self.to_graph_hashes = []
        self.data_cnt = 0
        self.maxlen = maxlen

    def add_data(
        self,
        from_graph,
        from_graph_hash,
        node_id,
        xfer_id,
        reward,
        to_graph,
        to_graph_hash,
    ):
        if self.data_cnt < self.maxlen:
            self.from_graphs.append(from_graph)
            self.from_graph_hashes.append(from_graph_hash)
            self.node_ids.append(node_id)
            self.xfer_ids.append(xfer_id)
            self.rewards.append(reward)
            self.to_graphs.append(to_graph)
            self.to_graph_hashes.append(to_graph_hash)
            self.data_cnt += 1
            return True
        else:
            return False

    def save_data(self):
        save_graphs('from_graphs.dat', self.from_graphs)
        save_graphs('to_graphs.dat', self.to_graphs)
        other_info = {}
        other_info['node_ids'] = self.node_ids
        other_info['xfer_ids'] = self.xfer_ids
        other_info['rewards'] = self.rewards
        other_info['from_graph_hashes'] = self.from_graph_hashes
        other_info['to_graph_hashes'] = self.to_graph_hashes
        with open('node_xfer_reward.json', 'w') as f:
            json.dump(other_info, f)

    def load_data(self):
        self.from_graphs = load_graphs('from_graphs.dat')[0]
        self.to_graphs = load_graphs('to_graphs.dat')[0]
        with open('node_xfer_reward.json', 'r') as f:
            other_info = json.load(f)
        self.node_ids = other_info['node_ids']
        self.xfer_ids = other_info['xfer_ids']
        self.rewards = other_info['rewards']
        self.from_graph_hashes = other_info['from_graph_hashes']
        self.to_graph_hashes = other_info['to_graph_hashes']
        self.data_cnt = len(self.from_graphs)

    def all_data(self):
        data = []
        for i in range(self.data_cnt):
            data.append(
                (
                    self.from_graphs[i],
                    self.from_graph_hashes[i],
                    self.node_ids[i],
                    self.xfer_ids[i],
                    self.rewards[i],
                    self.to_graphs[i],
                    self.to_graph_hashes[i],
                )
            )
        return data

    def sample(self):
        idx = random.choice(list(range(self.data_cnt)))
        return (
            self.from_graphs[idx],
            self.from_graph_hashes[idx],
            self.node_ids[idx],
            self.xfer_ids[idx],
            self.rewards[idx],
            self.to_graphs[idx],
            self.to_graph_hashes[idx],
        )


# p = PosRewardData()
# p.load_data()
# print(p.all_data())
