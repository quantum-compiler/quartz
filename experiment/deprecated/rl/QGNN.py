import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return {'h_N': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        # g.edata['w'] = w #self.embed(torch.unsqueeze(w,1))
        g.update_all(self.message_func, self.reduce_func)
        h_N = g.ndata['h_N']
        h_total = torch.cat([h, h_N], dim=1)
        return self.linear2(h_total)


class QGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, inter_dim):
        super(QGNN, self).__init__()
        self.conv1 = QConv(in_feats, inter_dim, h_feats)
        self.conv2 = QConv(h_feats, inter_dim, h_feats)
        self.conv3 = QConv(h_feats, inter_dim, h_feats)
        self.conv4 = QConv(h_feats, inter_dim, h_feats)
        self.conv5 = QConv(h_feats, inter_dim, num_classes)
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
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        return h


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_logits = []
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['gate_type']

    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(20):
        # Forward
        logits = model(g)

        # Compute prediction
        ####TODO######
        pred = logits.argmax(1)

        # Compute loss
        # Note that we should only compute the losses of the nodes in the training set,
        # i.e. with train_mask 1.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach())

        if e % 5 == 0:
            print(
                'In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )


def train_supervised(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_logits = []
    best_val_acc = 0
    best_test_acc = 0
    epochs = 20

    features = g.ndata['gate_type']

    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(epochs):
        # Forward
        logits = model(g)

        # Compute loss
        # Note that we should only compute the losses of the nodes in the training set,
        # i.e. with train_mask 1.
        # print(logits)

        loss = torch.nn.MSELoss()(logits[train_mask], labels[train_mask])
        pred = logits > 0.5

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach())

        if e % 5 == 0:
            print(
                'In epoch {}, loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )


# import dgl.data
# dataset = dgl.data.CoraGraphDataset()
# g = dataset[0]
src_id = [0, 3, 1, 1, 5]
dst_id = [1, 1, 2, 4, 0]
src_idx = [0, 0, 0, 1, 0]
dst_idx = [0, 1, 0, 0, 0]
node_gate_tp = [1, 0, 2, 3, 4, 1]
src_id2 = src_id + dst_id
dst_id2 = dst_id + src_id
src_idx2 = src_idx + dst_idx
dst_idx2 = dst_idx + src_idx
reverse = [0] * len(src_id) + [1] * len(src_id)
g = dgl.graph((torch.tensor(src_id2), torch.tensor(dst_id2)))
g.edata['src_idx'] = torch.tensor(src_idx2)
g.edata['dst_idx'] = torch.tensor(dst_idx2)
g.edata['reversed'] = torch.tensor(reverse)
g.ndata['gate_type'] = torch.tensor(node_gate_tp)

# g.ndata['label'] = torch.tensor([1,1,1,1,1,1])
g.ndata['label'] = torch.tensor(
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    dtype=torch.float,
)
g.ndata['train_mask'] = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
g.ndata['val_mask'] = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool)
g.ndata['test_mask'] = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)

model = QGNN(5, 16, 3, 16)
train_supervised(g, model)
