import dgl
import torch.nn as nn
from dgl.nn import SAGEConv


class DeviceGNN(nn.Module):
    def __init__(self,
                 # embedding
                 num_feature_types,  # # of node features (i.e. max degree)
                 feature_embedding_dim,  # dimension of feature embedding
                 # graph conv
                 num_layers,  # # of convolution layers
                 hidden_dimension,  # dimension of each internal GNN layer
                 out_dimension,  # output dimension of final GNN layer
                 ):
        super(DeviceGNN, self).__init__()
        # embedding network
        self.node_feature_embedding = nn.Embedding(num_feature_types, feature_embedding_dim)
        # conv layers
        conv_layers = [SAGEConv(feature_embedding_dim, hidden_dimension, 'mean')]
        for _ in range(num_layers - 2):
            conv_layers.append(SAGEConv(hidden_dimension, hidden_dimension, 'mean'))
        conv_layers.append(SAGEConv(hidden_dimension, out_dimension, 'mean'))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, graph: dgl.DGLGraph):
        feature = self.node_feature_embedding(graph.ndata['degree'])
        for i in range(len(self.conv_layers)):
            feature = self.conv_layers[i](graph, feature)
        return feature
