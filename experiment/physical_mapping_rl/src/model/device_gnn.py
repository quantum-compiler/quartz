import dgl
import torch
import torch.nn as nn
from dgl.nn import SAGEConv, GINConv


class DeviceGNNSAGE(nn.Module):
    def __init__(self,
                 # embedding
                 feature_type,  # degree / id / both
                 num_degree_types,  # max degree + 1
                 num_id_types,  # max node id
                 degree_embedding_dim,  # dimension of feature embedding
                 id_embedding_dim,  # dimension of id embedding
                 # graph conv
                 num_layers,  # # of convolution layers
                 hidden_dimension,  # dimension of each internal GNN layer
                 out_dimension,  # output dimension of final GNN layer
                 ):
        super(DeviceGNNSAGE, self).__init__()
        # embedding network
        self.feature_type = feature_type
        if feature_type == 'both':
            self.degree_embedding = nn.Embedding(num_degree_types, degree_embedding_dim)
            self.id_embedding = nn.Embedding(num_id_types, id_embedding_dim)
            self.input_dimension = degree_embedding_dim + id_embedding_dim
        elif feature_type == 'degree':
            self.degree_embedding = nn.Embedding(num_degree_types, degree_embedding_dim)
            self.id_embedding = None
            self.input_dimension = degree_embedding_dim
        elif feature_type == 'id':
            self.degree_embedding = None
            self.id_embedding = nn.Embedding(num_id_types, id_embedding_dim)
            self.input_dimension = id_embedding_dim
        else:
            raise NotImplementedError
        # conv layers
        conv_layers = [SAGEConv(self.input_dimension, hidden_dimension, 'mean')]
        for _ in range(num_layers - 2):
            conv_layers.append(SAGEConv(hidden_dimension, hidden_dimension, 'mean'))
        conv_layers.append(SAGEConv(hidden_dimension, out_dimension, 'mean'))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, graph: dgl.DGLGraph):
        # get initial feature
        if self.feature_type == 'both':
            degree_feature = self.degree_embedding(graph.ndata['degree'])
            id_feature = self.id_embedding(graph.ndata['id'])
            feature = torch.concat([degree_feature, id_feature], dim=1)
        elif self.feature_type == 'degree':
            feature = self.degree_embedding(graph.ndata['degree'])
        elif self.feature_type == 'id':
            feature = self.id_embedding(graph.ndata['id'])
        else:
            raise NotImplementedError
        # apply convolution
        for i in range(len(self.conv_layers)):
            feature = self.conv_layers[i](graph, feature)
        return feature


class DeviceGNNGINLocal(nn.Module):
    def __init__(self,
                 # embedding
                 feature_type,  # degree / id / both
                 num_degree_types,  # max degree + 1
                 num_id_types,  # max node id
                 degree_embedding_dim,  # dimension of feature embedding
                 id_embedding_dim,  # dimension of id embedding
                 # graph conv
                 num_layers,  # # of convolution layers
                 hidden_dimension,  # dimension of each internal GNN layer
                 out_dimension,  # output dimension of final GNN layer
                 ):
        super(DeviceGNNGINLocal, self).__init__()
        # embedding network
        self.feature_type = feature_type
        if feature_type == 'both':
            self.degree_embedding = nn.Embedding(num_degree_types, degree_embedding_dim)
            self.id_embedding = nn.Embedding(num_id_types, id_embedding_dim)
            self.input_dimension = degree_embedding_dim + id_embedding_dim
        elif feature_type == 'degree':
            self.degree_embedding = nn.Embedding(num_degree_types, degree_embedding_dim)
            self.id_embedding = None
            self.input_dimension = degree_embedding_dim
        elif feature_type == 'id':
            self.degree_embedding = None
            self.id_embedding = nn.Embedding(num_id_types, id_embedding_dim)
            self.input_dimension = id_embedding_dim
        else:
            raise NotImplementedError
        # conv layers
        conv_layers = [GINConv(torch.nn.Linear(self.input_dimension, hidden_dimension), 'sum')]
        for _ in range(num_layers - 2):
            conv_layers.append(GINConv(torch.nn.Linear(hidden_dimension, hidden_dimension), 'sum'))
        conv_layers.append(GINConv(torch.nn.Linear(hidden_dimension, out_dimension), 'sum'))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, graph: dgl.DGLGraph):
        # get initial feature
        if self.feature_type == 'both':
            degree_feature = self.degree_embedding(graph.ndata['degree'])
            id_feature = self.id_embedding(graph.ndata['id'])
            feature = torch.concat([degree_feature, id_feature], dim=1)
        elif self.feature_type == 'degree':
            feature = self.degree_embedding(graph.ndata['degree'])
        elif self.feature_type == 'id':
            feature = self.id_embedding(graph.ndata['id'])
        else:
            raise NotImplementedError
        # apply convolution
        for i in range(len(self.conv_layers)):
            feature = self.conv_layers[i](graph, feature)
        return feature
