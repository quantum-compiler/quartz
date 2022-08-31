from typing import List, cast

import dgl
import dgl.function as fn
import torch
import torch as th
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from dgl.utils import expand_as_pair
from model.basis import *
from torch import nn

"""QGINConv is stolen with modification from DGL source code for dgl.nn.pytorch.conv.ginconv"""
# https://docs.dgl.ai/en/0.8.x/_modules/dgl/nn/pytorch/conv/ginconv.html#GINConv

"""Torch Module for Graph Isomorphism Network layer"""


class QGINConv(nn.Module):
    r"""Graph Isomorphism Network layer from `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    where :math:`e_{ji}` is the weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that `e_{ji}` is broadcastable with `h_j^{l}`.

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula, default: None.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``), default: 'sum'.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)

    >>> # With activation
    >>> from torch.nn.functional import relu
    >>> conv = GINConv(lin, 'max', activation=relu)
    >>> res = conv(g, feat)
    >>> res
    tensor([[5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [2.5011, 0.0000, 0.0089, 2.0541, 0.8262, 0.0000, 0.0000, 0.1371, 0.0000,
             0.0000]], grad_fn=<ReluBackward0>)
    """

    def __init__(
        self,
        input_dim: int,
        apply_func=None,
        aggregator_type='sum',
        init_eps=0,
        learn_eps=False,
        activation=None,
    ):
        super(QGINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

    def __msg_func_with_edge_info(self, edges):
        return {'m': torch.cat([edges.src['h'], edges.data['w']], dim=1)}

    def __reduce_neigh_edge_to_feat(self, nodes):
        # nodes.mailbox['m']: (num_nodes, num_neighbors, msg_dim)
        msgs: torch.Tensor = nodes.mailbox['m']
        if self._aggregator_type == 'sum':
            neigh_feat = torch.sum(msgs, dim=1)
        elif self._aggregator_type == 'mean':
            neigh_feat = torch.mean(msgs, dim=1)
        elif self._aggregator_type == 'max':
            neigh_feat = torch.max(msgs, dim=1).values
        return {'neigh': neigh_feat}

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        reducer = self.__reduce_neigh_edge_to_feat
        with graph.local_scope():
            """We incorporate edge info here by cat"""
            # aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn = self.__msg_func_with_edge_info
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            graph.ndata['h'] = feat
            graph.update_all(aggregate_fn, reducer)
            _size_to_pad: int = graph.ndata['neigh'].shape[-1] - feat.shape[-1]
            # feat = F.pad(feat, (0, _size_to_pad))
            feat = torch.cat(
                [feat, torch.zeros(feat.shape[0], _size_to_pad).to(feat.device)], dim=-1
            )
            rst = (1 + self.eps) * feat + graph.ndata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


"""Below are stolen with modification from DGL example for GIN"""
# https://github.com/dmlc/dgl/blob/7cd531c4d67f8de5c597ae5ad589e6acae36a4c3/examples/pytorch/gin/gin.py

"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


class ApplyNodeFunc(nn.Module):
    """Update the node feature h with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.leaky_relu(h)
        return h


class QGIN(nn.Module):
    """QGIN model"""

    def __init__(
        self,
        num_layers: int,
        num_mlp_layers: int,
        num_gate_types: int,
        gate_type_embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        learn_eps: bool,
        neighbor_pooling_type: str,
        graph_pooling_type: str = 'none',
        final_dropout: float = 0.0,
    ):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean, max or none)
            when this is 'none', only return per-node feature
            when this is not 'none', return a pair (pre-node feature, pre-graph feature)
        """
        super(QGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # Embedding layer: gate_type_idx in (1, ) -> (input_dim, ) (input_dim == num_gate_types)
        self.gate_type_embedding = nn.Embedding(num_gate_types, gate_type_embed_dim)
        input_dim = gate_type_embed_dim

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            mlp_input_dim = input_dim if layer == 0 else hidden_dim
            mlp_input_dim += 3
            mlp_output_dim = hidden_dim if layer < self.num_layers - 1 else output_dim
            mlp = MLP(num_mlp_layers, mlp_input_dim, hidden_dim, mlp_output_dim)
            self.ginlayers.append(
                QGINConv(
                    mlp_input_dim,
                    ApplyNodeFunc(mlp),
                    neighbor_pooling_type,
                    0,
                    self.learn_eps,
                )
            )
            # NOTE: ApplyNodeFunc already has BN and Activation

        self.global_pool: bool = False
        if graph_pooling_type != 'none':
            self.global_pool = True
            if graph_pooling_type == 'sum':
                self.pool = SumPooling()
            elif graph_pooling_type == 'mean':
                self.pool = AvgPooling()
            elif graph_pooling_type == 'max':
                self.pool = MaxPooling()
            else:
                raise NotImplementedError

            self.drop = nn.Dropout(final_dropout)

            self.linears_prediction = torch.nn.ModuleList()
            for layer in range(num_layers):
                in_dim = hidden_dim if layer < self.num_layers - 1 else output_dim
                self.linears_prediction.append(nn.Linear(in_dim, output_dim))
            self.output_dim = output_dim

    def forward(self, g: dgl.DGLGraph):
        # Use embedding layer to generate init features for nodes in the graph
        h = self.gate_type_embedding(g.ndata['gate_type'])
        # Create edge info
        g.edata['w'] = torch.cat(
            [
                torch.unsqueeze(g.edata['src_idx'], 1),
                torch.unsqueeze(g.edata['dst_idx'], 1),
                torch.unsqueeze(g.edata['reversed'], 1),
            ],
            dim=1,
        )

        # list of hidden representation at each layer (including input)
        hidden_rep: List[torch.Tensor] = []

        for i in range(self.num_layers):
            h = self.ginlayers[i](g, h)
            hidden_rep.append(h)

        if self.global_pool:
            # num_graphs = len(g.batch_num_nodes())
            feat_over_layer = cast(torch.Tensor, 0)

            # perform pooling over all nodes in each graph in every layer
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                feat_over_layer += self.drop(self.linears_prediction[i](pooled_h))

            return hidden_rep[-1], feat_over_layer
        else:
            return hidden_rep[-1]
