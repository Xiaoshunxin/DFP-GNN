'''
@Project: DFP-GNN
@File   : graph_conv
@Time   : 2021/8/26 21:17
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Definition of graph convolution layer
'''
import torch
import torch.nn as nn
import numpy as np


class GraphConvolution(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        """
            Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
        :param input_dim: the number dimension of input feature
        :param output_dim: the output dimension of GCN
        :param bias: whether to use bias
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.initial_parameter(bias)

    def initial_parameter(self, bias: bool) -> None:
        """
            initial the weight and bias by Glorot method
        :param bias: wheter to use bias
        """
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        initial_weight = torch.rand(self.input_dim, self.output_dim) * 2 * init_range - init_range
        self.weight = nn.Parameter(initial_weight)
        if bias:
            initial_bias = torch.rand(self.output_dim) * 2 * init_range - init_range
            self.bias = nn.Parameter(initial_bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        """
            the computing of graph convolution
        :param fea: input feature matrix
        :param adj: adjacency matrix
        :return:  the aggregating embedding by graph convolution
        """
        hidden = torch.mm(fea, self.weight)
        output = torch.spmm(adj, hidden)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'