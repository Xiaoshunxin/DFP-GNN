'''
@Project: DFP-GNN
@File   : mlfpn_gcn
@Time   : 2021/8/26 21:19
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    define a multi layer forward propagation network, which each layer is a graph convolution operation.
'''
import torch
import  torch.nn as nn
from typing import List

from utils.ops_al import build_layer_units


class MLFPN_GCN(nn.Module):

    def __init__(self, dims: List[int], act_func: nn.Module=nn.ReLU()):
        """
        :param dims: the dimensions of all hidden layers
        :param act_f: the activation function used in the output of each hidden layer, except the last layer
        """
        super(MLFPN_GCN, self).__init__()

        # build the network
        self.network = build_layer_units(layer_type='gcn', dims=dims, act_func=act_func)

    def forward(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        """
        :param fea: the input feature matrix
        :param adj: the input adjacent matrix
        :return: the embedding feature matrix produced by MLFPN_GCN
        """
        output = fea
        for seq in self.network:
            if len(seq) == 1:  # the last layer without activate function
                output = seq[0](output, adj)
            elif len(seq) == 2:  # the first l-1 layers with activation function
                output = seq[0](output, adj)
                output = seq[1](output) # the activation layer
        return output