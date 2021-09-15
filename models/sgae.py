'''
@Project: DFP-GNN
@File   : sgae
@Time   : 2021/8/26 21:07
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Stacked Graph Autoencoder with three output (embedding features, reconstructed features, reconstructed adjacency)
'''
import torch
import torch.nn as nn
from typing import List

from layers.mlfpn_fc import MLFPN_FC
from layers.mlfpn_gcn import MLFPN_GCN
from utils.ops_al import dot_product_decode


class SGAE(nn.Module):
    def __init__(self, dims: List[int], act_func: nn.Module=nn.ReLU()):
        """
            Stacked Graph AutoEncoder
        :param dims: the dimensions of all hidden layers
        :param act_func: the activation function used in the output of each hidden layer, except the last layer
        """
        super(SGAE, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.hidden_dim = dims[-1]

        # construct the encoder
        self.encoder = MLFPN_GCN(dims=dims, act_func=act_func)

        # construct the decoder
        self.decoder = MLFPN_FC(dims=list(reversed(dims)), act_func=act_func)

    def forward(self, fea: torch.Tensor, adj: torch.sparse):
        """
            the computing of SGAE
        :param fea: the input feature matrix
        :param adj: the input adjacency matrix
        :return: embedding features, reconstructed features, reconstructed adjacency
        """
        embedding = self.encoder(fea, adj)
        fea_bar = self.decoder(embedding)
        adj_bar = dot_product_decode(embedding)

        return embedding, fea_bar, adj_bar

    def copy_weights(self, input_model: nn.Module) -> None:
        """
            Copy the weights of self.encoder into the given network
        :param input_model: the encoders of initial autoencoder of DFP-GNN
        :return: None
        """
        input_model.network.data.copy_(self.encoder.data)