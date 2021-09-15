'''
@Project: DFP-GNN
@File   : cluster_layer
@Time   : 2021/8/26 21:16
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Definition of clustering layer
'''
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ClusterLayer(nn.Module):

    def __init__(self, num_clusters, hidden_dim, alpha=1):
        """
        :param num_clusters: the number of clusters
        :param hidden_dim: the dimension of hidden layer
        :param alpha:
        """
        super(ClusterLayer, self).__init__()
        self.alpha = alpha
        self.network = Parameter(torch.Tensor(num_clusters, hidden_dim)).float()
        torch.nn.init.xavier_normal_(self.network.data)

    def forward(self, z) -> torch.Tensor:
        """
            computation in the clustering layer
        :param z: the input hidden embedding
        :return: the soft distribution of data
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.network, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q