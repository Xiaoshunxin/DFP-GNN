'''
@Project: DFP-GNN
@File   : dfp_gnn
@Time   : 2021/8/26 21:06
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Definition of the proposed DFP-GNN framework
'''
import torch.nn as nn
from models.sgae import SGAE
from layers.mlfpn_gcn import MLFPN_GCN
from layers.fusion_layer import FusionLayer
from layers.cluster_layer import ClusterLayer


class DFPGNN(nn.Module):

    def __init__(self, view_dims, pm_hidden_dims, sm_hidden_dims, num_clusters):
        super(DFPGNN, self).__init__()
        self.view_dims = view_dims
        self.pm_hidden_dims = pm_hidden_dims
        self.sm_hidden_dims = sm_hidden_dims
        self.num_views = len(view_dims)

        # define the preliminary module for all views
        self.preliminary_module = nn.ModuleList()
        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(view_dims[i])
            temp_dims.extend(pm_hidden_dims)
            self.preliminary_module.append(MLFPN_GCN(temp_dims, nn.ReLU()))

        # define the fusion module
        self.fusion_module = FusionLayer(num_views=self.num_views)

        # define the shared module
        temp_dims = []
        temp_dims.append(self.pm_hidden_dims[-1])
        temp_dims.extend(self.sm_hidden_dims)
        self.shared_module = SGAE(temp_dims, nn.ReLU())

        # define the clustering layer
        self.cluster_layer = ClusterLayer(num_clusters, self.sm_hidden_dims[-1])

    def forward(self, feats, adjs):
        # computation in the preliminary module
        hidden_list = []
        for i in range(self.num_views):
            hidden = self.preliminary_module[i](feats[i], adjs[i])
            hidden_list.append(hidden)

        # computation in the fusion module
        combined_feature, combined_adjacent = self.fusion_module(hidden_list, adjs)

        # computation in the shared module
        hidden, X_bar, A_bar = self.shared_module(combined_feature, combined_adjacent)

        # computation in the clustering layer
        q = self.cluster_layer(hidden)

        return hidden, X_bar, A_bar, q

    def get_preliminary_ae(self, index: int) -> nn.Module:
        """
            return the index-th encoder of the initial ae module for initializing weights
        :param index: the index of current
        :return: the index-th encoder of the initial ae module
        """
        if index > len(self.preliminary_module) or index < 0:
            raise ValueError('Requested subautoencoder cannot be constructed, index out of range.')
        return self.preliminary_module[index]