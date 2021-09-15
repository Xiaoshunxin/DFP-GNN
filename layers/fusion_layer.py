'''
@Project: DFP-GNN
@File   : fusion_layer
@Time   : 2021/8/26 21:16
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Definition of fusion layer
'''
import torch
import torch.nn as nn


class FusionLayer(nn.Module):

    def __init__(self, num_views=5, fusion_type='weighted'):
        """
        :param fusion_type: include concatenate/average
        """
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views

        # define the attention weights for feature matrix and adjacent matrix
        self.pai_fea = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        self.pai_adj = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)

    def forward(self, features, adjs):
        if self.fusion_type == "concatenate":
            combined_feature = torch.cat(features, dim=1)
        elif self.fusion_type == "average":
            pass
        elif self.fusion_type == "weighted":
            # combine the feature matrix
            exp_sum_pai_fea = 0
            for i in range(self.num_views):
                exp_sum_pai_fea += torch.exp(self.pai_fea[i])
            combined_feature = (torch.exp(self.pai_fea[0]) / exp_sum_pai_fea) * features[0]
            for i in range(1, self.num_views):
                combined_feature = combined_feature + (torch.exp(self.pai_fea[i]) / exp_sum_pai_fea) * features[i]

            # combine the adjacent matrix
            exp_sum_pai_adj = 0
            for i in range(self.num_views):
                exp_sum_pai_adj += torch.exp(self.pai_adj[i])
            combined_adjacent = (torch.exp(self.pai_adj[0]) / exp_sum_pai_adj) * adjs[0]
            for i in range(1, self.num_views):
                combined_adjacent = combined_adjacent + (torch.exp(self.pai_adj[i]) / exp_sum_pai_adj) * adjs[i]

        else:
            print("Please using a correct fusion type")
            exit()

        return combined_feature, combined_adjacent