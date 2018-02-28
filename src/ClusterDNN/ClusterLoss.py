import torch
import numpy as np
from torch.autograd import Function, Variable
import torch.nn as nn


class ClusterLoss(nn.Module):

    def __init__(self, cluster_num, feature_size, use_cpu):
        super(ClusterLoss, self).__init__()
        self._cluster_num = cluster_num
        self._feature_size = feature_size
        self._use_cpu = use_cpu

    def forward(self, feature_vec, normalized_w, feature_cog):

        # feature_vec 16x64
        # normalized_w 16x21
        # feature_cog 21x64

        vec_diff = feature_vec - torch.mm(normalized_w, feature_cog)
        # vec_diff = Variable(torch.zeros(feature_vec.size()))
        # predicted_age = np.zeros(w.shape)

        # for i in range(feature_vec.size()[0]):
        #     feature_vec_expand = feature_vec[i].unsqueeze(1).expand([self._feature_size, self._cluster_num]) # 64x21
        #     diff = feature_cog.t() - feature_vec_expand  # 64x21
        #     vec_diff[i] = torch.sum(diff * normalized_w[i], dim=1)

            # best_cluster_id = np.sum(vec_diff ** 2, axis=0).argmin()
            # predicted_age[i] = self._cluster_st + best_cluster_id * self._cluster_step
            #
            # if age[i] < 23:
            #     print(age[i], feature_vec[i, 0], expected_vec[i, 0], expected_vec[i, 0] - feature_vec[i, 0])
        #
        # age_diff = np.c_[age, predicted_age]
        # age_AE = np.mean(abs(age - predicted_age))
        # print(age_diff)
        #
        # MAE += age_AE

        result = torch.mean(vec_diff ** 2)
        return result
