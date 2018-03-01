import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
from torch.utils.data import DataLoader

from src.ClusterDNN.BatchSet import BatchSet
from src.ClusterDNN.ClusterNet import create_cluster_net
import src.utils as utils
from src.Logger import Logger


class ClusterModel:

    def __init__(self, data_path, mode, use_cpu):

        print('Construct cluster model.')

        np.set_printoptions(precision=3, suppress=True, threshold=2000)

        self._use_cpu = use_cpu
        self._mode = mode
        self._expt_path = 'expt/cluster/'

        self._cluster_st = 21
        self._cluster_ed = 82
        self._cluster_step = 3
        self._clusters = np.arange(self._cluster_st, self._cluster_ed, self._cluster_step)
        self._cluster_num = self._clusters.shape[0]
        self._feature_size = 64

        if not os.path.exists('expt/'):
            os.mkdir('expt/')
        if not os.path.exists(self._expt_path):
            os.mkdir(self._expt_path)

        if mode == 'training':
            data = np.load(data_path + 'training_data.npy')
            ages = np.load(data_path + 'training_ages.npy')
        elif mode == 'validation':
            data = np.load(data_path + 'validation_data.npy')
            ages = np.load(data_path + 'validation_ages.npy')
        elif mode == 'test':
            data = np.load(data_path + 'test_data.npy')
            ages = np.load(data_path + 'test_ages.npy')
        else:
            raise Exception('mode must be in [training, validation, test].')

        batch_set = BatchSet(data, ages)

        if use_cpu:
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 0
            pin_memory = True

        batch_size = 16

        self._data_loader = DataLoader(dataset=batch_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)

        if cudnn.version() is None:
            cudnn.enabled = False

    def load_pretrain(self, pretrain_model):

        expt_baseline_path = 'expt/baseline/'
        baseline_net_file = expt_baseline_path + 'baseline_net_' + str(pretrain_model) + '.pkl'

        cluster_net = create_cluster_net()

        if os.path.exists(baseline_net_file):
            baseline_net = torch.load(baseline_net_file)
            cluster_net.load_state_dict(baseline_net.state_dict(), strict=False)

        torch.save(cluster_net, self._expt_path + 'cluster_net_pretrain.pkl')

    def _calc_drift(self, data):

        data_sum = data.sum(axis=0)

        drift = np.zeros(data.shape)
        for i in range(drift.shape[0]):
            data_other = (data_sum - data[i,:]) / (data.shape[0] - 1)
            drift[i] = data[i,:] - data_other
            drift_norm = np.sum(drift[i] ** 2) ** 0.5
            drift[i] /= drift_norm

        return drift

    def train(self, st_epoch=0):

        print('Train Dataset.')

        max_epoches = 10000
        lr0 = 1e-2
        lr_shirnk_step = 100
        lr_shrink_gamma = 0.5
        drift_force = 5

        if st_epoch == 0:
            print('Construct cluster model. Load from pretrain net.')
            cluster_net = torch.load(self._expt_path + 'cluster_net_pretrain.pkl')
            feature_cog = None
        else:
            cluster_net_file_name = 'cluster_net_%d.pkl' % (st_epoch - 1)
            cluster_cog_file_name = 'cluster_cog_%d.npy' % (st_epoch - 1)
            if os.path.exists(self._expt_path + cluster_net_file_name):
                print('Construct cluster model. Load from pkl file.')
                cluster_net = torch.load(self._expt_path + cluster_net_file_name)
                feature_cog = np.load(self._expt_path + cluster_cog_file_name)
            else:
                print('Construct cluster model. Load from pretrain net.')
                cluster_net = torch.load(self._expt_path + 'cluster_net_pretrain.pkl')
                feature_cog = None

        cluster_net.float()
        cluster_net.train()
        if not self._use_cpu:
            cluster_net.cuda()
        else:
            cluster_net.cpu()

        lr = lr0 * lr_shrink_gamma ** (st_epoch // lr_shirnk_step)

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(cluster_net.parameters(), lr=lr, alpha=0.9)
        # optimizer = optim.SGD(net.parameters(), lr=st_lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_shirnk_step, gamma=lr_shrink_gamma)

        logger = Logger('train', 'train_cluster.log')

        for epoch in range(st_epoch, max_epoches):

            # Update cluster net

            if feature_cog is not None:

                message = 'Epoch: %d. Update cluster net.' % epoch
                print(message)
                logger.log(message)

                running_loss = 0
                MAE = 0

                drift = self._calc_drift(feature_cog)
                feature_cog += drift * drift_force

                for batch_id, sample in enumerate(self._data_loader):

                    data, age = sample['data'].unsqueeze(dim=1).float(), sample['age'].float().numpy()
                    if not self._use_cpu:
                        data = data.cuda()
                    data = Variable(data)

                    w = np.exp(-((age - self._clusters) / 1.5) ** 2)  # 16x21
                    w_norm = w.sum(axis=1).repeat(self._cluster_num).reshape(w.shape[0], self._cluster_num)  # 16x21
                    normalized_w = w / w_norm

                    expected_vec = normalized_w.dot(feature_cog)
                    expected_vec = Variable(torch.from_numpy(expected_vec).float(), requires_grad=False)
                    if not self._use_cpu:
                        expected_vec = expected_vec.cuda()

                    feature_vec = cluster_net(data)

                    loss = criterion(feature_vec, expected_vec)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.data[0]

                    feature_vec = feature_vec.cpu().data.numpy()
                    predicted_age = np.zeros(age.shape)

                    for i in range(feature_vec.shape[0]):
                        vec_diff = feature_vec[i] - feature_cog
                        best_cluster = np.sum(np.abs(vec_diff), axis=1).argmin()
                        predicted_age[i] = self._cluster_st + best_cluster * self._cluster_step

                    comp_age = np.c_[age, predicted_age, predicted_age - age]
                    error = np.abs(comp_age[:,2]).mean()
                    MAE += error
                    message = '  Batch: %d, AE: %.3f, MAE: %.3f, Loss: %.3f, Epoch Loss: %.3f' % \
                              (batch_id, error, MAE / (batch_id + 1), loss.data[0], running_loss / (batch_id + 1))
                    print(message)
                    print(comp_age)
                    logger.log(message)

            scheduler.step()

            # Update features center of gravity

            message = 'Epoch: %d. Update features center of gravity.' % epoch
            print(message)
            logger.log(message)

            cluster_w = np.zeros(self._cluster_num)
            new_feature_cog = np.zeros((self._cluster_num, self._feature_size))

            for batch_id, sample in enumerate(self._data_loader):

                data, age = sample['data'].unsqueeze(dim=1).float(), sample['age'].float().numpy()
                if not self._use_cpu:
                    data = data.cuda()
                data = Variable(data)

                w = np.exp(-((age - self._clusters) / 1.5) ** 2)  # 16x21
                cluster_w += w.sum(axis=0)  # 21x1

                feature_vec = cluster_net(data) # 16x64

                feature_vec = feature_vec.cpu().data.numpy()
                new_feature_cog += w.T.dot(feature_vec)  # 21x64

            cluster_w = cluster_w.repeat(self._feature_size).reshape(self._cluster_num, self._feature_size) # 21x64
            feature_cog = new_feature_cog / cluster_w # 21x64
            print(feature_cog[:,:3])

            torch.save(cluster_net, self._expt_path + 'cluster_net_%d.pkl' % epoch)
            np.save(self._expt_path + 'cluster_cog_%d.npy' % epoch, feature_cog)

            message = 'Epoch: %d. cluster_net.pkl and cluster_cog.npy saved.' % epoch
            print(message)
            logger.log(message)

    def test(self, model_epoch):

        print('Test model.', 'Mode:', self._mode)

        cluster_net_file_name = 'cluster_net_%d.pkl' % model_epoch
        if os.path.exists(self._expt_path + cluster_net_file_name):
            print('Construct cluster_net. Load from pkl file.')
            cluster_net = torch.load(self._expt_path + cluster_net_file_name)
        else:
            raise Exception('No such model file.')

        cluster_net.float()
        cluster_net.eval()
        if not self._use_cpu:
            cluster_net.cuda()
        else:
            cluster_net.cpu()

        test_res = None

        for batch_id, sample in enumerate(self._data_loader):

            data = sample['data'].unsqueeze(dim=1).float()
            age = sample['age'].float().numpy()

            if self._use_cpu is False:
                data = data.cuda()

            data = Variable(data)

            predicted_age = cluster_net(data)

            predicted_age = predicted_age.data.cpu().numpy()
            batch_res = np.c_[age, predicted_age, predicted_age - age]

            if test_res is None:
                test_res = batch_res
            else:
                test_res = np.r_[test_res, batch_res]

        test_res = test_res[test_res[:, 0].argsort()]
        np.save(self._expt_path + self._mode + 'cluster_test_result.npy', np.array(test_res))
        print(test_res)

        abs_error = abs(test_res[:, 2])
        MAE = abs_error.mean()
        CI_75 = np.percentile(abs_error, 75)
        CI_95 = np.percentile(abs_error, 95)

        print('Test size: %d, MAE: %.3f, CI 75%%: %.3f, CI 95%%: %.3f. ' % (test_res.shape[0], MAE, CI_75, CI_95))

        utils.plot_scatter(test_res, CI_75, CI_95, self._expt_path, self._mode + '_cluster_' + str(model_epoch))
