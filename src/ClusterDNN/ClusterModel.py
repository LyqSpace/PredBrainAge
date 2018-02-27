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

    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
        pass

    @staticmethod
    def train(data_path, retrain, use_cpu, st_epoch=0):

        print('Train Dataset.')

        expt_path = 'expt/'
        max_epoches = 100000
        batch_size = 16
        lr0 = 1e-2
        lr_shirnk_step = 5000
        lr_shrink_gamma = 0.5

        training_data = np.load(data_path + 'training_data.npy')
        training_ages = np.load(data_path + 'training_ages.npy')

        batch_set = BatchSet(training_data, training_ages)

        if use_cpu:
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 2
            pin_memory = True
        data_loader = DataLoader(dataset=batch_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

        if (st_epoch-1) < 0 or retrain:
            print('Construct cluster model. Create a new network.')
            cluster_net = create_cluster_net()
        else:
            cluster_net_file_name = 'cluster_net_%d.pkl' % (st_epoch-1)
            if os.path.exists(expt_path + cluster_net_file_name):
                print('Construct cluster model. Load from pkl file.')
                cluster_net = torch.load(expt_path + cluster_net_file_name)
            else:
                print('Construct cluster model. Create a new network.')
                cluster_net = create_cluster_net()

        cluster_net.float()
        cluster_net.train()
        if use_cpu is False:
            cudnn.enabled = True
            cluster_net.cuda()
        else:
            cudnn.enabled = False

        lr = lr0 * lr_shrink_gamma ** (st_epoch // lr_shirnk_step)

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(cluster_net.parameters(), lr=lr, alpha=0.9)
        # optimizer = optim.SGD(net.parameters(), lr=st_lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_shirnk_step, gamma=lr_shrink_gamma)

        logger = Logger('train', 'train_cluster.log')

        for epoch in range(st_epoch, max_epoches):

            running_loss = 0

            for batch_id, sample in enumerate(data_loader):

                data, age = sample['data'].unsqueeze(dim=1).float(), sample['age'].unsqueeze(dim=1).float()

                if use_cpu is False:
                    data, age = data.cuda(), age.cuda()

                data, age = Variable(data), Variable(age)

                predicted_age = cluster_net(data)

                loss = criterion(predicted_age, age)
                optimizer.zero_grad()
                loss.backward()
                scheduler.step()
                optimizer.step()

                running_loss += loss.data[0]

                message = 'Epoch: %d, Batch: %d, Loss: %.3f, Epoch Loss: %.3f' % \
                          (epoch, batch_id, loss.data[0], running_loss / (batch_id+1))
                print(message)
                logger.log(message)

                comp_res = np.c_[age.data.cpu().numpy()[:, :, 0], predicted_age.data.cpu().numpy()]
                print(comp_res)
                logger.log(comp_res)

            torch.save(cluster_net, expt_path + 'cluster_net_%d.pkl' % epoch)

    @staticmethod
    def test(data_path, model_epoch, mode, use_cpu):

        print('Test model.', data_path, 'Mode:', mode)

        expt_path = 'expt/'
        batch_size = 16

        if mode == 'validation':
            test_data = np.load(data_path + 'validation_data.npy')
            test_ages = np.load(data_path + 'validation_ages.npy')
        else:
            test_data = np.load(data_path + 'test_data.npy')
            test_ages = np.load(data_path + 'test_ages.npy')

        batch_set = BatchSet(test_data, test_ages)

        if use_cpu:
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 2
            pin_memory = True
        data_loader = DataLoader(dataset=batch_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

        cluster_net_file_name = 'cluster_net_%d.pkl' % (model_epoch)
        if os.path.exists(expt_path + cluster_net_file_name):
            print('Construct cluster_net. Load from pkl file.')
            cluster_net = torch.load(expt_path + cluster_net_file_name)
        else:
            raise Exception('No such model file.')

        cluster_net.float()
        cluster_net.eval()
        if use_cpu is False:
            cudnn.enabled = True
            cluster_net.cuda()
        else:
            cudnn.enabled = False
            cluster_net.cpu()

        test_res = None

        for batch_id, sample in enumerate(data_loader):

            data = sample['data'].unsqueeze(dim=1).float()
            age = sample['age'].float().numpy()

            if use_cpu is False:
                data = data.cuda()

            data = Variable(data)

            predicted_age = cluster_net(data)

            predicted_age = predicted_age.data.cpu().numpy()
            batch_res = np.c_[age, predicted_age, predicted_age - age]

            if test_res is None:
                test_res = batch_res
            else:
                test_res = np.r_[test_res, batch_res]

        test_res = test_res[test_res[:,0].argsort()]
        np.save(expt_path + 'cluster_test_result.npy', np.array(test_res))
        print(test_res)

        abs_error = abs(test_res[:,2])
        MAE = abs_error.mean()
        CI_75 = np.percentile(abs_error, 75)
        CI_95 = np.percentile(abs_error, 95)

        print('Test size: %d, MAE: %.3f, CI 75%%: %.3f, CI 95%%: %.3f. ' % (test_res.shape[0], MAE, CI_75, CI_95))

        utils.plot_scatter(test_res, CI_75, CI_95, expt_path, mode + '_cluster_' + str(model_epoch))