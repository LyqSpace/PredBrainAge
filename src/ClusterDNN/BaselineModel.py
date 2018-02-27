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
from src.ClusterDNN.BaselineNet import create_baseline_net
import src.utils as utils
from src.Logger import Logger


class BaselineModel:

    def __init__(self, data_path, mode, use_cpu):

        np.set_printoptions(precision=3, suppress=True)

        self._use_cpu = use_cpu
        self._mode = mode
        self._expt_path = 'expt/baseline/'

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
            num_workers = 2
            pin_memory = True

        batch_size = 16

        self._data_loader = DataLoader(dataset=batch_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    def train(self, st_epoch=0):

        print('Train Dataset.')

        max_epoches = 100000
        lr0 = 1e-2
        lr_shirnk_step = 5000
        lr_shrink_gamma = 0.5

        if st_epoch < 1:
            print('Construct baseline model. Create a new network.')
            baseline_net = create_baseline_net()
        else:
            baseline_net_file_name = 'baseline_net_%d.pkl' % (st_epoch-1)
            if os.path.exists(self._expt_path + baseline_net_file_name):
                print('Construct baseline model. Load from pkl file.')
                baseline_net = torch.load(self._expt_path + baseline_net_file_name)
            else:
                print('Construct baseline model. Create a new network.')
                baseline_net = create_baseline_net()

        baseline_net.float()
        baseline_net.train()
        if self._use_cpu is False:
            cudnn.enabled = True
            baseline_net.cuda()
        else:
            cudnn.enabled = False

        lr = lr0 * lr_shrink_gamma ** (st_epoch // lr_shirnk_step)

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(baseline_net.parameters(), lr=lr, alpha=0.9)
        # optimizer = optim.SGD(net.parameters(), lr=st_lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_shirnk_step, gamma=lr_shrink_gamma)

        logger = Logger('train', 'train_baseline.log')

        for epoch in range(st_epoch, max_epoches):

            running_loss = 0

            for batch_id, sample in enumerate(self._data_loader):

                data, age = sample['data'].unsqueeze(dim=1).float(), sample['age'].unsqueeze(dim=1).float()

                if self._use_cpu is False:
                    data, age = data.cuda(), age.cuda()

                data, age = Variable(data), Variable(age)

                predicted_age = baseline_net(data)

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

            torch.save(baseline_net, self._expt_path + 'baseline_net_%d.pkl' % epoch)

    def test(self, model_epoch):

        print('Test model.', 'Mode:', self._mode)

        baseline_net_file_name = 'baseline_net_%d.pkl' % model_epoch
        if os.path.exists(self._expt_path + baseline_net_file_name):
            print('Construct baseline_net. Load from pkl file.')
            baseline_net = torch.load(self._expt_path + baseline_net_file_name)
        else:
            raise Exception('No such model file.')

        baseline_net.float()
        baseline_net.eval()
        if self._use_cpu is False:
            cudnn.enabled = True
            baseline_net.cuda()
        else:
            cudnn.enabled = False
            baseline_net.cpu()

        test_res = None

        for batch_id, sample in enumerate(self._data_loader):

            data = sample['data'].unsqueeze(dim=1).float()
            age = sample['age'].float().numpy()

            if self._use_cpu is False:
                data = data.cuda()

            data = Variable(data)

            predicted_age = baseline_net(data)

            predicted_age = predicted_age.data.cpu().numpy()
            batch_res = np.c_[age, predicted_age, predicted_age - age]

            if test_res is None:
                test_res = batch_res
            else:
                test_res = np.r_[test_res, batch_res]

        test_res = test_res[test_res[:,0].argsort()]
        np.save(self._expt_path + self._mode + 'baseline_test_result.npy', np.array(test_res))
        print(test_res)

        abs_error = abs(test_res[:,2])
        MAE = abs_error.mean()
        CI_75 = np.percentile(abs_error, 75)
        CI_95 = np.percentile(abs_error, 95)

        print('Test size: %d, MAE: %.3f, CI 75%%: %.3f, CI 95%%: %.3f. ' % (test_res.shape[0], MAE, CI_75, CI_95))

        utils.plot_scatter(test_res, CI_75, CI_95, self._expt_path, self._mode + '_baseline_' + str(model_epoch))
