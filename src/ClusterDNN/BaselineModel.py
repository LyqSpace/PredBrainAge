import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
from torch.utils.data import DataLoader

from src.ClusterDNN.Database import Database
from src.ClusterDNN.BatchSet import BatchSet
from src.ClusterDNN.BaselineNet import create_baseline_net
import src.utils as utils
from src.Logger import Logger


class BaselineModel:

    def __init__(self):
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
            print('Construct baseline model. Create a new network.')
            baseline_net = create_baseline_net()
        else:
            baseline_net_file_name = 'baseline_net_%d.pkl' % (st_epoch-1)
            if os.path.exists(expt_path + baseline_net_file_name):
                print('Construct baseline model. Load from pkl file.')
                baseline_net = torch.load(expt_path + baseline_net_file_name)
            else:
                print('Construct baseline model. Create a new network.')
                baseline_net = create_baseline_net()

        baseline_net.float()
        baseline_net.train()
        if use_cpu is False:
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

            for batch_id, sample in enumerate(data_loader):

                data, age = sample['data'].unsqueeze(dim=1).float(), sample['age'].unsqueeze(dim=1).float()

                if use_cpu is False:
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

                comp_res = np.c_[predicted_age.data.cpu().numpy(), age.data.cpu().numpy()[:, :, 0]]
                print(comp_res)
                logger.log(comp_res)

            torch.save(baseline_net, expt_path + 'baseline_net_%d.pkl' % epoch)

    @staticmethod
    def test(data_path, model_epoch, mode, use_cpu):

        print('Test model.', data_path, 'Mode:', mode)

        expt_path = 'expt/'

        if mode == 'validation':
            test_data = np.load(data_path + 'validation_data.npy')
            test_ages = np.load(data_path + 'validation_ages.npy')
        else:
            test_data = np.load(data_path + 'test_data.npy')
            test_ages = np.load(data_path + 'test_ages.npy')

        baseline_net_file_name = 'baseline_net_%d.pkl' % (model_epoch)
        if os.path.exists(expt_path + baseline_net_file_name):
            print('Construct baseline_net. Load from pkl file.')
            baseline_net = torch.load(expt_path + baseline_net_file_name)
        else:
            raise Exception('No such model file.')

        baseline_net.float()
        baseline_net.eval()
        if use_cpu is False:
            cudnn.enabled = True
            baseline_net.cuda()
        else:
            cudnn.enabled = False
            baseline_net.cpu()

        data = torch.from_numpy(test_data).unsqueeze(dim=1).float()
        if use_cpu is False:
            data = data.cuda()
        data = Variable(data)

        predicted_age = baseline_net(data)

        predicted_age = predicted_age.data.cpu().numpy()

        error = predicted_age - test_ages
        MAE = abs(error).mean()

        np.save(expt_path + 'baseline_test_result.npy', np.array(test_res))

        MAE /= data_num

        test_result_list.sort(key=lambda data: data[2])
        CI_75 = test_result_list[int(0.75 * data_num)][2]
        CI_95 = test_result_list[int(0.95 * data_num)][2]

        # CI_75 = 0
        # CI_95 = 0

        print('Test size: %d, MAE: %.3f, CI 75%%: %.3f, CI 95%%: %.3f, ' % (data_num, MAE, CI_75, CI_95))

        utils.plot_scatter(test_result_list, CI_75, CI_95, expt_path)
