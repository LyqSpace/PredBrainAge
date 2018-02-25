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

    def __init__(self, data_path, dataset_name, resample, ):

        print('Initialize database.', data_path, dataset_name, 'Resample:', resample)

        if not resample:
            return

        database = Database()
        database.load_database(data_path, dataset_name, mode='training', resample=resample)

        training_data = []
        training_ages = []

        while database.has_next_data():
            data_name, data, age = database.get_next_data(required_data=True)
            training_data.append(data)
            training_ages.append(age)

        np.save(data_path + 'training_data.npy', np.array(training_data))
        np.save(data_path + 'training_ages.npy', np.array(training_ages))

    def train(self, data_path, retrain, use_cpu, st_epoch=0):

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

    def test(self, data_path, dataset_name, model_epoch, mode, use_cpu):

        print('Test model. Dataset {0}. Mode {1}'.format(dataset_name, mode))

        self._database.load_database(data_path, dataset_name, mode=mode)
        exper_path = 'experiments/'

        baseline_net_file_name = 'baseline_net_%d.pkl' % (model_epoch)
        if os.path.exists(exper_path + baseline_net_file_name):
            print('Construct baseline_net. Load from pkl file.')
            baseline_net = torch.load(exper_path + baseline_net_file_name)
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

        MAE = 0
        loss = 0
        test_result_list = []
        data_num = self._database.get_data_size()

        fa_index = np.array(range(64)) // 8
        test_res = []

        while self._database.has_next_data():

            index = self._database.get_data_index()
            data_name, test_data, test_age = self._database.get_next_data()
            test_am = utils.get_attention_map(test_data)

            self._divide_block(0, test_data, test_am)

            data = torch.from_numpy(self._block_data).unsqueeze(dim=1).float()
            if use_cpu is False:
                data = data.cuda()
            data = Variable(data)

            predicted_age = baseline_net(data)

            predicted_age = predicted_age.data.cpu().numpy()
            comp = np.c_[predicted_age, predicted_age - test_age, fa_index]
            test_res.append(comp)
            print(comp)

            loss += ((predicted_age - test_age) ** 2).mean()
            predicted_age = predicted_age.mean()
            error = abs(predicted_age - test_age)
            MAE += error

            test_result_list.append((test_age, predicted_age, error))

            print('Id: %d, Test Age: %d, Pred Age: %d, Err: %d, Loss %.3f, MAE: %.3f' % (
                index, test_age, predicted_age, error, loss / (index + 1), MAE / (index + 1)
            ))

            # if index > 0:
            #     break

        np.save(exper_path + 'test_res.npy', np.array(test_res))

        MAE /= data_num

        test_result_list.sort(key=lambda data: data[2])
        CI_75 = test_result_list[int(0.75 * data_num)][2]
        CI_95 = test_result_list[int(0.95 * data_num)][2]

        # CI_75 = 0
        # CI_95 = 0

        print('Test size: %d, MAE: %.3f, CI 75%%: %.3f, CI 95%%: %.3f, ' % (data_num, MAE, CI_75, CI_95))

        utils.plot_scatter(test_result_list, CI_75, CI_95, exper_path)
