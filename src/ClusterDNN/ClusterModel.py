import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from scipy import ndimage, stats
import numpy as np
from torch.utils.data import DataLoader

from src.ClusterDNN.Database import Database
from src.ClusterDNN.BatchSet import BatchSet
from src.ClusterDNN.BaselineNet import create_baseline_net
import src.utils as utils
from src.Logger import Logger


class ClusterModel:

    def __init__(self):
        self._database = Database()

        self._max_epoches = 1000
        self._batch_size = 32
        self._lr0 = 1e-2
        self._lr_shirnk_step = 50000
        self._lr_shrink_gamma = 0.5

    def train(self, data_path, dataset_name, resample, retrain, use_cpu, st_epoch=0):

        print('Train Dataset {0}.'.format(dataset_name))

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)
        batch_set = BatchSet()
        training_data = np.zeros(self._database.get_data_size())
        training_ages = np.zeros(self._database.get_data_size())

        while self._database.has_next_data():

            index = self._database.get_data_index()
            data_name, data, age = self._database.get_next_data(required_data=True)
            training_data[index] = data
            training_ages[index] = age

        batch_set.init(block_data, object_ages)

        if use_cpu:
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 2
            pin_memory = True
        data_loader = DataLoader(self._block_dataset,
                                 batch_size=self._batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

        if (st_epoch-1) < 0 or retrain:
            print('Construct ResNet. Create a new network.')
            resnet = create_baseline_net()
        else:
            resnet_file_name = 'resnet_%d.pkl' % (st_epoch-1)
            if os.path.exists(exper_path + resnet_file_name):
                print('Construct ResNet. Load from pkl file.')
                resnet = torch.load(exper_path + resnet_file_name)
            else:
                print('Construct ResNet. Create a new network.')
                resnet = create_baseline_net()

        resnet.float()
        resnet.train()
        if use_cpu is False:
            cudnn.enabled = True
            resnet.cuda()
        else:
            cudnn.enabled = False

        epoch_step = self._lr_shirnk_step * self._batch_size / block_num
        lr = self._lr0 * self._lr_shrink_gamma ** (st_epoch // epoch_step)

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(resnet.parameters(), lr=lr, alpha=0.9)
        # optimizer = optim.SGD(net.parameters(), lr=st_lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self._lr_shirnk_step, gamma=self._lr_shrink_gamma)

        logger = Logger('train', 'train.log')

        for epoch in range(st_epoch, self._max_epoches):

            running_loss = 0

            for batch_id, sample in enumerate(data_loader):

                data, age = sample['data'].unsqueeze(dim=1).float(), sample['age'].unsqueeze(dim=1).float()

                if use_cpu is False:
                    data, age = data.cuda(), age.cuda()

                data, age = Variable(data), Variable(age)

                predicted_age = resnet(data)

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

            torch.save(resnet, exper_path + 'resnet_%d.pkl' % epoch)

    def test(self, data_path, dataset_name, model_epoch, mode, use_cpu):

        print('Test model. Dataset {0}. Mode {1}'.format(dataset_name, mode))

        self._database.load_database(data_path, dataset_name, mode=mode)
        exper_path = 'experiments/'

        resnet_file_name = 'resnet_%d.pkl' % (model_epoch)
        if os.path.exists(exper_path + resnet_file_name):
            print('Construct ResNet. Load from pkl file.')
            resnet = torch.load(exper_path + resnet_file_name)
        else:
            raise Exception('No such model file.')

        resnet.float()
        resnet.eval()
        if use_cpu is False:
            cudnn.enabled = True
            resnet.cuda()
        else:
            cudnn.enabled = False
            resnet.cpu()

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

            predicted_age = resnet(data)

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
