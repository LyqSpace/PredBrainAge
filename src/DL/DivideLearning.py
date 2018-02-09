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

from src.DL.Database import Database
from src.DL.BlockDataset import BlockDataset
from src.DL.ResNet import create_resnet
import src.utils as utils
from src.Logger import Logger


class DivideLearning:

    def __init__(self):
        self._database = Database()
        self._block_dataset = BlockDataset()

        self._divisor = 2
        self._divide_level_limit = 3
        self._child_num = self._divisor ** 3

        self._block_num = 0
        self._block_shape = []
        self._leaf_block_shape = np.array([22, 27, 22], dtype=int)

        for i in range(self._divide_level_limit):
            self._block_num += self._child_num ** i
            self._block_shape.append(self._leaf_block_shape * (self._divisor ** i))
        self._block_shape = np.flip(np.array(self._block_shape), 0)
        self._leaf_block_id_st = self._child_num ** (self._divide_level_limit - 2) + 1
        self._leaf_block_num = self._child_num ** (self._divide_level_limit - 1)

        self._attr_num = 8
        self._block_attr = np.zeros((self._block_num, self._attr_num))
        self._block_data = np.zeros(np.hstack((self._leaf_block_num, self._block_shape[-1])))
        self._block_am = np.zeros(np.hstack((self._leaf_block_num, self._block_shape[-1])))

        self._max_epoches = 1000
        self._batch_size = 32

    def divide(self, data_path, dataset_name, resample=False):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Divide data. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        data_num = self._database.get_data_size()
        block_attr = np.zeros(((data_num,) + self._block_attr.shape))
        block_data = np.zeros(((data_num,) + self._block_data.shape))
        block_am = np.zeros(((data_num,) + self._block_am.shape))

        while self._database.has_next_data():

            index = self._database.get_data_index()
            data_name, data, age = self._database.get_next_data()

            print(index, data_name, age)

            am = utils.get_attention_map(data)
            self._divide_block(0, data, am)

            block_attr[index] = self._block_attr
            block_data[index] = self._block_data
            block_am[index] = self._block_am

        np.save('experiments/block_attr.npy', block_attr)
        np.save('experiments/block_data.npy', block_data)
        np.save('experiments/block_am.npy', block_am)

        print('Done.')

    def _divide_block(self, block_id, data, am):

        divide_level = 0
        father_id = block_id
        while father_id > 0:
            father_id = int((father_id - 1) / self._child_num)
            divide_level += 1

        if divide_level == self._divide_level_limit:
            return

        # print(data.shape)
        if min(data.shape) < 5:
            utils.show_3Ddata_comp(data, am, 'ORIGIN')
            return

        template_center = self._block_shape[divide_level] / 2
        zoom_rate = np.zeros(3)
        data_center = np.zeros(3)
        for i in range(3):
            zoom_rate[i], data_center[i] = utils.get_zoom_parameter(template_center[i], am, i)
        offset = template_center - data_center

        # print(data.shape, template_center, data_center)

        zoomed_am = np.clip(ndimage.zoom(am, zoom_rate), a_min=0, a_max=None)
        zoomed_data = ndimage.zoom(data, zoom_rate)
        zoomed_data_center = data_center * zoom_rate
        zoomed_offset = template_center - zoomed_data_center

        data_mean = data.mean()
        am_mean = am.mean()

        self._block_attr[block_id] = np.r_[zoom_rate, offset, data_mean, am_mean]

        if divide_level == self._divide_level_limit - 1:

            proj_data = utils.get_template_proj(self._block_shape[divide_level], zoomed_data, zoomed_offset)
            proj_am = utils.get_template_proj(self._block_shape[divide_level], zoomed_am, zoomed_offset)

            self._block_data[block_id - self._leaf_block_id_st] = proj_data
            self._block_am[block_id - self._leaf_block_id_st] = proj_am

            # if block_id == 37:
            #     print(block_id, self._block_attr[block_id], data_center, zoomed_offset)
            #     utils.show_3Ddata_comp(am, proj_am, 'AM')
            return
        
        sub_block_id = block_id * 8
        zoomed_data_center = zoomed_data_center.astype(int)

        for i in range(2):

            if i == 0:
                block_anchor0 = [0, zoomed_data_center[0]]
            else:
                block_anchor0 = [zoomed_data_center[0], zoomed_data.shape[0]]

            for j in range(2):

                if j == 0:
                    block_anchor1 = [0, zoomed_data_center[1]]
                else:
                    block_anchor1 = [zoomed_data_center[1], zoomed_data.shape[1]]

                for k in range(2):

                    if k == 0:
                        block_anchor2 = [0, zoomed_data_center[2]]
                    else:
                        block_anchor2 = [zoomed_data_center[2], zoomed_data.shape[2]]

                    # print(block_anchor0, block_anchor1, block_anchor2)

                    sub_data = zoomed_data[
                               block_anchor0[0]:block_anchor0[1],
                               block_anchor1[0]:block_anchor1[1],
                               block_anchor2[0]:block_anchor2[1]]
                    sub_am = zoomed_am[
                             block_anchor0[0]:block_anchor0[1],
                             block_anchor1[0]:block_anchor1[1],
                             block_anchor2[0]:block_anchor2[1]]

                    sub_block_id += 1

                    self._divide_block(sub_block_id, sub_data, sub_am)

    def induce(self, data_path, dataset_name, retrain, use_cpu):

        print('Inductive Learning. Dataset {0}.'.format(dataset_name))

        self._database.load_database(data_path, dataset_name, mode='training')
        exper_path = 'experiments/'

        # block_attr = np.load(exper_path + 'block_attr.npy')
        block_data = np.load(exper_path + 'block_data.npy')
        # block_am = np.load(exper_path + 'block_am.npy')

        object_num = block_data.shape[0]
        object_ages = np.zeros(object_num)
        while self._database.has_next_data():

            index = self._database.get_data_index()
            data_name, data, age = self._database.get_next_data(required_data=False)
            object_ages[index] = age

        self._block_dataset.init_dataset(block_data, object_ages)

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

        if retrain is False and os.path.exists(exper_path + 'resnet.pkl'):
            print('Construct ResNet. Load from pkl file.')
            resnet = torch.load(exper_path + 'resnet.pkl')
        else:
            print('Construct ResNet. Create a new network.')
            resnet = create_resnet()

        resnet.float()
        resnet.train()
        if use_cpu is False:
            cudnn.enabled = True
            resnet.cuda()
        else:
            cudnn.enabled = False

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(resnet.parameters(), lr=1e-4, alpha=0.9)
        # optimizer = optim.SGD(net.parameters(), lr=st_lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)

        logger = Logger('train', 'train.log')

        for epoch in range(self._max_epoches):

            running_loss = 0

            for batch_id, sample in enumerate(data_loader):

                if use_cpu is False:
                    data, age = sample['data'].cuda(), sample['age'].cuda()
                else:
                    data, age = sample['data'], sample['age']

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

            torch.save(resnet, exper_path + 'resnet.pkl')


    def test(self, data_path, dataset_name, mode):

        print('Test model. Dataset {0}. Mode {1}'.format(dataset_name, mode))

        self._database.load_database(data_path, dataset_name, mode=mode)
        group_block_path = 'experiments/group_block/'

        group_block_params = np.load(group_block_path + 'group_block_params.npy')
        group_block_mean = np.load(group_block_path + 'group_block_mean.npy')
        cov_shape = np.r_[self._database.get_group_size(), 3, self._block_num, self._leaf_block_shape].astype(int)

        MAE = 0

        while self._database.has_next_data_from_dataset():

            data_name, test_data, test_age = self._database.get_next_data_from_dataset()
            test_am = utils.get_attention_map(test_data)

            print(self._database.get_data_from_dataset_index(), ' Age: ', test_age)

            self._divide_block(0, test_data, test_am)
            
            block_am = self._block_am - group_block_mean

            block_am_cov0 = np.zeros(block_am.shape)
            for i in range(1, self._leaf_block_shape[0]):
                block_am_cov0[:,:,i] = block_am[:,:,i-1] * block_am[:,:,i]

            block_am_cov1 = np.zeros(block_am.shape)
            for i in range(1, self._leaf_block_shape[1]):
                block_am_cov1[:,:,:,i] = block_am[:,:,:,i-1] * block_am[:,:,:,i]

            block_am_cov2 = np.zeros(block_am.shape)
            for i in range(1, self._leaf_block_shape[2]):
                block_am_cov2[:,:,:,:,i] = block_am[:,:,:,:,i-1] * block_am[:,:,:,:,i]

            block_am_cov = np.zeros(cov_shape)
            block_am_cov[:,0] = block_am_cov0
            block_am_cov[:,1] = block_am_cov1
            block_am_cov[:,2] = block_am_cov2

            group_block_div = abs(group_block_params - block_am_cov).sum(axis=(1,3,4,5))
            block_age = np.argmin(group_block_div, axis=0)
            unique, counts = np.unique(block_age, return_counts=True)
            age_stats = dict(zip(unique, counts))

            print(age_stats)

            predicted_age = 0
            error = abs(predicted_age - test_age)
            print(' Err: ', error)

            MAE += error

        MAE /= self._database.get_data_from_dataset_size()
        print('Total MAE: ', MAE)

