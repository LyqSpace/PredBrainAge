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
        self._lr0 = 1e-2
        self._lr_shirnk_step = 50000
        self._lr_shrink_gamma = 0.5

    def divide(self, data_path, dataset_name, resample=False):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Divide data. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        data_num = self._database.get_data_size()
        block_attr = np.zeros(((data_num,) + self._block_attr.shape))
        block_data = np.zeros(((data_num,) + self._block_data.shape))
        block_am = np.zeros(((data_num,) + self._block_am.shape))

        logger = Logger('divide', 'divide.log')

        while self._database.has_next_data():

            index = self._database.get_data_index()
            data_name, data, age = self._database.get_next_data()

            print(index, data_name, age)
            message = 'Index: %d, Name %s, Age: %d' % (index, data_name, age)
            logger.log(message)

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

    def induce(self, data_path, dataset_name, retrain, use_cpu, st_epoch=0):

        print('Inductive Learning. Dataset {0}.'.format(dataset_name))

        self._database.load_database(data_path, dataset_name, mode='training')
        exper_path = 'experiments/'

        # block_attr = np.load(exper_path + 'block_attr.npy')
        block_data = np.load(exper_path + 'block_data.npy')
        # block_am = np.load(exper_path + 'block_am.npy')

        object_num = block_data.shape[0]
        block_num = object_num * block_data.shape[1]
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

        if (st_epoch-1) < 0 or retrain:
            print('Construct ResNet. Create a new network.')
            resnet = create_resnet()
        else:
            resnet_file_name = 'resnet_%d.pkl' % (st_epoch-1)
            if os.path.exists(exper_path + resnet_file_name):
                print('Construct ResNet. Load from pkl file.')
                resnet = torch.load(exper_path + resnet_file_name)
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

            loss += ((predicted_age - test_age) ** 2).mean()
            predicted_age = predicted_age.mean()
            error = abs(predicted_age - test_age)
            MAE += error

            test_result_list.append((test_age, predicted_age, error))

            print('Id: %d, Test Age: %d, Pred Age: %d, Err: %d, Loss %.3f, MAE: %.3f' % (
                index, test_age, predicted_age, error, loss / (index + 1), MAE / (index + 1)
            ))

            if index > 0:
                break

        MAE /= data_num

        test_result_list.sort(key=lambda data: data[2])
        CI_75 = test_result_list[int(0.75 * data_num)][2]
        CI_95 = test_result_list[int(0.95 * data_num)][2]

        print('Test size: %d, MAE: %.3f, CI 75%%: %.3f, CI 95%%: %.3f, ' % (data_num, MAE, CI_75, CI_95))

        utils.plot_scatter(test_result_list, CI_75, CI_95, exper_path)
