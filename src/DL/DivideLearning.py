from scipy import ndimage, stats
import numpy as np

from src.DL.Database import Database
import src.utils as utils


class DivideLearning:

    def __init__(self):
        self._database = Database()

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
        self._leaf_block_ids = range(self._child_num ** (self._divide_level_limit - 2) + 1, self._block_num)

        self._attr_num = 8
        self._block_attr = np.zeros((self._block_num, self._attr_num))
        self._block_data = np.zeros(np.hstack((self._block_num, self._block_shape[-1])))
        self._block_am = np.zeros(np.hstack((self._block_num, self._block_shape[-1])))
        self._significant_block_num = 50

    def train(self, data_path, dataset_name, st_group=None, resample=False):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Divide data. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        while self._database.get_next_group() is not None:

            group_age = self._database.get_cur_group_age()

            print('\tGroup Age: {0}.'.format(group_age), end=' ')

            if st_group is not None and group_age < st_group:
                print('Pass.')
                continue

            data_from_group_num = self._database.get_data_from_group_size()
            group_block_attr = np.zeros(((data_from_group_num,) + self._block_attr.shape))
            group_block_data = np.zeros(((data_from_group_num,) + self._block_data.shape))
            group_block_am = np.zeros(((data_from_group_num,) + self._block_am.shape))

            while self._database.has_next_data_from_group():

                index = self._database.get_data_from_group_index()
                print(index, end=' ')

                data_name, data, age = self._database.get_next_data_from_group()
                am = utils.get_attention_map(data)
                self._divide(0, data, am)

                group_block_attr[index] = self._block_attr
                group_block_data[index] = self._block_data
                group_block_am[index] = self._block_am

            print(' ')
            # self._database.set_group_index(50)
            # continue
            np.save('experiments/group_block/' + str(group_age) + '_attr.npy', group_block_attr)
            np.save('experiments/group_block/' + str(group_age) + '_data.npy', group_block_data)
            np.save('experiments/group_block/' + str(group_age) + '_am.npy', group_block_am)

        print('Done.')

    def _divide(self, block_id, data, am):

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

        # print(block_id, self._block_attr[block_id], data_center, zoomed_offset)

        # print(block_id, self._block_attr[block_id])
        # print(template_am.shape, template_center, matched_am.shape, matched_center)

        # if divide_level > 2:
        #     utils.show_3Ddata_comp(template_am, zoomed_matched_am_proj, 'PROJECTION')

        if divide_level == self._divide_level_limit - 1:

            proj_data = utils.get_template_proj(self._block_shape[divide_level], zoomed_data, zoomed_offset)
            proj_am = utils.get_template_proj(self._block_shape[divide_level], zoomed_am, zoomed_offset)

            self._block_data[block_id] = proj_data
            self._block_am[block_id] = proj_am

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

                    self._divide(sub_block_id, sub_data, sub_am)

    def induce(self, data_path, dataset_name):

        print('Inductive Learning. Dataset {0}.'.format(dataset_name))

        self._database.load_database(data_path, dataset_name, mode='training')
        group_block_path = 'experiments/group_block/'

        param_shape = np.r_[self._database.get_group_size(), 3, self._block_num, self._leaf_block_shape].astype(int)
        mean_shape = np.r_[self._database.get_group_size(), self._block_num, self._leaf_block_shape].astype(int)
        cov_shape = np.r_[3, self._block_num, self._leaf_block_shape].astype(int)

        group_block_params = np.zeros(param_shape)
        group_block_mean = np.zeros(mean_shape)

        while self._database.get_next_group() is not None:

            group_age = self._database.get_cur_group_age()

            print('Group Age: ', group_age)

            if group_age < 30: continue

            group_block_attr = np.load(group_block_path + str(group_age) + '_attr.npy')
            group_block_data = np.load(group_block_path + str(group_age) + '_data.npy')
            group_block_am = np.load(group_block_path + str(group_age) + '_am.npy')

            # for object_id in range(group_block_am.shape[0]):
            #     print(object_id, group_block_attr[object_id, 30])
            #     utils.show_3Ddata_comp(group_block_data[object_id,30], group_block_am[object_id,30])

            group_block_data_mean = group_block_data.mean(axis=0)
            # group_block_am_mean = group_block_am.mean(axis=0)

            group_block_data -= group_block_data_mean
            # group_block_am -= group_block_am_mean

            block_data_cov0 = np.zeros(group_block_data.shape)
            for i in range(1, self._leaf_block_shape[0]):
                block_data_cov0[:,:,i] = group_block_data[:,:,i-1] * group_block_data[:,:,i]

            block_data_cov1 = np.zeros(group_block_data.shape)
            for i in range(1, self._leaf_block_shape[1]):
                block_data_cov1[:,:,:,i] = group_block_data[:,:,:,i-1] * group_block_data[:,:,:,i]

            block_data_cov2 = np.zeros(group_block_data.shape)
            for i in range(1, self._leaf_block_shape[2]):
                block_data_cov2[:,:,:,:,i] = group_block_data[:,:,:,:,i-1] * group_block_data[:,:,:,:,i]

            block_data_cov = np.zeros(cov_shape)
            block_data_cov[0] = block_data_cov0.mean(axis=0)
            block_data_cov[1] = block_data_cov1.mean(axis=0)
            block_data_cov[2] = block_data_cov2.mean(axis=0)

            for block_id in self._leaf_block_ids:
                block_id += 25
                print(block_id)
                utils.show_3Ddata(block_data_cov[0,block_id])
                if block_id >= 35:
                    break

            np.save(group_block_path + str(group_age) + '_cov.npy', block_data_cov)

            group_block_params[self._database.get_group_index()-1] = block_data_cov
            group_block_mean[self._database.get_group_index()-1] = group_block_data_mean

        np.save(group_block_path + 'group_block_params.npy', group_block_params)
        np.save(group_block_path + 'group_block_mean.npy', group_block_mean)

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

            self._divide(0, test_data, test_am)
            
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

