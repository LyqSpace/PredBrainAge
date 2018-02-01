from scipy import ndimage, stats
import numpy as np

from src.DL.Database import Database
import src.utils as utils


class DivideLearning:

    def __init__(self):
        self._database = Database()

        self._divisor = 2
        self._divide_level_limit = 4
        self._block_overlap = 0.1
        self._child_num = self._divisor ** 3
        self._block_num = 0
        for i in range(self._divide_level_limit):
            self._block_num += self._child_num ** i

        self._attr_num = 11
        self._block_attr = np.zeros((self._block_num, self._attr_num))
        self._significant_block_num = 50

    def load_group(self):
        pass

    def train(self, data_path, dataset_name, st_group=None, resample=False):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Divide data. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        while self._database.get_next_group() is not None:

            group_age = self._database.get_group_index() - 1

            print('\tGroup Age: {0}. '.format(group_age), end='')

            if st_group is not None and group_age < st_group:
                print('Pass.')
                continue

            template_id, template_data, template_age = self._database.get_next_data_from_group()
            template_am = utils.get_attention_map(template_data)

            data_from_group_num = self._database.get_data_from_group_size() - 1
            group_block_attr = np.zeros((data_from_group_num, self._block_num, self._attr_num))

            while self._database.has_next_data_from_group():

                print(' ', str(self._database.get_data_from_group_index()))

                data_name, matched_data, age = self._database.get_next_data_from_group()
                matched_am = utils.get_attention_map(matched_data)
                self._divide(0, template_data, template_am, matched_data, matched_am)

                group_block_attr[self._database.get_data_from_group_index() - 2] = self._block_attr

            print(' ')
            np.save('experiments/group_block_attr/' + str(group_age) + '.npy', group_block_attr)

        print('Done.')

    def _divide(self, block_id, template_data, template_am, matched_data, matched_am):

        divide_level = 0
        father_id = block_id
        while father_id > 0:
            father_id = int((father_id - 1) / self._child_num)
            divide_level += 1

        if divide_level == self._divide_level_limit:
            return

        if min(template_data.shape) <= 2 or min(matched_data.shape) <= 2:
            return

        zoom_rate = np.zeros(3)
        template_center = np.zeros(3, dtype=int)
        matched_center = np.zeros(3, dtype=int)
        for i in range(3):
            zoom_rate[i], template_center[i], matched_center[i] = utils.get_zoom_parameter(template_am, matched_am, i)
        offset = template_center - matched_center

        # print(block_id, zoom_rate, template_am.shape, matched_am.shape, template_center, matched_center)

        zoomed_matched_am = np.clip(ndimage.zoom(matched_am, zoom_rate), a_min=0, a_max=None)
        zoomed_matched_data = np.clip(ndimage.zoom(matched_data, zoom_rate), a_min=0, a_max=None)
        zoomed_matched_center = (matched_center * zoom_rate).astype(int)
        zoomed_offset = template_center - zoomed_matched_center

        # utils.show_3Ddata_comp(template_am, matched_am, 'PRE')
        # utils.show_3Ddata_comp(template_am, zoomed_matched_am, 'AFTER')

        zoomed_matched_am_proj = utils.get_template_proj(template_am, zoomed_matched_am, zoomed_offset)
        zoomed_matched_data_proj = utils.get_template_proj(template_data, zoomed_matched_data, zoomed_offset)

        am_mean = template_am.mean()
        am_div = utils.get_div(template_am, zoomed_matched_am_proj)
        am_significant = (am_mean + 1e-8) / (am_div + 1e-8)
        intensity_mean = template_data.mean() - zoomed_matched_data_proj.mean()
        valid = 1

        self._block_attr[block_id] = np.r_[zoom_rate, offset, intensity_mean, am_mean, am_div, am_significant, valid]

        # print(block_id, self._block_attr[block_id])
        # print(template_am.shape, template_center, matched_am.shape, matched_center)

        # if divide_level > 2:
        #     utils.show_3Ddata_comp(template_am, zoomed_matched_am_proj, 'PROJECTION')

        if divide_level == self._divide_level_limit - 1:
            return
        
        sub_block_id = block_id * 8

        for i in range(2):

            template_block_overlap = int(self._block_overlap * template_am.shape[0])
            matched_block_overlap = int(self._block_overlap * matched_am.shape[0])
            if i == 0:
                template_block_anchor0 = [0, template_center[0] + template_block_overlap]
                matched_block_anchor0 = [0, zoomed_matched_center[0] + matched_block_overlap]
            else:
                template_block_anchor0 = [template_center[0] - template_block_overlap, template_am.shape[0]]
                matched_block_anchor0 = [zoomed_matched_center[0] - matched_block_overlap, matched_am.shape[0]]
                
            for j in range(2):

                template_block_overlap = int(self._block_overlap * template_am.shape[1])
                matched_block_overlap = int(self._block_overlap * matched_am.shape[1])
                if j == 0:
                    template_block_anchor1 = [0, template_center[1] + template_block_overlap]
                    matched_block_anchor1 = [0, zoomed_matched_center[1] + matched_block_overlap]
                else:
                    template_block_anchor1 = [template_center[1] - template_block_overlap, template_am.shape[1]]
                    matched_block_anchor1 = [zoomed_matched_center[1] - matched_block_overlap, matched_am.shape[1]]

                for k in range(2):

                    template_block_overlap = int(self._block_overlap * template_am.shape[2])
                    matched_block_overlap = int(self._block_overlap * matched_am.shape[2])
                    if k == 0:
                        template_block_anchor2 = [0, template_center[2] + template_block_overlap]
                        matched_block_anchor2 = [0, zoomed_matched_center[2] + matched_block_overlap]
                    else:
                        template_block_anchor2 = [template_center[2] - template_block_overlap, template_am.shape[2]]
                        matched_block_anchor2 = [zoomed_matched_center[2] - matched_block_overlap, matched_am.shape[2]]

                    sub_template_data = template_data[
                                            template_block_anchor0[0]:template_block_anchor0[1],
                                            template_block_anchor1[0]:template_block_anchor1[1],
                                            template_block_anchor2[0]:template_block_anchor2[1]]
                    sub_template_am = template_am[
                                          template_block_anchor0[0]:template_block_anchor0[1],
                                          template_block_anchor1[0]:template_block_anchor1[1],
                                          template_block_anchor2[0]:template_block_anchor2[1]]
                    sub_matched_data = zoomed_matched_data[
                                           matched_block_anchor0[0]:matched_block_anchor0[1],
                                           matched_block_anchor1[0]:matched_block_anchor1[1],
                                           matched_block_anchor2[0]:matched_block_anchor2[1]]
                    sub_matched_am = zoomed_matched_am[
                                         matched_block_anchor0[0]:matched_block_anchor0[1],
                                         matched_block_anchor1[0]:matched_block_anchor1[1],
                                         matched_block_anchor2[0]:matched_block_anchor2[1]]

                    sub_block_id += 1

                    self._divide(sub_block_id, sub_template_data, sub_template_am, sub_matched_data, sub_matched_am)

    def induce(self, data_path, dataset_name):

        print('Inductive Learning. Dataset {0}.'.format(dataset_name))

        self._database.load_database(data_path, dataset_name, mode='training')
        group_block_path = 'experiments/group_block_attr/'

        group_block_params = np.zeros((self._database.get_group_size(), self._block_num))

        for group_id in range(self._database.get_group_size()):

            group_age = group_id + self._database.get_group_st()

            print('group age ', group_age)

            group_block_attr = np.load(group_block_path + str(group_age) + '.npy')
            group_block_attr = group_block_attr[:,:,8] * (1/group_block_attr[:,:,10])

            group_block_pct = [np.percentile(group_block_attr[:, block_id], 80) for block_id in range(self._block_num)]

            group_block_params[group_id] = np.array(group_block_pct)

        np.save(group_block_path + 'group_block_params.npy', group_block_params)

    def test(self, data_path, dataset_name, mode):

        print('Test model. Dataset {0}. Mode {1}'.format(dataset_name, mode))

        self._database.load_database(data_path, dataset_name, mode=mode)
        group_block_path = 'experiments/group_block_attr/'

        group_block_params = np.load(group_block_path + 'group_block_params.npy')

        MAE = 0

        while self._database.has_next_data_from_dataset():

            data_name, test_data, test_age = self._database.get_next_data_from_dataset()
            test_am = utils.get_attention_map(test_data)

            print(self._database.get_data_from_dataset_index(), ' Age ', test_age)

            # self._database.set_group_index(int(test_age))
            # self._database.set_group_index(33)
            self._database.set_group_index()

            best_group_age = 0
            best_significant = np.inf

            while self._database.get_next_group() is not None:

                group_age = self._database.get_group_index() - 1
                group_id = group_age - self._database.get_group_st()

                print('\tGroup Age: {0}. '.format(group_age), end='')

                template_id, template_data, template_age = self._database.get_next_data_from_group()
                template_am = utils.get_attention_map(template_data)

                self._divide(0, template_data, template_am, test_data, test_am)

                block_attr = self._block_attr[:, 8] * ( 1/self._block_attr[:, 10] )
                block_sum = 0
                # matched_block_sum = 0
                significant_sum = 0
                for block_id in range(self._block_num):
                    if group_block_params[group_id, block_id] > 0.2:
                        continue
                    # if group_age == 39:
                    # print(block_id, block_attr[block_id], group_block_params[group_id, block_id])
                    block_sum += 1
                    significant_sum += block_attr[block_id]
                    # if block_attr[block_id] > group_block_params[group_id, block_id]:
                    #     matched_block_sum += 1

                significant_avg = significant_sum / block_sum

                if best_significant > significant_avg:
                    best_significant = significant_avg
                    best_group_age = group_age

                print('#blocks:', block_sum, 's:', significant_avg, 'best_age:', best_group_age)

            error = abs(best_group_age - test_age)
            print(' Err: ', error)

            MAE += error

        MAE /= self._database.get_data_from_dataset_size()
        print('Total MAE: ', MAE)

