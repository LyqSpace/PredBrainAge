from scipy import ndimage
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

        group_significant_blocks = np.zeros((self._database.get_group_size(), self._block_num))

        for group_id in range(self._database.get_group_size()):

            group_age = group_id + self._database.get_group_st()

            print('group age ', group_age)

            group_block_attr = np.load(group_block_path + str(group_age) + '.npy')

            significant_block = [(group_block_attr[:,block_id,9] * group_block_attr[:,block_id,10]).min()
                                 for block_id in range(self._block_num)]
            top_significant = np.flip(np.argsort(significant_block), 0)
            top_significant = np.sort(top_significant[:self._significant_block_num])

            for i in range(self._significant_block_num):
                group_significant_blocks[group_id,top_significant[i]] = 1

        block_weights = np.zeros(self._block_num)
        for block_id in range(self._block_num):
            significant_sum = group_significant_blocks[:,block_id].sum()
            if significant_sum > 0:
                block_weights[block_id] = 1.0 / significant_sum

        group_block_weights = np.zeros((self._database.get_group_size(), self._block_num))
        for group_id in range(self._database.get_group_size()):

            for block_id in range(self._block_num):
                if group_significant_blocks[group_id, block_id] == 0:
                    continue
                group_block_weights[group_id,block_id] = block_weights[block_id]

            weight_sum = group_block_weights[group_id,:].sum()
            group_block_weights[group_id,:] /= weight_sum

        np.save(group_block_path + 'group_block_weights.npy', group_block_weights)

    def test(self, data_path, dataset_name, mode):

        print('Test model. Dataset {0}. Mode {1}'.format(dataset_name, mode))

        self._database.load_database(data_path, dataset_name, mode=mode)
        group_block_path = 'experiments/group_block_attr/'

        group_block_weights = np.load(group_block_path + 'group_block_weights.npy')

        MAE = 0

        while self._database.has_next_data_from_dataset():

            data_name, test_data, test_age = self._database.get_next_data_from_dataset()
            test_am = utils.get_attention_map(test_data)

            print(self._database.get_data_from_dataset_index(), ' Age ', test_age)

            self._database.set_group_index(int(test_age)+20)
            # self._database.set_group_index()

            group_confidence = np.zeros(self._database.get_group_size())

            while self._database.get_next_group() is not None:

                group_age = self._database.get_group_index() - 1
                group_id = group_age - self._database.get_group_st()

                print('\tGroup Age: {0}. '.format(group_age), end='')

                template_id, template_data, template_age = self._database.get_next_data_from_group()
                template_am = utils.get_attention_map(template_data)

                self._divide(0, template_data, template_am, test_data, test_am)

                significant_block = self._block_attr[:,9] * self._block_attr[:,10]
                top_significant = np.flip(np.argsort(significant_block), 0)
                top_significant = np.sort(top_significant[:self._significant_block_num])

                confidence = 0
                for block_id in top_significant:
                    confidence += group_block_weights[group_id, block_id]
                    print(block_id, group_block_weights[group_id, block_id])

                group_confidence[group_id] = confidence

                print('CF: ', confidence)

                group_block_attr = np.load(group_block_path + str(group_age) + '.npy')
                group_block_attr = group_block_attr[:, :, 9] * group_block_attr[:, :, 10]

                min_np = [group_block_attr[:,i].min() for i in range(group_block_attr.shape[1])]
                avg_np = [group_block_attr[:,i].mean() for i in range(group_block_attr.shape[1])]

                comp_np = np.vstack((group_block_attr, min_np, avg_np, significant_block))
                print(comp_np)

            top_confidence = np.flip(np.argsort(group_confidence), 0)

            top5_match = 'N'
            for i in range(5):
                group_age = top_confidence[i] + self._database.get_group_st()
                print('Age: ', group_age, ' CF: ', group_confidence[top_confidence[i]], end='')
                if group_age == test_age:
                    top5_match = 'Y'

            error = abs(top_confidence[0] + self._database.get_group_st() - test_age)
            print(' ', top5_match, ' Err: ', error)

            MAE += error

        MAE /= self._database.get_data_from_dataset_size()
        print('Total MAE: ', MAE)

