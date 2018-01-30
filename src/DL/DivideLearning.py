from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

from src.DL.Database import Database
import src.utils as utils


class DivideLearning:

    def __init__(self):
        self._database = Database()

        self._divisor = 2
        self._divide_level_limit = 5
        self._block_overlap = 0.1
        self._child_num = self._divisor ** 3
        self._block_num = 0
        for i in range(self._divide_level_limit):
            self._block_num += self._child_num ** i

        self._attr_num = 10
        self._block_attr = np.zeros((self._block_num, self._attr_num))
        self._block_significant = np.zeros(self._block_num, dtype=bool)

        self._zoom_rates = np.arange(0.7, 1.4, 0.1)

    def load_group(self):
        pass

    def train(self, data_path, dataset_name, st_group=None, resample=False):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Divide data. database. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        while self._database.get_next_group() is not None:

            group_age = self._database.get_group_index() - 1

            print('\tGroup Age: {0}. '.format(group_age), end='')

            if st_group is not None and group_age < st_group:
                print('Pass.')
                continue

            _, template_data, _ = self._database.get_next_data_from_group()
            template_am = utils.get_attention_map(template_data)

            data_from_group_num = self._database.get_data_from_group_size() - 1
            group_block_attr = np.zeros((data_from_group_num, self._block_num, self._attr_num))

            while self._database.has_next_data_from_group():

                print(' ', str(self._database.get_data_index()))

                data_name, matched_data, age = self._database.get_next_data_from_group()
                matched_am = utils.get_attention_map(matched_data)
                self._divide(0, template_data, template_am, matched_data, matched_am)

                group_block_attr[self._database.get_data_index() - 2] = self._block_attr

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

        zoom_rate = np.zeros(3)
        template_center = np.zeros(3, dtype=int)
        matched_center = np.zeros(3, dtype=int)
        for i in range(3):
            zoom_rate[i], template_center[i], matched_center[i] = utils.get_zoom_parameter(template_am, matched_am, i)

        zoomed_matched_am = ndimage.zoom(matched_am, zoom_rate)
        zoomed_matched_data = ndimage.zoom(matched_data, zoom_rate)
        zoomed_matched_center = (matched_center * zoom_rate).astype(int)
        offset = template_center - zoomed_matched_center

        # utils.show_3Ddata_comp(template_am, matched_am, 'PRE')
        # utils.show_3Ddata_comp(template_am, zoomed_matched_am, 'AFTER')

        zoomed_matched_am_proj = utils.get_template_proj(template_am, zoomed_matched_am, offset)
        zoomed_matched_data_proj = utils.get_template_proj(template_data, zoomed_matched_data, offset)

        am_mean = template_am.mean()
        am_div = utils.get_div(template_am, zoomed_matched_am_proj)
        am_significant = (am_mean + 1e-8) / (am_div + 1e-8)
        intensity_mean = template_data.mean() - zoomed_matched_data_proj.mean()


        self._block_attr[block_id] = np.r_[zoom_rate, offset, intensity_mean, am_mean, am_div, am_significant]

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

        print('Inductive Learning. database. Dataset {0}.'.format(dataset_name))

        self._database.load_database(data_path, dataset_name, mode='training')
        group_block_attr_path = 'experiments/group_block_attr/'

        for group_age in range(self._database.get_group_index(), self._database.get_group_end()):

            print('group age ', group_age)

            group_block_attr = np.load(group_block_attr_path + str(group_age) + '.npy')
            flag = group_block_attr[:,:,8]
            valid_block = []
            for i in range(73, group_block_attr.shape[1]):
            # for i in range(9,73):
                if flag[:,i].mean() > 0.8:
                    print(i)
                    m = []
                    s = []
                    for j in range(6):
                        m.append(group_block_attr[:, i, j].mean())
                        s.append(group_block_attr[:, i, j].std())
                    # print(group_block_attr[:,i,:])
                    valid_block.append(i)
                    print('mean ', m)
                    print('std  ', s)
            print(valid_block)
            pass


    def validate(self):
        self._database.load_database('../../data/', 'IXI-T1', mode='validation')

        pass
