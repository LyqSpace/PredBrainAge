from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

from src.IL.Database import Database
import src.utils as utils


class InductiveLearning:

    def __init__(self):
        self._database = Database()

        self._divisor = 2
        self._divide_level_limit = 4
        self._block_anchors = np.array([[0, 0.6], [0.4, 1]])
        self._child_num = self._divisor ** 3
        self._block_num = 0
        for i in range(self._divide_level_limit):
            self._block_num += self._child_num ** i

        self._block_attr = np.zeros((self._block_num, 8))
        self._block_significant = np.zeros(self._block_num, dtype=bool)

        self._zoom_rates = np.arange(0.7, 1.4, 0.1)

    def load_group(self):
        pass

    def train(self, data_path, dataset_name, resample=False):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Train model: Inductive Learning. database. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        while self._database.get_next_group() is not None:

            group_age = self._database.get_group_index()

            print('\tGroup Age: {0}.'.format(group_age))

            _, template_data, _ = self._database.get_next_data_from_group()
            template_am = utils.get_attention_map(template_data)

            data_from_group_num = self._database.get_data_from_group_size() - 1
            group_block_attr = np.array([data_from_group_num, self._block_num, 8])

            while self._database.has_next_data_from_group():

                data_name, matched_data, age = self._database.get_next_data_from_group()
                matched_am = utils.get_attention_map(matched_data)
                self._induce(0, template_data, template_am, matched_data, matched_am)

                group_block_attr[self._database.get_data_index() - 1] = self._block_attr

            np.save('experiments/group_block_attr/' + group_age + '.npy', group_block_attr)

        print('Done.')

    def _induce(self, block_id, template_data, template_am, matched_data, matched_am):

        divide_level = 0
        father_id = block_id
        while father_id > 0:
            father_id = int((father_id - 1) / self._child_num)
            divide_level += 1

        if divide_level == self._divide_level_limit:
            return

        template_center = np.array(ndimage.measurements.center_of_mass(template_am))
        matched_center = np.array(ndimage.measurements.center_of_mass(matched_am))

        best_div = np.inf
        best_zoom_rate = np.ones(3)

        for zoom_rate0 in self._zoom_rates:
            for zoom_rate1 in self._zoom_rates:
                for zoom_rate2 in self._zoom_rates:

                    zoomed_matched_am = ndimage.zoom(matched_am, [zoom_rate0, zoom_rate1, zoom_rate2])
                    zoomed_matched_center = np.array([matched_center[0] * zoom_rate0,
                                                      matched_center[1] * zoom_rate1,
                                                      matched_center[2] * zoom_rate2])

                    offset = (template_center - zoomed_matched_center + 0.5).astype(int)

                    inter_am = utils.get_inter_data(template_am, zoomed_matched_am, offset)
                    div = utils.get_div(template_am, inter_am)

                    if best_div > div:
                        best_div = div
                        best_zoom_rate[0] = zoom_rate0
                        best_zoom_rate[1] = zoom_rate1
                        best_zoom_rate[2] = zoom_rate2

        zoomed_matched_am = ndimage.zoom(matched_am, [best_zoom_rate[0], best_zoom_rate[1], best_zoom_rate[2]])
        zoomed_matched_data = ndimage.zoom(matched_data, [best_zoom_rate[0], best_zoom_rate[1], best_zoom_rate[2]])

        best_matched_center = matched_center * best_zoom_rate
        best_offset = (template_center - best_matched_center + 0.5).astype(int)

        inter_am = utils.get_inter_data(template_am, zoomed_matched_am, best_offset)
        inter_data = utils.get_inter_data(template_data, zoomed_matched_data, best_offset)

        intensity_div = template_data.mean() - zoomed_matched_data.mean()

        self._block_attr[block_id] = np.r_[best_offset, best_zoom_rate, intensity_div, best_div]

        print(block_id, self._block_attr[block_id], template_am.shape, inter_am.shape)

        if divide_level > 1:

            # utils.show_3Ddata(template_am, 'Template')
            # utils.show_3Ddata(zoomed_matched_am, 'Zoomed Matched')
            utils.show_3Ddata_comp(template_am, inter_am, 'Comp')

        sub_block_id = block_id * 8

        for block_anchor0 in self._block_anchors:

            sub_anchor0 = (block_anchor0 * template_am.shape[0]).astype(int)

            for block_anchor1 in self._block_anchors:

                sub_anchor1 = (block_anchor1 * template_am.shape[1]).astype(int)

                for block_anchor2 in self._block_anchors:

                    sub_anchor2 = (block_anchor2 * template_am.shape[2]).astype(int)

                    sub_template_data = template_data[
                                            sub_anchor0[0]:sub_anchor0[1],
                                            sub_anchor1[0]:sub_anchor1[1],
                                            sub_anchor2[0]:sub_anchor2[1]]
                    sub_template_am = template_am[
                                            sub_anchor0[0]:sub_anchor0[1],
                                            sub_anchor1[0]:sub_anchor1[1],
                                            sub_anchor2[0]:sub_anchor2[1]]
                    sub_matched_data = zoomed_matched_data[
                                            sub_anchor0[0]:sub_anchor0[1],
                                            sub_anchor1[0]:sub_anchor1[1],
                                            sub_anchor2[0]:sub_anchor2[1]]
                    sub_matched_am = zoomed_matched_am[
                                            sub_anchor0[0]:sub_anchor0[1],
                                            sub_anchor1[0]:sub_anchor1[1],
                                            sub_anchor2[0]:sub_anchor2[1]]

                    sub_block_id += 1

                    self._induce(sub_block_id, sub_template_data, sub_template_am, sub_matched_data, sub_matched_am)


        pass

    def validate(self):
        self._database.load_database('../../data/', 'IXI-T1', mode='validation')

        pass
