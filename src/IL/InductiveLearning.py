from scipy import ndimage
import numpy as np

from src.IL.Database import Database
import src.utils as utils


class InductiveLearning:

    def __init__(self):
        self._database = Database()

        self._divisor = 2
        self._log_c = np.log(2 ** 3)
        self._block_anchors = np.array([[0, 0.6], [0.4, 1]])
        self._divide_level_limit = 4

        self._zoom_rates = np.arange(0.7, 1.4, 0.1)

    def load_group(self):
        pass

    def train(self, data_path, dataset_name, resample=True):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Train model: Inductive Learning. database. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        while self._database.get_next_group() is not None:

            group_age = self._database.get_group_index()

            print('\tGroup Age: {0}.'.format(group_age))

            _, template_data, _ = self._database.get_next_data_from_group()
            template_am = utils.get_attention_map(template_data)

            # utils.show_3Ddata(template_am)

            while self._database.has_next_data_from_group():

                data_name, matched_data, age = self._database.get_next_data_from_group()
                matched_am = utils.get_attention_map(matched_data)
                self._induce(0, template_data, template_am, matched_data, matched_am)

        print('Done.')

    def _induce(self, block_id, template_data, template_am, matched_data, matched_am):

        divide_level = 0
        if block_id > 0:
            divide_level = np.floor(np.log(block_id) / self._log_c) + 1

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

                    offset = template_center - zoomed_matched_center + 0.5
                    offset = offset.astype(int)

                    div = utils.get_div(template_am, zoomed_matched_am, offset)
                    if best_div > div:
                        best_div = div
                        best_zoom_rate[0] = zoom_rate0
                        best_zoom_rate[1] = zoom_rate1
                        best_zoom_rate[2] = zoom_rate2

        print(best_zoom_rate, best_div)

        zoomed_matched_am = ndimage.zoom(matched_am, [best_zoom_rate[0], best_zoom_rate[1], best_zoom_rate[2]])
        zoomed_matched_data = ndimage.zoom(matched_data, [best_zoom_rate[0], best_zoom_rate[1], best_zoom_rate[2]])

        sub_block_id = block_id * 8 - 1

        for block_anchor0 in self._block_anchors:

            sub_template_anchor0 = block_anchor0 * template_am.shape[0]
            sub_matched_anchor0 = block_anchor0 * matched_am.shape[0]

            for block_anchor1 in self._block_anchors:

                sub_template_anchor1 = block_anchor1 * template_am.shape[1]
                sub_matched_anchor1 = block_anchor1 * matched_am.shape[1]

                for block_anchor2 in self._block_anchors:

                    sub_template_anchor2 = block_anchor2 * template_am.shape[2]
                    sub_matched_anchor2 = block_anchor2 * matched_am.shape[2]

                    sub_template_data = template_data[
                                            sub_template_anchor0[0]:sub_template_anchor0[1],
                                            sub_template_anchor1[0]:sub_template_anchor1[1],
                                            sub_template_anchor2[0]:sub_template_anchor2[1]]

                    sub_template_am = template_am[
                                            sub_template_anchor0[0]:sub_template_anchor0[1],
                                            sub_template_anchor1[0]:sub_template_anchor1[1],
                                            sub_template_anchor2[0]:sub_template_anchor2[1]]
                    
                    sub_matched_data = zoomed_matched_data[
                                            sub_matched_anchor0[0]:sub_matched_anchor0[1],
                                            sub_matched_anchor1[0]:sub_matched_anchor1[1],
                                            sub_matched_anchor2[0]:sub_matched_anchor2[1]]

                    sub_matched_am = zoomed_matched_am[
                                            sub_matched_anchor0[0]:sub_matched_anchor0[1],
                                            sub_matched_anchor1[0]:sub_matched_anchor1[1],
                                            sub_matched_anchor2[0]:sub_matched_anchor2[1]]

                    sub_block_id += 1

                    self._induce(sub_block_id, sub_template_data, sub_template_am, sub_matched_data, sub_matched_am)


        pass

    def validate(self):
        self._database.load_database('../../data/', 'IXI-T1', mode='validation')

        pass
