import nilearn.image as ni_img
import pandas as pd
import numpy as np
import os
import re
import torch


class Database:

    def __init__(self):
        self._index = 0
        self._data_size = 0
        self._dataset_loaded = False
        self._dataset_name = None
        self._dataset_path = None
        self._data_name_list = None
        self._data_info_df = None

    def load_database(self, dataset_name):
        self._dataset_name = dataset_name
        self._dataset_path = './data/' + dataset_name
        if dataset_name == 'IXI-T1':
            self._index = 0
            self._data_name_list = os.listdir(self._dataset_path)
            self._data_size = len(self._data_name_list)

            xls_file = pd.read_excel(self._dataset_path + '.xls')
            self._data_info_df = xls_file.loc[:, ['IXI_ID', 'AGE']]
            self._data_info_df.dropna(how='any', inplace=True)
            self._data_info_df.index = self._data_info_df['IXI_ID']

            self._dataset_loaded = True
        else:
            raise Exception(dataset_name + ' dataset is not found.')

    def get_index(self):
        return self._index

    def reset_index(self):
        self._index = 0

    def has_next(self):
        if self._dataset_loaded is True and self._index < self._data_size:
            return True
        else:
            return False

    def load_data_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._index == self._data_size:
            raise Exception('This dataset has nothing to be loaded.')

        img_name = self._data_name_list[self._index]
        re_result = re.findall(r'IXI(\d+)-', img_name)
        img_id = int(re_result[0])
        self._index += 1
        img = ni_img.load_img(self._dataset_path + '/' + img_name)
        img_tensor = torch.from_numpy(img.get_data())

        try:
            age = self._data_info_df.loc[img_id, 'AGE']
            age_tensor = torch.from_numpy(np.array([age]))
        except KeyError:
            raise Exception('The image id ' + str(img_id) + ' is not included in the xls file.')

        return img_name, img_tensor, age_tensor

if __name__ == '__main__':
    database = Database()
    database.load_database('IXI-T1')
    database.load_data_next()
