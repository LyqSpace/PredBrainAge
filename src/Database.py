import nilearn.image as ni_img
import pandas as pd
import numpy as np
import os
import re
import random
import torch


class Database:

    def __init__(self):

        self._training_index = 0
        self._training_data_size = 0
        self._training_name_list = None

        self._test_index = 0
        self._test_data_size = 0
        self._test_name_list = None

        self._dataset_loaded = False
        self._dataset_name = None
        self._dataset_path = None
        self._data_info_df = None

    def load_database(self, dataset_name):

        self._dataset_name = dataset_name
        self._dataset_path = './data/' + dataset_name

        if dataset_name == 'IXI-T1':
            data_name_list = os.listdir(self._dataset_path)

            xls_file = pd.read_excel(self._dataset_path + '.xls')
            self._data_info_df = xls_file.loc[:, ['IXI_ID', 'AGE']]
            self._data_info_df.dropna(how='any', inplace=True)
            self._data_info_df.index = self._data_info_df['IXI_ID']

            data_size = len(data_name_list)
            self._training_index = 0
            self._training_data_size = int(0.9 * data_size)
            self._training_name_list = random.sample(data_name_list, self._training_data_size)

            self._test_index = 0
            self._test_name_list = list(set(data_name_list) - (set(self._training_name_list)))
            self._test_data_size = len(self._test_name_list)
            self._dataset_loaded = True
        else:
            raise Exception(dataset_name + ' dataset is not found.')

    def get_training_index(self):
        return self._training_index

    def get_test_index(self):
        return self._test_index

    def reset_training_index(self):
        self._training_index = 0

    def reset_test_index(self):
        self._test_index = 0

    def has_training_next(self):
        if self._dataset_loaded is True and self._training_index < self._training_data_size:
            return True
        else:
            return False

    def has_test_next(self):
        if self._dataset_loaded is True and self._test_index < self._test_data_size:
            return True
        else:
            return False

    def load_training_data_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._training_index == self._training_data_size:
            raise Exception('This dataset has nothing to be loaded.')

        img_name = self._training_name_list[self._training_index]
        self._training_index += 1

        re_result = re.findall(r'IXI(\d+)-', img_name)
        img_id = int(re_result[0])
        img = ni_img.load_img(self._dataset_path + '/' + img_name)

        img_tensor = torch.from_numpy(img.get_data())

        try:
            age = self._data_info_df.loc[img_id, 'AGE']
            age_tensor = torch.from_numpy(np.array([age]))
        except KeyError:
            raise Exception('The image id ' + str(img_id) + ' is not included in the xls file.')

        return img_name, img_tensor, age_tensor
    
    def load_test_data_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._test_index == self._test_data_size:
            raise Exception('This dataset has nothing to be loaded.')

        img_name = self._test_name_list[self._test_index]
        self._test_index += 1

        re_result = re.findall(r'IXI(\d+)-', img_name)
        img_id = int(re_result[0])
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
    database.load_training_data_next()
