import nilearn.image as ni_img
from nibabel.affines import apply_affine
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

        self._shape = None
        self._dataset_loaded = False
        self._dataset_name = None
        self._dataset_path = None
        self._data_info_df = None

    def load_database(self, dataset_name, shape=None, resample=True):

        self._dataset_name = dataset_name
        self._dataset_path = './data/' + dataset_name
        self._shape = shape

        if dataset_name == 'IXI-T1':

            xls_file = pd.read_excel(self._dataset_path + '.xls')
            self._data_info_df = xls_file.loc[:, ['IXI_ID', 'AGE']]
            self._data_info_df.dropna(how='any', inplace=True)
            self._data_info_df.index = self._data_info_df['IXI_ID']

            if resample is True:
                data_name_list = os.listdir(self._dataset_path)

                data_size = len(data_name_list)
                self._training_index = 0
                self._training_data_size = int(0.9 * data_size)
                self._training_name_list = random.sample(data_name_list, self._training_data_size)

                self._test_index = 0
                self._test_name_list = list(set(data_name_list) - (set(self._training_name_list)))
                self._test_data_size = len(self._test_name_list)

                self.save_name_list()
            else:
                self.read_name_list()
                # print('Training name list: ', self._training_name_list)
                # print('Test name list: ', self._test_name_list)

                self._training_index = 0
                self._training_data_size = len(self._training_name_list)

                self._test_index = 0
                self._test_data_size = len(self._test_name_list)
                
            self._dataset_loaded = True
        else:
            raise Exception(dataset_name + ' dataset is not found.')

    def save_name_list(self):
        file = open('training_name_list.txt', 'w')
        for name in self._training_name_list:
            print(name, file=file)
        file.close()

        file = open('test_name_list.txt', 'w')
        for name in self._test_name_list:
            print(name, file=file)
        file.close()

    def read_name_list(self):
        file = open('training_name_list.txt', 'r')
        self._training_name_list = []
        for line in file:
            self._training_name_list.append(line.strip())
        file.close()

        file = open('test_name_list.txt', 'r')
        self._test_name_list = []
        for line in file:
            self._test_name_list.append(line.strip())
        file.close()

    def get_training_data_size(self):
        return self._training_data_size

    def get_test_data_size(self):
        return self._test_data_size

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

    def resize_img(self, img):
        old_shape = img.shape
        resize_rate = []
        for i in range(3):
            resize_rate.append(self._shape[i] * 1.0 / old_shape[i])

        new_img = np.zeros(shape=self._shape, dtype='int16')
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                for z in range(self._shape[2]):
                    old_x = int(x / resize_rate[0])
                    old_y = int(y / resize_rate[1])
                    old_z = int(z / resize_rate[2])
                    # data = 1.0/8 * (img[old_x, old_y, old_z] + img[old_x, old_y, old_z+1] +
                    #                 img[old_x, old_y+1, old_z] + img[old_x, old_y+1, old_z+1] +
                    #                 img[old_x+1, old_y, old_z] + img[old_x+1, old_y, old_z+1] +
                    #                 img[old_x+1, old_y+1, old_z] + img[old_x+1, old_y+1, old_z+1])
                    data = img[old_x, old_y, old_z]
                    new_img[x, y, z] = data
        # print(new_img.mean())
        # print(new_img.max())
        # print(new_img.min())
        # new_img = new_img - new_img.mean()
        return new_img

    def load_data(self, img_name):

        re_result = re.findall(r'IXI(\d+)-', img_name)
        img_id = int(re_result[0])
        img = ni_img.load_img(self._dataset_path + '/' + img_name)
        img_data = self.resize_img(img.get_data())
        img_tensor = torch.from_numpy(img_data)
        try:
            age = self._data_info_df.loc[img_id, 'AGE']
            age_tensor = torch.from_numpy(np.array([age]))
        except KeyError:
            raise Exception('The image id ' + str(img_id) + ' is not included in the xls file.')

        return img_name, img_tensor, age_tensor

    def load_training_data_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._training_index == self._training_data_size:
            raise Exception('This dataset has nothing to be loaded.')

        img_name = self._training_name_list[self._training_index]
        self._training_index += 1

        return self.load_data(img_name)
    
    def load_test_data_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._test_index == self._test_data_size:
            raise Exception('This dataset has nothing to be loaded.')

        img_name = self._test_name_list[self._test_index]
        self._test_index += 1

        return self.load_data(img_name)

if __name__ == '__main__':
    database = Database()
    database.load_database('IXI-T1', (128, 128, 75))
    database.load_training_data_next()
    database.load_training_data_next()
