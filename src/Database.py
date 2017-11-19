import pandas as pd
import numpy as np
import os
import re
import random
import torch


class Database:

    def __init__(self):

        self._training_pair_index = 0
        self._training_pair_size = 0
        self._training_pair_list = None

        self._test_index = 0
        self._test_data_size = 0
        self._test_name_list = None

        self._training_data_index = 0
        self._training_data_size = 0
        self._training_name_list = None

        self._shape = None
        self._dataset_loaded = False
        self._test_loaded = False
        self._dataset_name = None
        self._dataset_path = None
        self._data_info_df = None

    def load_database(self, data_path, dataset_name, shape=None, test=False, resample=True):

        self._dataset_name = dataset_name
        self._dataset_path = data_path + dataset_name
        self._shape = shape

        if dataset_name == 'IXI-T1':

            xls_file = pd.read_excel(self._dataset_path + '.xls')
            self._data_info_df = xls_file.loc[:, ['IXI_ID', 'AGE']]
            self._data_info_df.dropna(how='any', inplace=True)
            self._data_info_df.index = self._data_info_df['IXI_ID']

            if test is False and resample is True:
                data_name_list = os.listdir(self._dataset_path)

                data_size = len(data_name_list)
                self._training_pair_index = 0
                training_data_size = int(0.9 * data_size)
                training_name_list = random.sample(data_name_list, training_data_size)

                unit_list = []
                for i in range(training_data_size):
                    for j in range(i):
                        unit_list.append((training_name_list[i], training_name_list[j]))

                self._training_pair_list = random.sample(unit_list, len(unit_list))
                self._training_pair_size = len(self._training_pair_list)

                self._test_index = 0
                self._test_name_list = list(set(data_name_list) - (set(training_name_list)))
                self._test_data_size = len(self._test_name_list)

                self.save_list(training_name_list, 'training_name_list.txt')
                self.save_list(self._training_pair_list, 'training_pair_list.txt')
                self.save_list(self._test_name_list, 'test_name_list.txt')

            else:
                if test is False:
                    self._training_pair_index = 0
                    self._training_pair_list = self.read_list('training_pair_list.txt')
                    self._training_pair_size = len(self._training_pair_list)
                    for i in range(self._training_pair_size):
                        line = self._training_pair_list[i]
                        line = line.replace('(', '').replace(')', '').replace(' ', '').replace('\'', '')
                        tmp = line.split(',')
                        self._training_pair_list[i] = (tmp[0], tmp[1])
                    # print('Training name list: ', self._training_pair_list)
                else:
                    self._test_name_list = self.read_list('test_name_list.txt')
                    self._test_index = 0
                    self._test_data_size = len(self._test_name_list)
                    # print('Test name list: ', self._test_name_list)
                
            self._dataset_loaded = True
        else:
            raise Exception(dataset_name + ' dataset is not found.')

    def random_training_data(self, training_data_size):
        training_name_list = self.read_list('training_name_list.txt')
        self._training_name_list = random.sample(training_name_list, training_data_size)
        self._training_data_index = 0
        self._training_data_size = len(self._training_name_list)
        self._test_loaded = True

    def save_list(self, the_list, file_name):
        file = open(file_name, 'w')
        for name in the_list:
            print(name, file=file)
        file.close()

    def read_list(self, file_name):
        file = open(file_name, 'r')
        the_list = []
        for line in file:
            the_list.append(line.strip())
        file.close()

        return the_list

    def get_training_pair_size(self):
        return self._training_pair_size

    def get_test_data_size(self):
        return self._test_data_size

    def get_training_data_index(self):
        return self._training_pair_index

    def get_training_pair_index(self):
        return self._training_pair_index

    def get_test_index(self):
        return self._test_index

    def set_training_data_index(self, index=0):
        self._training_pair_index = index

    def set_training_pair_index(self, index=0):
        self._training_pair_index = index

    def set_test_index(self, index=0):
        self._test_index = index

    def has_training_data_next(self):
        if self._test_loaded is True and self._training_data_index < self._training_data_size:
            return True
        else:
            return False

    def has_training_pair_next(self):
        if self._dataset_loaded is True and self._training_pair_index < self._training_pair_size:
            return True
        else:
            return False

    def has_test_next(self):
        if self._dataset_loaded is True and self._test_index < self._test_data_size:
            return True
        else:
            return False

    def load_data(self, img_name):

        re_result = re.findall(r'(\d+)\.npy', img_name)
        img_id = int(re_result[0])
        img = np.load(self._dataset_path + '/' + img_name)
        img_tensor = torch.from_numpy(img)
        try:
            age = self._data_info_df.loc[img_id, 'AGE']
            age_tensor = torch.from_numpy(np.array([age]))
        except KeyError:
            raise Exception('The image id ' + str(img_id) + ' is not included in the xls file.')

        return img_name, img_tensor, age_tensor

    def load_training_pair_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._training_pair_index == self._training_pair_size:
            raise Exception('This training pair dataset has nothing to be loaded.')

        img_name = self._training_pair_list[self._training_pair_index][0]
        img_name1, img_tensor1, age_tensor1 = self.load_data(img_name)
        img_name = self._training_pair_list[self._training_pair_index][1]
        img_name2, img_tensor2, age_tensor2 = self.load_data(img_name)
        self._training_pair_index += 1

        return img_name1, img_tensor1, img_name2, img_tensor2, age_tensor1 - age_tensor2
    
    def load_test_data_next(self):

        if self._dataset_loaded is False:
            raise Exception('Load the dataset first.')
        if self._test_index == self._test_data_size:
            raise Exception('This test dataset has nothing to be loaded.')

        img_name = self._test_name_list[self._test_index]
        self._test_index += 1

        return self.load_data(img_name)

    def load_training_data_next(self):

        if self._dataset_loaded is False or self._test_loaded is False:
            raise Exception('Load the dataset first.')
        if self._training_data_index == self._training_data_size:
            raise Exception('This training dataset has nothing to be loaded.')

        img_name = self._training_name_list[self._training_data_index]
        self._training_data_index += 1

        return self.load_data(img_name)


if __name__ == '__main__':
    database = Database()
    database.load_database('../data/', 'IXI-T1', (128, 128, 75), resample=True)
    img_name1, _, img_name2, _, age_diff = database.load_training_pair_next()
    print(img_name1, img_name2, age_diff)
    img_name1, _, img_name2, _, age_diff = database.load_training_pair_next()
    print(img_name1, img_name2, age_diff)
    img_name1, _, img_name2, _, age_diff = database.load_training_pair_next()
    print(img_name1, img_name2, age_diff)
    img_name1, _, img_name2, _, age_diff = database.load_training_pair_next()
    print(img_name1, img_name2, age_diff)
