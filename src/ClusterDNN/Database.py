import pandas as pd
import numpy as np


class Database:

    def __init__(self):

        self._mode = None
        self._dataset_name = None
        self._data_path = None
        self._dataset_loaded = False

        self._dataset_df = None
        self._data_index = 0

    def load_database(self, data_path, dataset_name, mode='training', resample=False):

        print('Load database. Dataset: {0}. Mode: {1}. Resample: {2}. '.format(
            dataset_name, mode, resample), end='')

        self._mode = mode
        self._dataset_name = dataset_name
        self._data_path = data_path

        if dataset_name == 'IXI-T1':

            if mode not in ['training', 'validation', 'test']:
                raise Exception('Mode must be one of \'training\', \'validation\' or \'test\'')

            if mode != 'training' and resample:
                raise Exception('Resample must be in training mode.')

            if mode == 'training' and resample:
                xls_file = pd.read_excel(data_path + dataset_name + '.xls')
                universe_df = xls_file.loc[:, ['IXI_ID', 'AGE']]
                universe_df.dropna(how='any', inplace=True)
                universe_df.index = universe_df['IXI_ID']
                universe_df.drop('IXI_ID', axis=1, inplace=True)
                universe_df.sort_values(by=['AGE'], inplace=True)

                training_df = universe_df.sample(frac=0.7)
                rest_df = universe_df.loc[universe_df.index.difference(training_df.index)]
                validation_df = rest_df.sample(frac=0.5)
                test_df = rest_df.loc[rest_df.index.difference(validation_df.index)]

                training_df.to_csv(data_path + 'training_df.csv')
                validation_df.to_csv(data_path + 'validation_df.csv')
                test_df.to_csv(data_path + 'test_df.csv')

            if mode == 'training':
                self._dataset_df = pd.read_csv(data_path + 'training_df.csv')

            if mode == 'validation':
                self._dataset_df = pd.read_csv(data_path + 'validation_df.csv')

            if mode == 'test':
                self._dataset_df = pd.read_csv(data_path + 'test_df.csv')

            self._data_index = 0
            self._dataset_loaded = True

            print('Done.')

        else:
            raise Exception(dataset_name + ' dataset is not found.')

    def get_data_size(self):
        if self._dataset_loaded is False:
            raise Exception('Dataset must be loaded first.')
        return self._dataset_df.shape[0]

    def get_data_index(self):
        if self._dataset_loaded is False:
            raise Exception('Dataset must be loaded first.')
        return self._data_index

    def set_data_index(self, id):
        if self._dataset_loaded is False:
            raise Exception('Dataset must be loaded first.')
        self._data_index = id

    def has_next_data(self):
        if self._data_index < self._dataset_df.shape[0]:
            return True
        else:
            return False

    def get_next_data(self, required_data=True):

        if self._dataset_loaded is False:
            raise Exception('Dataset must be loaded first.')

        if self.has_next_data() is False:
            return None

        row_series = self._dataset_df.iloc[self._data_index]
        data_id = str(int(row_series['IXI_ID']))
        data = None
        if required_data:
            data = np.load(self._data_path + self._dataset_name + '/' + data_id + '.npy')

        self._data_index += 1

        return data_id, data, row_series['AGE']

if __name__ == '__main__':

    database = Database()
    database.load_database('../../data/', 'IXI-T1', mode='training', resample=False)
