import torch
import numpy as np
from torch.utils.data import Dataset


class BatchSet(Dataset):

    def __init__(self, data, ages):
        self._data = data
        self._ages = ages
        self._num = data.shape[0]

    def __len__(self):
        return self._num

    def __getitem__(self, id):

        sample = {
            'data': torch.from_numpy(self._data[id]),
            'age': torch.from_numpy(np.array([self._ages[id]]))
        }

        return sample


if __name__ == '__main__':

    pass
