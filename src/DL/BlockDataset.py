import torch
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):

    def __init__(self):
        self._block_data = None
        self._object_ages = None

        self._object_num = 0
        self._object_block_num = 0
        self._total_block_num = 0

    def __len__(self):
        return self._total_block_num

    def __getitem__(self, idx):

        object_id = idx // self._object_block_num
        block_id = idx % self._object_block_num

        sample = {
            'data': torch.from_numpy(self._block_data[object_id][block_id]).unsqueeze(0).float(),
            'age': torch.from_numpy(np.array([self._object_ages[object_id]])).unsqueeze(0).float()
        }

        return sample

    def init_dataset(self, block_data, object_ages):

        self._block_data = block_data
        self._object_ages = object_ages

        self._object_num = block_data.shape[0]
        self._object_block_num = block_data.shape[1]
        self._total_block_num = self._object_num * self._object_block_num


if __name__ == '__main__':

    block_dataset = BlockDataset()
