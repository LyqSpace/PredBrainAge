import os
import numpy as np
from ksvd import ApproximateKSVD
import src.utils as utils


class SparseDict:

    def __init__(self, data_path, expt_path, n_components, mode):

        print('Construct baseline model.')

        np.set_printoptions(precision=3, suppress=True)

        self._expt_path = expt_path
        self._n_components = n_components
        self._mode = mode

        if not os.path.exists(expt_path):
            os.mkdir(expt_path)

        if mode == 'training':
            self._data = np.load(data_path + 'training_data.npy')
            self._ages = np.load(data_path + 'training_ages.npy')
        elif mode == 'validation':
            self._data = np.load(data_path + 'validation_data.npy')
            self._ages = np.load(data_path + 'validation_ages.npy')
        elif mode == 'test':
            self._data = np.load(data_path + 'test_data.npy')
            self._ages = np.load(data_path + 'test_ages.npy')
        else:
            raise Exception('mode must be in [training, validation, test].')

        self._data_num = self._data.shape[0]
        self._data_shape = self._data.shape[1:]
        self._patch_shape = np.array([10, 10, 10])
        self._patch_size = (self._data_shape / self._patch_shape).astype(int)

    def divide(self):

        patches = None

        for i0 in range(self._patch_size[0]-1):
            u0 = self._patch_shape[0] * i0
            v0 = self._patch_shape[0] * (i0 + 1)
            for i1 in range(self._patch_size[0]-1):
                u1 = self._patch_shape[1] * i1
                v1 = self._patch_shape[1] * (i1 + 1)
                for i2 in range(self._patch_size[0]-1):
                    u2 = self._patch_shape[2] * i2
                    v2 = self._patch_shape[2] * (i2 + 1)

                    patch = np.reshape(self._data[:, u0:v0, u1:v1, u2:v2], (self._data_num,-1))
                    if patches is None:
                        patches = patch
                    else:
                        patches = np.r_[patches, patch]

        np.save(self._expt_path + 'patches.npy', patches)
        print('Saved patches.')
        return patches

    def fit(self):

        patches = self.divide()
        aksvd = ApproximateKSVD(n_components=self._n_components, max_iter=30)
        dictionary = aksvd.fit(patches).components_
        np.save(self._expt_path + 'dictionary.npy', dictionary)
        print('Saved dictionary.')

        gamma = aksvd.transform(patches)
        np.save(self._expt_path + 'gamma.npy', gamma)
        print('Saved gamma.')

        pass

    def transform(self):
        pass
