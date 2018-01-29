import subprocess
import os
import nilearn.image as ni_img
import nibabel.affines as ni_affine
import re
import numpy as np
import src.utils as utils


def get_unused_list():
    f = open('unused_list.txt', 'w')

    data_list = os.listdir('data/IXI-T1-unused')
    for name in data_list:
        print(name, file=f)
    f.close()


def rm_unused_data():
    f = open('../data/IXI-T1-unused_list.txt', 'r')

    for line in f:
        subprocess.call('rm ../data/IXI-T1-raw-niigz/' + line.strip(), shell=True)

    f.close()


def resize_data(st_id = 0, file_name=None):

    data_name_list = os.listdir('../data/IXI-T1-raw-niigz')
    count = 0

    for name in data_name_list:

        count += 1

        if count <= st_id:
            continue

        if file_name is not None and name != file_name:
            continue

        print(count, name, end='')

        re_result = re.findall(r'IXI(\d+)-', name)
        img_id = int(re_result[0])
        img = ni_img.load_img('../data/IXI-T1-raw-niigz/' + name)

        img_affine = ni_img.resample_img(img, target_affine=np.eye(3) * 4)
        print(img_affine.get_fdata().shape)

        # img_data = img_affine.get_data()
        #
        # slice_0 = img_data[50, :, :]
        # slice_1 = img_data[:, 70, :]
        # slice_2 = img_data[:, :, 70]
        # utils.show_slices([slice_0, slice_1, slice_2], 'affine')
        #
        # print(img_affine.shape)
        #
        # continue

        np.save('../data/IXI-T1-small-npy/' + str(img_id), img_affine.get_fdata())


def get_training_name_list():

    all_name_list = os.listdir('../data/IXI-T1')

    file = open('../test_name_list.txt', 'r')
    the_list = []
    for line in file:
        the_list.append(line.strip())
    file.close()

    training_name_list = list(set(all_name_list) - (set(the_list)))

    file = open('../training_name_list.txt', 'w')
    for name in training_name_list:
        print(name, file=file)
    file.close()


def save_list(the_list, file_name):
    file = open(file_name, 'w')
    for name in the_list:
        print(name, file=file)
    file.close()


def read_list(file_name):
    file = open(file_name, 'r')
    the_list = []
    for line in file:
        the_list.append(line.strip())
    file.close()
    return the_list


def delete_data(data_id):
    the_list = read_list('../training_pair_list.txt')
    new_list = []
    count = 0
    for i in range(len(the_list)):
        line = the_list[i]
        line = line.replace('(', '').replace(')', '').replace(' ', '').replace('\'', '')
        tmp = line.split(',')
        if tmp[0] != (str(data_id) + '.npy') and tmp[1] != (str(data_id) + '.npy'):
            new_list.append((tmp[0], tmp[1]))
        else:
            count += 1
    print(count)
    save_list(new_list, '../training_pair_list.txt')

    the_list = read_list('../training_name_list.txt')
    new_list = []
    count = 0
    for i in range(len(the_list)):
        if the_list[i] != (str(data_id) + '.npy'):
            new_list.append(the_list[i])
        else:
            count += 1
    print(count)
    save_list(new_list, '../training_name_list.txt')


def normalize_npy(file_name=None):
    path = '../data/IXI-T1-small-npy/'
    data_name_list = os.listdir(path)
    count = 0
    for name in data_name_list:
        print(count)
        count += 1

        if file_name is not None and file_name != name:
            continue

        data = np.load(path + name)
        new_data = (data - data.mean()) / data.std()
        np.save('../data/IXI-T1/' + name, new_data)

        # slice_0 = data[50, :, :]
        # slice_1 = data[:, 70, :]
        # slice_2 = data[:, :, 70]
        # utils.show_slices([slice_0, slice_1, slice_2])
        #
        # slice_0 = new_data[50, :, :]
        # slice_1 = new_data[:, 70, :]
        # slice_2 = new_data[:, :, 70]
        # utils.show_slices([slice_0, slice_1, slice_2])


def rename():
    path = '../experiments/group_block_attr/'
    for i in range(20, 87):
        x = np.load(path + str(i) + '.npy')
        np.save(path+str(i-1)+'.npy', x)


if __name__ == '__main__':
    # delete_data(533)
    # normalize_npy()

    # resize_data(file_name='IXI533-Guys-1066-T1.nii.gz')
    # normalize_npy(file_name='533.npy')
    # rename()
    pass