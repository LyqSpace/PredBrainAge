import subprocess
import os
import nilearn.image as ni_img
import re
import numpy as np


def get_unused_list():
    f = open('unused_list.txt', 'w')

    data_list = os.listdir('data/IXI-T1-unused')
    for name in data_list:
        print(name, file=f)
    f.close()


def rm_unused_data():
    f = open('../data/IXI-T1-unused_list.txt', 'r')

    for line in f:
        subprocess.call('rm ../data/IXI-T1-raw/' + line.strip(), shell=True)

    f.close()


def resize_data():

    data_name_list = os.listdir('../data/IXI-T1-raw')
    for name in data_name_list:
        re_result = re.findall(r'IXI(\d+)-', name)
        img_id = int(re_result[0])
        img = ni_img.load_img('../data/IXI-T1-raw/' + name).get_data()

        new_shape = (256, 256, 140)

        if img.shape == new_shape:
            np.save('../data/IXI-T1-new/' + str(img_id), img)
            continue

        resize_rate = [new_shape[0]/img.shape[0], new_shape[1]/img.shape[1], new_shape[2]/img.shape[2]]
        print(name, img.shape)
        new_img = np.zeros(shape=new_shape, dtype='int16')
        for x in range(new_shape[0]):
            for y in range(new_shape[1]):
                for z in range(new_shape[2]):
                    # old_x = int(x / resize_rate[0])
                    # old_y = int(y / resize_rate[1])
                    old_z = int(z / resize_rate[2])
                    u = z / resize_rate[2] - old_z
                    # data = 1.0/8 * (float(img[old_x, old_y, old_z]) + float(img[old_x, old_y, old_z+1]) +
                    #                 float(img[old_x, old_y+1, old_z]) + float(img[old_x, old_y+1, old_z+1]) +
                    #                 float(img[old_x+1, old_y, old_z]) + float(img[old_x+1, old_y, old_z+1]) +
                    #                 float(img[old_x+1, old_y+1, old_z]) + float(img[old_x+1, old_y+1, old_z+1]))
                    # data2 = img[x, y, old_z]
                    data3 = (1-u) * img[x, y, old_z] + u *  img[x, y, old_z+1]
                    # if (data != 0) :
                    #     print(data, data0)
                    new_img[x, y, z] = data3
        # print(new_img.mean())
        # print(new_img.max())
        # print(new_img.min())
        # print(img.mean())
        # print(img.max())
        # print(img.min())
        # new_img = new_img - new_img.mean()

        np.save('../data/IXI-T1-new/' + str(img_id), new_img)


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

if __name__ == '__main__':
    # delete_data(533)
    resize_data()