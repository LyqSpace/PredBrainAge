import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def show_3Ddata(data, title=None):

    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle(title)

    idx0 = int(data.shape[0] / 2)
    idx1 = int(data.shape[1] / 2)
    idx2 = int(data.shape[2] / 2)
    axes[0].imshow(data[idx0, :, :], cmap='gray', origin='lower')
    axes[1].imshow(data[:, idx1, :], cmap='gray', origin='lower')
    axes[2].imshow(data[:, :, idx2], cmap='gray', origin='lower')

    ax_idx0 = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_idx1 = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_idx2 = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider_idx0 = Slider(ax_idx0, 'Idx0', 0, data.shape[0] - 1, valinit=idx0, valfmt='%d')
    slider_idx1 = Slider(ax_idx1, 'Idx1', 0, data.shape[1] - 1, valinit=idx1, valfmt='%d')
    slider_idx2 = Slider(ax_idx2, 'Idx2', 0, data.shape[2] - 1, valinit=idx2, valfmt='%d')

    def update_slider_idx0(val):
        axes[0].imshow(data[int(val), :, :], cmap='gray', origin='lower')
        pass

    def update_slider_idx1(val):
        axes[1].imshow(data[:, int(val), :], cmap='gray', origin='lower')
        pass

    def update_slider_idx2(val):
        axes[2].imshow(data[:, :, int(val)], cmap='gray', origin='lower')
        pass

    slider_idx0.on_changed(update_slider_idx0)
    slider_idx1.on_changed(update_slider_idx1)
    slider_idx2.on_changed(update_slider_idx2)

    plt.show()


def save_list(the_list, file_name):
    file = open(file_name, 'w')
    for name in the_list:
        print(name, file=file)
    file.close()


def load_list(file_name):
    file = open(file_name, 'r')
    the_list = []
    for line in file:
        the_list.append(line.strip())
    file.close()

    return the_list


def check_outside(point, shape):
    for i in range(3):
        if point[i] < 0 or point[i] >= shape[i]:
            return True
    return False


def get_attention_map(img):

    ways = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                ways.append((i,j,k))

    attention_map = np.zeros(img.shape)

    print(img.shape)

    for i in range(img.shape[0]):
        print(i)
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                cur_point = (i, j, k)
                cur_intensity = img[cur_point]
                sum_attention = 0
                sum_neighbors = 0

                for way in ways:

                    neighbor_point = tuple([cur_point[d] + way[d] for d in range(3)])
                    if check_outside(neighbor_point, img.shape):
                        continue
                    neighbor_intensiry = img[neighbor_point]
                    sum_attention += (cur_intensity - neighbor_intensiry) ** 2
                    sum_neighbors += 1

                attention_map[cur_point] = sum_attention / (sum_neighbors - 1)

    return attention_map


if __name__ == '__main__':
    '''
    slice_0 = epi_img_data[26, :, :]
    slice_1 = epi_img_data[:, 30, :]
    slice_2 = epi_img_data[:, :, 16]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image")
    '''
