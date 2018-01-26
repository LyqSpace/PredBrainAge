import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import ndimage
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


def threshold(a, threshmin=None, threshmax=None, newval=0):
    """
    Clip array to a given value.
    Similar to numpy.clip(), except that values less than `threshmin` or
    greater than `threshmax` are replaced by `newval`, instead of by
    `threshmin` and `threshmax` respectively.
    Parameters
    ----------
    a : ndarray
        Input data
    threshmin : {None, float}, optional
        Lower threshold. If None, set to the minimum value.
    threshmax : {None, float}, optional
        Upper threshold. If None, set to the maximum value.
    newval : {0, float}, optional
        Value outside the thresholds.
    Returns
    -------
    threshold : ndarray
        Returns `a`, with values less then `threshmin` and values greater
        `threshmax` replaced with `newval`.
    """
    a_mask = np.ma.array(a, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a_mask < threshmin).filled(False)
    if threshmax is not None:
        mask |= (a_mask > threshmax).filled(False)
    a[mask] = newval
    return a


def get_attention_map(img):

    # ways = []
    # for i in range(-1, 2):
    #     for j in range(-1, 2):
    #         for k in range(-1, 2):
    #             ways.append((i,j,k))

    # for i in range(img.shape[0]):
    #     print(i)
    #     for j in range(img.shape[1]):
    #         for k in range(img.shape[2]):
    #
    #             cur_point = (i, j, k)
    #             cur_intensity = img[cur_point]
    #             sum_attention = 0
    #             sum_neighbors = 0
    #
    #             for way in ways:
    #
    #                 neighbor_point = tuple([cur_point[d] + way[d] for d in range(3)])
    #                 if check_outside(neighbor_point, img.shape):
    #                     continue
    #                 neighbor_intensiry = img[neighbor_point]
    #                 sum_attention += (cur_intensity - neighbor_intensiry) ** 2
    #                 sum_neighbors += 1
    #
    #             attention_map[cur_point] = sum_attention / (sum_neighbors - 1)

    kernel = np.array([[[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]],

                       [[-2, 0, 2],
                        [-4, 0, 4],
                        [-2, 0, 2]],

                       [[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]])
    sobel_map0 = ndimage.convolve(img, kernel)

    kernel = np.array([[[-1, -2, -1],
                        [-2, -4, -2],
                        [-1, -2, -1]],

                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],

                       [[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]])
    sobel_map1 = ndimage.convolve(img, kernel)

    kernel = np.array([[[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]],

                       [[-2, -4, -2],
                        [0, 0, 0],
                        [2, 4, 2]],

                       [[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]])
    sobel_map2 = ndimage.convolve(img, kernel)

    attention_map = (sobel_map0 ** 2 + sobel_map1 ** 2 + sobel_map2 ** 2) ** 0.5
    attention_map = threshold(attention_map, 1e-6, newval=0)

    return attention_map


def get_div(template_am, matched_am, offset):

    template_anchor_st = np.array([0, 0, 0], dtype=int)
    template_anchor_ed = np.array(template_am.shape, dtype=int)
    matched_anchor_st = np.array([0, 0, 0], dtype=int) + offset
    matched_anchor_ed = np.array(matched_am.shape, dtype=int) + offset

    inter_anchor_st = np.zeros(3, dtype=int)
    inter_anchor_ed = np.zeros(3, dtype=int)
    cube_flag = True
    for i in range(3):

        inter_anchor_st[i] = max(template_anchor_st[i], matched_anchor_st[i])
        inter_anchor_ed[i] = min(template_anchor_ed[i], matched_anchor_ed[i])

        if inter_anchor_st[i] >= inter_anchor_ed[i]:
            cube_flag = False

    if cube_flag is False:
        return -1

    ROI_anchor_st = inter_anchor_st - offset
    ROI_anchor_ed = inter_anchor_ed - offset

    matched_am_inter = np.zeros(template_am.shape)
    matched_am_inter[
        inter_anchor_st[0]:inter_anchor_ed[0],
        inter_anchor_st[1]:inter_anchor_ed[1],
        inter_anchor_st[2]:inter_anchor_ed[2]] = matched_am[
                                                    ROI_anchor_st[0]:ROI_anchor_ed[0],
                                                    ROI_anchor_st[1]:ROI_anchor_ed[1],
                                                    ROI_anchor_st[2]:ROI_anchor_ed[2]]

    div = (abs(template_am - matched_am_inter)).mean()

    return div


if __name__ == '__main__':
    '''
    slice_0 = epi_img_data[26, :, :]
    slice_1 = epi_img_data[:, 30, :]
    slice_2 = epi_img_data[:, :, 16]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image")
    '''
