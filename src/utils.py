import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import ndimage
from scipy import optimize
from skimage import feature
from skimage import measure
from skimage import transform
import numpy as np


def show_3Ddata(data, title=None):

    max_value = data.max()
    min_value = data.min()

    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle(title)

    idx0 = int(data.shape[0] / 2)
    idx1 = int(data.shape[1] / 2)
    idx2 = int(data.shape[2] / 2)
    axes[0].imshow(data[idx0, :, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)
    axes[1].imshow(data[:, idx1, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)
    axes[2].imshow(data[:, :, idx2], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    ax_idx0 = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_idx1 = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_idx2 = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider_idx0 = Slider(ax_idx0, 'Idx0', 0, data.shape[0] - 1, valinit=idx0, valfmt='%d')
    slider_idx1 = Slider(ax_idx1, 'Idx1', 0, data.shape[1] - 1, valinit=idx1, valfmt='%d')
    slider_idx2 = Slider(ax_idx2, 'Idx2', 0, data.shape[2] - 1, valinit=idx2, valfmt='%d')

    def update_slider_idx0(val):
        axes[0].imshow(data[int(val), :, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    def update_slider_idx1(val):
        axes[1].imshow(data[:, int(val), :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    def update_slider_idx2(val):
        axes[2].imshow(data[:, :, int(val)], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    slider_idx0.on_changed(update_slider_idx0)
    slider_idx1.on_changed(update_slider_idx1)
    slider_idx2.on_changed(update_slider_idx2)

    plt.show()
    plt.pause(30)
    plt.close()


def show_3Ddata_comp(data0, data1, title=None):

    max_value = max(data0.max(), data1.max())
    min_value = min(data0.min(), data1.min())
    print('Max Value: ', max_value, 'Min Value:', min_value)

    fig, axes_data0 = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle(title)

    idx0 = int(data0.shape[0] / 2)
    idx1 = int(data0.shape[1] / 2)
    idx2 = int(data0.shape[2] / 2)
    axes_data0[0].imshow(data0[idx0, :, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)
    axes_data0[1].imshow(data0[:, idx1, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)
    axes_data0[2].imshow(data0[:, :, idx2], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    ax_idx0 = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_idx1 = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_idx2 = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider_idx0_data0 = Slider(ax_idx0, 'Idx0', 0, data0.shape[0] - 1, valinit=idx0, valfmt='%d')
    slider_idx1_data0 = Slider(ax_idx1, 'Idx1', 0, data0.shape[1] - 1, valinit=idx1, valfmt='%d')
    slider_idx2_data0 = Slider(ax_idx2, 'Idx2', 0, data0.shape[2] - 1, valinit=idx2, valfmt='%d')

    def update_slider_idx0_data0(val):
        axes_data0[0].imshow(data0[int(val), :, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    def update_slider_idx1_data0(val):
        axes_data0[1].imshow(data0[:, int(val), :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    def update_slider_idx2_data0(val):
        axes_data0[2].imshow(data0[:, :, int(val)], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    slider_idx0_data0.on_changed(update_slider_idx0_data0)
    slider_idx1_data0.on_changed(update_slider_idx1_data0)
    slider_idx2_data0.on_changed(update_slider_idx2_data0)

    fig, axes_data1 = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle(title)

    idx0 = int(data1.shape[0] / 2)
    idx1 = int(data1.shape[1] / 2)
    idx2 = int(data1.shape[2] / 2)
    axes_data1[0].imshow(data1[idx0, :, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)
    axes_data1[1].imshow(data1[:, idx1, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)
    axes_data1[2].imshow(data1[:, :, idx2], cmap='gray',origin='lower', vmax=max_value, vmin=min_value)

    ax_idx0 = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_idx1 = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_idx2 = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider_idx0_data1 = Slider(ax_idx0, 'Idx0', 0, data1.shape[0] - 1, valinit=idx0, valfmt='%d')
    slider_idx1_data1 = Slider(ax_idx1, 'Idx1', 0, data1.shape[1] - 1, valinit=idx1, valfmt='%d')
    slider_idx2_data1 = Slider(ax_idx2, 'Idx2', 0, data1.shape[2] - 1, valinit=idx2, valfmt='%d')

    def update_slider_idx0_data1(val):
        axes_data1[0].imshow(data1[int(val), :, :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    def update_slider_idx1_data1(val):
        axes_data1[1].imshow(data1[:, int(val), :], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    def update_slider_idx2_data1(val):
        axes_data1[2].imshow(data1[:, :, int(val)], cmap='gray', origin='lower', vmax=max_value, vmin=min_value)

    slider_idx0_data1.on_changed(update_slider_idx0_data1)
    slider_idx1_data1.on_changed(update_slider_idx1_data1)
    slider_idx2_data1.on_changed(update_slider_idx2_data1)

    plt.show()
    plt.pause(30)
    plt.close()


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


def get_sobel_axis0(img):
    kernel = np.array([[[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]],

                       [[-2, -4, -2],
                        [0, 0, 0],
                        [2, 4, 2]],

                       [[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]])
    sobel_map = ndimage.convolve(img, kernel) / 32
    return sobel_map


def get_sobel_axis1(img):
    kernel = np.array([[[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]],

                       [[-2, 0, 2],
                        [-4, 0, 4],
                        [-2, 0, 2]],

                       [[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]])
    sobel_map = ndimage.convolve(img, kernel) / 32
    return sobel_map


def get_sobel_axis2(img):
    kernel = np.array([[[-1, -2, -1],
                        [-2, -4, -2],
                        [-1, -2, -1]],

                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],

                       [[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]])
    sobel_map = ndimage.convolve(img, kernel) / 32
    return sobel_map


def get_attention_map(img):

    sobel_map0 = get_sobel_axis0(img)
    sobel_map1 = get_sobel_axis1(img)
    sobel_map2 = get_sobel_axis2(img)

    attention_map = (sobel_map0 ** 2 + sobel_map1 ** 2 + sobel_map2 ** 2) ** 0.5

    return attention_map


def get_inter_data(data0, data1, offset=np.zeros(3)):

    data0_anchor_st = np.array([0, 0, 0], dtype=int)
    data0_anchor_ed = np.array(data0.shape, dtype=int)
    data1_anchor_st = np.array([0, 0, 0], dtype=int) + offset
    data1_anchor_ed = np.array(data1.shape, dtype=int) + offset

    inter_anchor_st = np.maximum(data0_anchor_st, data1_anchor_st)
    inter_anchor_ed = np.minimum(data0_anchor_ed, data1_anchor_ed)

    ROI_anchor_st = inter_anchor_st - offset
    ROI_anchor_ed = inter_anchor_ed - offset

    data1_inter = np.zeros(data0.shape)
    data1_inter[
        inter_anchor_st[0]:inter_anchor_ed[0],
        inter_anchor_st[1]:inter_anchor_ed[1],
        inter_anchor_st[2]:inter_anchor_ed[2]] = data1[
                                                     ROI_anchor_st[0]:ROI_anchor_ed[0],
                                                     ROI_anchor_st[1]:ROI_anchor_ed[1],
                                                     ROI_anchor_st[2]:ROI_anchor_ed[2]]

    return data1_inter


def get_union_data(data0, data1):

    data0_anchor = np.array(data0.shape, dtype=int)
    data1_anchor = np.array(data1.shape, dtype=int)

    union_anchor = np.maximum(data0_anchor, data1_anchor)
    data0_union = np.zeros(union_anchor)
    data1_union = np.zeros(union_anchor)

    data0_union[:data0.shape[0], :data0.shape[1], :data0.shape[2]] = data0[:, :, :]
    data1_union[:data1.shape[0], :data1.shape[1], :data1.shape[2]] = data1[:, :, :]

    return data0_union, data1_union


def get_harris_map(img_data, k=0.05, sigma=1):

    sobel_map0 = get_sobel_axis0(img_data)
    sobel_map1 = get_sobel_axis1(img_data)
    sobel_map2 = get_sobel_axis2(img_data)

    sobel_map0_map0 = ndimage.gaussian_filter(sobel_map0 * sobel_map0, sigma=sigma)
    sobel_map0_map1 = ndimage.gaussian_filter(sobel_map0 * sobel_map1, sigma=sigma)
    sobel_map0_map2 = ndimage.gaussian_filter(sobel_map0 * sobel_map2, sigma=sigma)

    sobel_map1_map0 = ndimage.gaussian_filter(sobel_map1 * sobel_map0, sigma=sigma)
    sobel_map1_map1 = ndimage.gaussian_filter(sobel_map1 * sobel_map1, sigma=sigma)
    sobel_map1_map2 = ndimage.gaussian_filter(sobel_map1 * sobel_map2, sigma=sigma)

    sobel_map2_map0 = ndimage.gaussian_filter(sobel_map2 * sobel_map0, sigma=1)
    sobel_map2_map1 = ndimage.gaussian_filter(sobel_map2 * sobel_map1, sigma=1)
    sobel_map2_map2 = ndimage.gaussian_filter(sobel_map2 * sobel_map2, sigma=1)

    M = np.array([[sobel_map0_map0, sobel_map0_map1, sobel_map0_map2],
                  [sobel_map1_map0, sobel_map1_map1, sobel_map1_map2],
                  [sobel_map2_map0, sobel_map2_map1, sobel_map2_map2]])

    ''' M ==
    [0,0]  [0,1]  [0,2]
    [1,0]  [1,1]  [1,2]
    [2,0]  [2,1]  [2,2]
    '''

    detM = (M[0,0] * (M[1,1] * M[2,2] - M[2,1] * M[1,2]) -
            M[0,1] * (M[1,0] * M[2,2] - M[2,0] * M[1,2]) +
            M[0,2] * (M[1,0] * M[2,1] - M[2,0] * M[1,1]))
    traceM = M[0,0] + M[1,1] + M[2,2]
    harris_map = detM / (traceM * traceM + 1e-8)

    return harris_map


def get_harris_feature(img, harris_map, top_num=50, min_dist=5):

    # harris_threshold = harris_map.mean() * threshold
    harris_threshold = 0
    mask = 1 * (harris_map > harris_threshold)
    points = np.array(mask.nonzero()).T
    harris_value = [harris_map[p[0], p[1], p[2]] for p in points]
    descending_list = np.flip(np.argsort(harris_value), 0)

    allowed_map = np.zeros(harris_map.shape)
    allowed_map[min_dist:-min_dist, min_dist:-min_dist, min_dist:-min_dist] = 1

    harris_points = []
    for i in descending_list:
        if allowed_map[points[i,0], points[i,1], points[i,2]] == 0:
            continue

        print(points[i], harris_value[i])
        harris_points.append(points[i])
        allowed_map[points[i,0]-min_dist:points[i,0]+min_dist,
                    points[i,1]-min_dist:points[i,1]+min_dist,
                    points[i,2]-min_dist:points[i,2]+min_dist] = 0
        if len(harris_points) >= top_num:
            break

    descriptors = []
    for point in harris_points:
        patch = img[point[0]-min_dist:point[0]+min_dist+1,
                    point[1]-min_dist:point[1]+min_dist+1,
                    point[2]-min_dist:point[2]+min_dist+1]
        descriptors.append(patch.flatten())

    return np.array(harris_points), np.array(descriptors)


def get_affine_params(template_data, matched_data):

    template_harris_map = get_harris_map(template_data)
    matched_harris_map = get_harris_map(matched_data)

    template_harris_points, template_descriptor = get_harris_feature(template_data, template_harris_map)
    print('')
    matched_harris_points, matched_descriptor = get_harris_feature(matched_data, matched_harris_map)

    matches = feature.match_descriptors(template_descriptor, matched_descriptor, metric='euclidean')

    A0 = np.ones(matches.shape)
    b0 = np.ones(matches.shape[0])
    A1 = np.ones(matches.shape)
    b1 = np.ones(matches.shape[0])
    A2 = np.ones(matches.shape)
    b2 = np.ones(matches.shape[0])
    for i in range(len(matches)):
        A0[i, 0] = matched_harris_points[matches[i, 1], 0]
        b0[i] = template_harris_points[matches[i, 0], 0]
        A1[i, 0] = matched_harris_points[matches[i, 1], 1]
        b1[i] = template_harris_points[matches[i, 0], 1]
        A2[i, 0] = matched_harris_points[matches[i, 1], 2]
        b2[i] = template_harris_points[matches[i, 0], 2]

    result0 = optimize.lsq_linear(A0, b0)
    result1 = optimize.lsq_linear(A1, b1)
    result2 = optimize.lsq_linear(A2, b2)

    affine_params = np.array([result0.x[0], result1.x[0], result2.x[0], result0.x[1], result1.x[1], result2.x[1]])
    print(affine_params)

    # src = []
    # dst = []
    # for pair in matches:
    #     src.append(matched_harris_points[pair[1]])
    #     dst.append(template_harris_points[pair[0]])
    # src = np.array(src)
    # dst = np.array(dst)
    #
    # affine_robust, inliners = measure.ransac((src, dst), transform.AffineTransform, min_samples=4, residual_threshold=2)
    #
    # print(affine_robust)

    return affine_params


def get_zoom_parameter(template_data, matched_data, axis):

    template_1D_len = template_data.shape[axis]
    matched_1D_len = matched_data.shape[axis]

    template_1D = np.zeros(template_1D_len)
    matched_1D = np.zeros(matched_1D_len)

    if axis == 0:
        for i in range(template_1D_len):
            template_1D[i] = template_data[i,:,:].mean()
        for i in range(matched_1D_len):
            matched_1D[i] = matched_data[i,:,:].mean()

    if axis == 1:
        for i in range(template_1D_len):
            template_1D[i] = template_data[:,i,:].mean()
        for i in range(matched_1D_len):
            matched_1D[i] = matched_data[:,i,:].mean()

    if axis == 2:
        for i in range(template_1D_len):
            template_1D[i] = template_data[:,:,i].mean()
        for i in range(matched_1D_len):
            matched_1D[i] = matched_data[:,:,i].mean()

    template_1D_center = ndimage.measurements.center_of_mass(template_1D)
    matched_1D_center = ndimage.measurements.center_of_mass(matched_1D)

    template_1D_left = template_1D[:template_1D_center]
    template_1D_right = template_1D[template_1D_center:]

    matched_1D_left = matched_1D[:matched_1D_center]
    matched_1D_right = matched_1D[matched_1D_center:]

    pass


def get_div(template_am, matched_am):

    div = (abs(template_am - matched_am)).mean()

    return div


if __name__ == '__main__':
    '''
    slice_0 = epi_img_data[26, :, :]
    slice_1 = epi_img_data[:, 30, :]
    slice_2 = epi_img_data[:, :, 16]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image")
    '''

    data0 = np.zeros((40, 40, 40))
    data1 = np.zeros((50, 50, 50))

    data0_anchors = np.array([10, 30])
    data0[data0_anchors[0], data0_anchors[0]:data0_anchors[1], data0_anchors[0]:data0_anchors[1]] = 1
    data0[data0_anchors[1], data0_anchors[0]:data0_anchors[1], data0_anchors[0]:data0_anchors[1]] = 1
    data0[data0_anchors[0]:data0_anchors[1], data0_anchors[0], data0_anchors[0]:data0_anchors[1]] = 1
    data0[data0_anchors[0]:data0_anchors[1], data0_anchors[1], data0_anchors[0]:data0_anchors[1]] = 1
    data0[data0_anchors[0]:data0_anchors[1], data0_anchors[0]:data0_anchors[1], data0_anchors[0]] = 1
    data0[data0_anchors[0]:data0_anchors[1], data0_anchors[0]:data0_anchors[1], data0_anchors[1]] = 1

    data1_anchors = np.array([15, 30])
    data1[data1_anchors[0], data1_anchors[0]:data1_anchors[1], data1_anchors[0]:data1_anchors[1]] = 1
    data1[data1_anchors[1], data1_anchors[0]:data1_anchors[1], data1_anchors[0]:data1_anchors[1]] = 1
    data1[data1_anchors[0]:data1_anchors[1], data1_anchors[0], data1_anchors[0]:data1_anchors[1]] = 1
    data1[data1_anchors[0]:data1_anchors[1], data1_anchors[1], data1_anchors[0]:data1_anchors[1]] = 1
    data1[data1_anchors[0]:data1_anchors[1], data1_anchors[0]:data1_anchors[1], data1_anchors[0]] = 1
    data1[data1_anchors[0]:data1_anchors[1], data1_anchors[0]:data1_anchors[1], data1_anchors[1]] = 1

    get_affine_params(data0,data1)
    pass

    # params = get_affine_params(data0, data1)
    # print(params)
    #
    # show_3Ddata_comp(data0, data1)