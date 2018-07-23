import numpy as np
import sys


def _conv(img, conv_fliter):
    """计算一层的卷积"""
    filter_size = conv_fliter.shape[1]
    result = np.zeros(img.shape)
    for r in np.uint16(np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 + 1)):
        for c in np.uint16(np.arange(filter_size / 2, img.shape[1] - filter_size / 2.0 + 1)):
            curr_region = img[r - np.uint16(np.floor(filter_size / 2.0)):r + np.uint16(np.ceil(filter_size / 2.0))]
            curr_region = curr_region * conv_fliter
            conv_sum = np.sum(curr_region)
            result[r, c] = conv_sum

    final_result = result[np.uint16(filter_size / 2.0):result.shape[0] - np.uint16(filter_size / 2.0),
                   np.uint16(filter_size / 2.0):result.shape[1] - np.uint16(filter_size / 2.0)]
    return final_result


def conv(img, conv_filter):
    """执行卷及操作
    img.shape = [img_size , img_size , channels_num]
    conv_filter.shape = [,,,conv_size,size,channels_num]
    """

    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: 图像和过滤模板的通道数必须一致")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("Error: 过滤模板必须是方形的")
        sys.exit()

    if conv_filter.shape[1] % 2 == 0:
        print("Error: 模板的尺寸必须是奇数")
        sys.exit()

    # 预定义结果模板
    feature_map = np.zeros((
        img.shape[0] - conv_filter.shape[1] + 1,
        img.shape[1] - conv_filter.shape[1] + 1,
        conv_filter.shape[0]
    ))

    #     计算卷积
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]

        if len(curr_filter.shape) > 2:
            conv_map = _conv(img[:, :, 0], curr_filter[:, :, 0])
            # 如果模板和图片都有多个通道
            for ch_num in curr_filter.shape[-1]:
                conv_map = conv_map + _conv(img[:, :, ch_num], curr_filter[:, :, ch_num])
        else:
            conv_map = _conv(img, curr_filter)

        feature_map[:, :, filter_num] = conv_map


def relu(feature_map):
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in range(feature_map.shape[0]):
            for c in range(feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])

    return relu_out


def pooling(feature_map, size=2, stride=2):
    pool_out = np.zeros(
        (np.uint16((feature_map.shape[0] - size) / stride + 1), np.uint16((feature_map.shape[1] - size) / stride + 1)))

    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0] - size, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r + size, c:c + size]])
                c2 = c2 + 1
            r2 = r2 + 1
        return pool_out
