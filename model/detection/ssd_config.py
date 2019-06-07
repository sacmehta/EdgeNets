# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

'''
This file contains the standard SSD configuration
'''
import numpy as np
class SSD300Configuration(object):
    # match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
    iou_threshold = 0.45
    neg_pos_ratio = 3
    center_variance = 0.1
    size_variance = 0.2
    image_size = 300


    # PRIOR related settings
    feature_maps = [38, 19, 10, 5, 3, 1]
    strides = [8, 16, 32, 64, 100, 300]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    # aspect ratio contains a list of pair (e.g. [2, 2] or [2,3] or single valued list e.g. [2,]
    # This has a relationship with # of boxes per location. For example, [2,] means that 4 (=2*2) boxes per location
    # [2, 3] means that 6=(2*2) boxes per location
    aspect_ratio = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    box_per_location = []  # number of boxes per feature map location
    for pair in aspect_ratio:
        if len(pair) == 1:
            box_per_location.append(pair[0] * pair[0])
        else:
            box_per_location.append(np.prod(pair))
    clip = True

    # test specific options
    nms_threshold = 0.45
    conf_threshold = 0.01
    top_k = 200 # MAX detections per class
    max_per_image = -1


class SSD512Configuration(object):
    # match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
    iou_threshold = 0.45
    neg_pos_ratio = 3
    center_variance = 0.1
    size_variance = 0.2
    image_size = 512


    # PRIOR related settings
    feature_maps = [64, 32, 16, 8, 4, 2, 1]
    strides = [8, 16, 32, 64, 128, 256, 512]

    min_sizes = [36, 77, 154, 230, 307, 384, 461]
    max_sizes = [77, 154, 230, 307, 384, 461, 538]
    # aspect ratio contains a list of pair (e.g. [2, 2] or [2,3] or single valued list e.g. [2,]
    # This has a relationship with # of boxes per location. For example, [2,] means that 4 (=2*2) boxes per location
    # [2, 3] means that 6=(2*2) boxes per location
    aspect_ratio = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    box_per_location = []  # number of boxes per feature map location
    for pair in aspect_ratio:
        if len(pair) == 1:
            box_per_location.append(pair[0] * pair[0])
        else:
            box_per_location.append(np.prod(pair))
    clip = True

    assert len(feature_maps) == len(strides) == len(min_sizes) == len(max_sizes) == len(aspect_ratio)

    # test specific options
    nms_threshold = 0.45
    conf_threshold = 0.01
    top_k = 200 # MAX detections per class
    max_per_image = -1



def get_config(im_size):
    if im_size == 300:
        return SSD300Configuration()
    elif im_size == 512:
        return SSD512Configuration()
    else:
        print('{} image size not supported'.format(im_size))
