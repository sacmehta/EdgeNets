# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

'''
This file contains the standard SSD configuration
'''
import numpy as np
import math

class SSD300Configuration(object):
    # match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
    iou_threshold = 0.45
    neg_pos_ratio = 3
    center_variance = 0.1
    size_variance = 0.2
    image_size = 300


    # PRIOR related settings
    strides = [8, 16, 32, 64, 100, 300]
    m = len(strides)

    feature_maps = []
    for stride in strides:
        temp = int(math.ceil(image_size/ stride))
        feature_maps.append(temp)
    #feature_maps = [38, 19, 10, 5, 3, 1]

    scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]    #priors scales 
    # s_max_size = int(math.ceil(1.05 * image_size))
    # s_min_size = int(math.ceil(0.1 * image_size))
    # sizes = [int(k) for k in np.linspace(s_min_size, s_max_size, m+1)]
    # min_sizes = sizes[:m]
    # max_sizes = sizes[1:]
    #min_sizes = [30, 60, 111, 162, 213, 264]
    #max_sizes = [60, 111, 162, 213, 264, 315]

    # aspect ratio contains a list of pair (e.g. [2, 2] or [2,3] or single valued list e.g. [2,]
    # This has a relationship with # of boxes per location. For example, [2,] means that 4 (=2*2) boxes per location
    # [2, 3] means that 6=(2*2) boxes per location
    # aspect_ratio = [[2, 3]] * m

    # Aspects ratios for diffrent feature maps
    aspect_ratios = [
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 3.0, 0.5, 0.333],
        [1.0, 2.0, 3.0, 0.5, 0.333],
        [1.0, 2.0, 3.0, 0.5, 0.333],
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5],
    ]

    box_per_location = []  # number of boxes per feature map location  [4,6,6,6,4,4]
    for pair in aspect_ratios:
        if len(pair) == 3:
            box_per_location.append(4)
        else:
            box_per_location.append(6)

   assert len(feature_maps) == len(scales) == len(aspect_ratio)

    clip = True

    # test specific options
    nms_threshold = 0.45
    conf_threshold = 0.01 # change this value during demo
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
    strides = [8, 16, 32, 64, 128, 512]
    m = len(strides)
    feature_maps = []
    for stride in strides:
        temp = int(math.ceil(image_size / stride))
        feature_maps.append(temp)
    
    scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9] #priors scales 
    #min_sizes = [36, 77, 154, 230, 307,  461]
    #max_sizes = [77, 154, 230, 307, 384, 538]
    # s_max_size = int(math.ceil(1.05 * image_size))
    # s_min_size = int(math.ceil(0.1 * image_size))
    # sizes = [int(k) for k in np.linspace(s_min_size, s_max_size, m + 1)]
    # min_sizes = sizes[:m]
    # max_sizes = sizes[1:]

    # aspect ratio contains a list of pair (e.g. [2, 2] or [2,3] or single valued list e.g. [2,]
    # This has a relationship with # of boxes per location. For example, [2,] means that 4 (=2*2) boxes per location
    # [2, 3] means that 6=(2*2) boxes per location

    # Aspects ratios for diffrent feature maps
    aspect_ratios = [
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 3.0, 0.5, 0.333],
        [1.0, 2.0, 3.0, 0.5, 0.333],
        [1.0, 2.0, 3.0, 0.5, 0.333],
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5],
    ]
    box_per_location = []  # number of boxes per feature map location  [4,6,6,6,4,4]
    for pair in aspect_ratios:
        if len(pair) == 3:
            box_per_location.append(4)
        else:
            box_per_location.append(6)
    clip = True

    assert len(feature_maps) == len(scales) == len(aspect_ratio)

    # test specific options
    nms_threshold = 0.45
    conf_threshold = 0.01 # change this value during demo
    top_k = 200 # MAX detections per class
    max_per_image = -1



def get_config(im_size):
    if im_size == 300:
        return SSD300Configuration()
    elif im_size == 512:
        return SSD512Configuration()
    else:
        print('{} image size not supported'.format(im_size))


