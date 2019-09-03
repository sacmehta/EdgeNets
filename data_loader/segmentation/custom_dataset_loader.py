# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import torch
import torch.utils.data as data
import os
from PIL import Image
from transforms.segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose
import numpy as np

'''
This is a (untested) sample file for supporting custom datasets
'''

# let us assume that custom dataset has two classes, "Person" and "Background". Background is everything except persons
CUSTOM_DATASET_CLASS_LIST = ['background', 'person']


class CustomSegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, root, train=True, scale=(0.5, 1.0), crop_size=(513, 513), ignore_idx=255, debug_labels=True):
        super(CustomSegmentationDataset, self).__init__()
        # let us assume that you have data directory set-up like this
        # + vision_datasets
        # +++++ train.txt // each line in this file contains mapping about RGB image and mask image <image1.jpg>, <image1.png>
        # +++++ val.txt // each line in this file contains mapping about RGB image and mask image <image1.jpg>, <image1.png>
        # +++++ images // This directory contains RGB images
        # +++++++++ image1.jpg
        # +++++++++ image2.jpg
        # +++++ annotations // This directory contains segmentation mask images
        # +++++++++ image1.png
        # +++++++++ image2.png

        rgb_root_dir = os.path.join(root, 'images')
        label_root_dir = os.path.join(root, 'annotations')
        self.train = train
        if self.train:
            data_file = os.path.join(root, 'train.txt')
        else:
            data_file = os.path.join(root, 'val.txt')

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                # line is a comma separated file that contains mapping between RGB image and mask image
                # <image1.jpg>, <image1.png>
                line_split = line.split(',')
                rgb_img_loc = rgb_root_dir + os.sep + line_split[0].strip()
                label_img_loc = label_root_dir + os.sep + line_split[1].strip()
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)

                if debug_labels:
                    # let us also check that label file only contain labels as listed in CUSTOM_DATASET_CLASS_LIST
                    label_img = Image.open(label_img_loc)
                    label_img = np.asarray(label_img)
                    unique_values = np.unique(label_img)
                    min_val = np.min(unique_values)
                    max_val = np.max(unique_values)
                    assert 0 <= min_val <= max_val < len(
                        CUSTOM_DATASET_CLASS_LIST), "Label image at {} has following values. " \
                                                    "However, it should contain values only between 0 " \
                                                    "and {}".format(unique_values.tolist(),
                                                                    len(CUSTOM_DATASET_CLASS_LIST) - 1)
                self.masks.append(label_img_loc)

        if isinstance(crop_size, tuple):
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomFlip(),
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.crop_size),
                Normalize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.crop_size),
                Normalize()
            ]
        )
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.masks[index])

        if self.train:
            rgb_img, label_img = self.train_transforms(rgb_img, label_img)
        else:
            rgb_img, label_img = self.val_transforms(rgb_img, label_img)

        return rgb_img, label_img
