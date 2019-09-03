# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import torch
import torch.utils.data as data
import os
from PIL import Image
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize
import numpy as np
from transforms.classification.data_transforms import MEAN, STD

'''
This is a (untested) sample file for supporting custom datasets
'''

# let us assume that custom dataset has two classes, "Person" and "Background". Background is everything except persons
CUSTOM_DATASET_CLASS_LIST = ['background', 'person']


class CustomClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, root, inp_size=224, scale=(0.2, 1.0), is_training=True, debug_labels=True):
        super(CustomClassificationDataset, self).__init__()
        # let us assume that you have data directory set-up like this
        # + vision_datasets
        # +++++ train.txt // each line in this file contains mapping about RGB image and class <image1.jpg>, Class_id
        # +++++ val.txt // each line in this file contains mapping about RGB image and class <image1.jpg>, Class_id
        # +++++ images // This directory contains RGB images
        # +++++++++ image1.jpg
        # +++++++++ image2.jpg

        rgb_root_dir = os.path.join(root, 'images')
        self.train = is_training
        if self.train:
            data_file = os.path.join(root, 'train.txt')
        else:
            data_file = os.path.join(root, 'val.txt')

        self.images = []
        self.labels = []
        with open(data_file, 'r') as lines:
            for line in lines:
                # line is a comma separated file that contains mapping between RGB image and class iD
                # <image1.jpg>, Class_ID
                line_split = line.split(',') # index 0 contains rgb location and index 1 contains label id
                rgb_img_loc = rgb_root_dir + os.sep + line_split[0].strip()
                label_id = int(line_split[1].strip()) #strip to remove spaces
                assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)

                if debug_labels:
                    # let us also check that label index  only contain labels as listed in CUSTOM_DATASET_CLASS_LIST
                    assert 0 <= label_id < len(CUSTOM_DATASET_CLASS_LIST), "Label should contain values only between 0 " \
                                                    "and {}".format(len(CUSTOM_DATASET_CLASS_LIST) - 1)
                self.labels.append(label_id)

        self.transform = self.transforms(inp_size=inp_size, inp_scale=scale, is_training=is_training)

    def transforms(self, inp_size, inp_scale, is_training):
        if is_training:
            return Compose(
                [
                    RandomResizedCrop(inp_size, scale=inp_scale),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=MEAN, std=STD)
                ]
            )
        else:
            return Compose(
                [
                    Resize(size=(inp_size, inp_size)),
                    ToTensor(),
                    Normalize(mean=MEAN, std=STD)
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_id = self.labels[index]

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        return rgb_img, label_id
