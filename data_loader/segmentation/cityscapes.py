# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import torch
import torch.utils.data as data
import os
from PIL import Image
from transforms.segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose

CITYSCAPE_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']


class CityscapesSegmentation(data.Dataset):

    def __init__(self, root, train=True, scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255, coarse=True):

        self.train = train
        if self.train:
            data_file = os.path.join(root, 'train.txt')
            if coarse:
                coarse_data_file = os.path.join(root, 'train_coarse.txt')
        else:
            data_file = os.path.join(root, 'val.txt')

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(',')
                rgb_img_loc = root + os.sep + line_split[0].rstrip()
                label_img_loc = root + os.sep + line_split[1].rstrip()
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)

        # if you want to use Coarse data for training
        if train and coarse and os.path.isfile(coarse_data_file):
            with open(coarse_data_file, 'r') as lines:
                for line in lines:
                    line_split = line.split(',')
                    rgb_img_loc = root + os.sep + line_split[0].rstrip()
                    label_img_loc = root + os.sep + line_split[1].rstrip()
                    assert os.path.isfile(rgb_img_loc)
                    assert os.path.isfile(label_img_loc)
                    self.images.append(rgb_img_loc)
                    self.masks.append(label_img_loc)

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.size),
                RandomFlip(),
                Normalize()
            ]
        )
        val_transforms = Compose(
            [
                Resize(size=self.size),
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
