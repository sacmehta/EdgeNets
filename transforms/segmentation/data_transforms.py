#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import random
from PIL import Image
import math
import torch
import numpy as np
import numbers
from torchvision.transforms import Pad
from torchvision.transforms import functional as F
from transforms.classification.data_transforms import MEAN, STD

class Normalize(object):
    '''
        Normalize the tensors
    '''
    def __call__(self, rgb_img, label_img=None):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, MEAN, STD) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img


class RandomFlip(object):
    '''
        Random Flipping
    '''
    def __call__(self, rgb_img, label_img):
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb_img, label_img


class RandomScale(object):
    '''
    Random scale, where scale is logrithmic
    '''
    def __init__(self, scale=(0.5, 1.0)):
        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        rgb_img = rgb_img.resize(new_size, Image.ANTIALIAS)
        label_img = label_img.resize(new_size, Image.NEAREST)
        return rgb_img, label_img


class RandomCrop(object):
    '''
    Randomly crop the image
    '''
    def __init__(self, crop_size, ignore_idx=255):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size
        pad_along_w = max(0, int((1 + self.crop_size[0] - w) / 2))
        pad_along_h = max(0, int((1 + self.crop_size[1] - h) / 2))
        # padd the images
        rgb_img = Pad(padding=(pad_along_w, pad_along_h), fill=0, padding_mode='constant')(rgb_img)
        label_img = Pad(padding=(pad_along_w, pad_along_h), fill=self.ignore_idx, padding_mode='constant')(label_img)

        i, j, h, w = self.get_params(rgb_img, self.crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)
        return rgb_img, label_img


class RandomResizedCrop(object):
    '''
    Randomly crop the image and then resize it
    '''
    def __init__(self, size, scale=(0.5, 1.0), ignore_idx=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img):
        w, h = rgb_img.size

        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (
                    math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        crop_size = (int(round(w * random_scale)), int(round(h * random_scale)))

        i, j, h, w = self.get_params(rgb_img, crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)

        rgb_img = rgb_img.resize(self.size, Image.ANTIALIAS)
        label_img = label_img.resize(self.size, Image.NEAREST)

        return rgb_img, label_img


class Resize(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, rgb_img, label_img):
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        label_img = label_img.resize(self.size, Image.NEAREST)
        return rgb_img, label_img


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
