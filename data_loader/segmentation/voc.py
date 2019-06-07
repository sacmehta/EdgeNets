# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import torch
import torch.utils.data as data
import os
from PIL import Image
from transforms.segmentation.data_transforms import RandomFlip, RandomCrop, RandomScale, Normalize, Resize, Compose


VOC_CLASS_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                   'tv/monitor']

class VOCSegmentation(torch.utils.data.Dataset):

    def __init__(self, root, train=True, scale=(0.5, 1.0), crop_size=(513, 513), ignore_idx=255, coco_root_dir=''):
        super(VOCSegmentation, self).__init__()
        voc_root_dir = os.path.join(root, 'VOC2012')
        voc_list_dir = os.path.join(voc_root_dir, 'list')

        self.train = train
        if self.train:
            data_file = os.path.join(voc_list_dir, 'train_aug.txt')
            if coco_root_dir:
                data_file_coco = os.path.join(coco_root_dir, 'train_2017.txt')
        else:
            data_file = os.path.join(voc_list_dir, 'val.txt')

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                rgb_img_loc = voc_root_dir + os.sep  + line.split()[0]
                label_img_loc = voc_root_dir + os.sep + line.split()[1]
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)

        if self.train and coco_root_dir:
            with open(data_file_coco, 'r') as lines:
                for line in lines:
                    rgb_img_loc = coco_root_dir + os.sep + line.split()[0]
                    label_img_loc = coco_root_dir + os.sep + line.split()[1]
                    assert os.path.isfile(rgb_img_loc)
                    assert os.path.isfile(label_img_loc)
                    self.images.append(rgb_img_loc)
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



