# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import os
import torch.utils.data
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize
from transforms.classification.data_transforms import MEAN, STD
from torch.utils import data

COCO_CLASS_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush'
                   ]


class COCOClassification(data.Dataset):
    def __init__(self, root, split='train', year='2017', inp_size=224, scale=(0.2, 1.0), is_training=True):
        super(COCOClassification, self).__init__()
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        self.root = root
        self.img_dir = os.path.join(root, 'images/{}{}'.format(split, year))
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
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
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, len(COCO_CLASS_LIST)), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target
