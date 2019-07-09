# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import os
import torch.utils.data
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

COCO_CLASS_LIST = ['__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
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
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush'
                ]

class COCOObjectDetection(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None, is_training=True):
        super(COCOObjectDetection, self).__init__()
        if is_training:
            split = 'train'
            year = '2017'
        else:
            split = 'val'
            year = '2017'
        ann_file = os.path.join(root_dir, 'annotations/instances_{}{}.json'.format(split, year))
        self.coco = COCO(ann_file)
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images/{}{}'.format(split, year))
        self.transform = transform
        self.target_transform = target_transform
        if is_training:
            #Remove image IDS that don't have annotations from training set
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # use all images from Validation Set
            self.ids = list(self.coco.imgs.keys())
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}
        self.CLASSES = COCO_CLASS_LIST

    def __getitem__(self, index):
        img_id = self.ids[index]
        boxes, labels = self._get_annotation(img_id)
        image = self._get_image(img_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._get_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def _get_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.img_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
