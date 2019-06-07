# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

from pycocotools.coco import COCO
from pycocotools import mask
from tqdm import tqdm
import numpy as np
import torch
import os
from PIL import Image
import numba as numba
from numba import prange
from torch.utils import data


class COCOSegmentation(data.Dataset):
    # these are the same as the PASCAL VOC dataset
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    def __init__(self, root_dir, split='train', year='2017'):
        super(COCOSegmentation, self).__init__()
        ann_file = os.path.join(root_dir, 'annotations/instances_{}{}.json'.format(split, year))
        self.img_dir = os.path.join(root_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.ids = list(self.coco.imgs.keys())

    def generate_img_mask_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']

        if img_metadata['height'] < 256 or img_metadata['width'] < 256:
            return None, None, None

        try:
            _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
            cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

            _target = Image.fromarray(self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width']))

            len_uniq_vals = len(np.unique(_target))
            if len_uniq_vals < 2:
                return None, None, None

            return path, _img, _target
        except OSError:
            return None, None, None

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask


@numba.jit(target='cpu')
def generate_pairs(coco, rgb_dir_name, mask_dir_name, save_dir_rgb, save_dir_mask):
    lines = []
    for i in tqdm(prange(len(coco.ids))):
        fname, img, mask = coco.generate_img_mask_pair(i)
        if fname is None:
            continue
        img.save(os.path.join(save_dir_rgb, fname))

        mask_fname = fname.split('.')[0] + '.png'
        mask.save(os.path.join(save_dir_mask, mask_fname))
        loc_pair = '{}/{} {}/{}\n'.format(rgb_dir_name, fname, mask_dir_name, mask_fname)
        lines.append(loc_pair)

    return lines


if __name__ == "__main__":
    root_dir = '../../../vision_datasets/coco/'
    save_root_dir = '../../../vision_datasets/coco_preprocess'
    rgb_dir_name = 'rgb_train'
    mask_dir_name = 'annot_train'

    # process the training split
    split = 'train'
    year = '2017'

    save_dir_rgb = save_root_dir + os.sep + rgb_dir_name
    save_dir_mask = save_root_dir + os.sep + mask_dir_name
    if not os.path.isdir(save_dir_rgb):
        os.makedirs(save_dir_rgb)

    if not os.path.isdir(save_dir_mask):
        os.makedirs(save_dir_mask)

    coco = COCOSegmentation(root_dir, split=split, year=year)

    lines = generate_pairs(coco, rgb_dir_name=rgb_dir_name, mask_dir_name=mask_dir_name,
                           save_dir_rgb=save_dir_rgb, save_dir_mask=save_dir_mask)

    with open(save_root_dir + os.sep + '{}_{}.txt'.format(split, year), 'w') as txt_file:
        for line in lines:
            txt_file.write(line)
