# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

from itertools import product

import torch
import torch.nn as nn
from math import sqrt

class PriorBox(nn.Module):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg.image_size
        self.feature_maps = cfg.feature_maps
        self.min_sizes = cfg.min_sizes
        self.max_sizes = cfg.max_sizes
        self.strides = cfg.strides
        self.aspect_ratios = cfg.aspect_ratio
        self.clip = cfg.clip

    def forward(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.Tensor(priors)
        if self.clip:
            priors.clamp_(max=1.0, min=0)
        return priors


if __name__ == '__main__':
    from model.detection.ssd_config import SSD300Configuration as cfg
    center_form_priors = PriorBox(cfg)()
    from utilities import box_utils
    corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
    print(corner_form_priors)

