#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch import nn


class SegmentationLoss(nn.Module):
    def __init__(self, n_classes=21, loss_type='ce', device='cuda', ignore_idx=255, class_wts=None):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.device = device
        self.ignore_idx = ignore_idx
        self.smooth = 1e-6
        self.class_wts = class_wts

        if self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_wts)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=self.class_wts)

    def convert_to_one_hot(self, x):
        n, h, w = x.size()
        # remove the 255 index
        x[x == self.ignore_idx] = self.n_classes
        x = x.unsqueeze(1)

        # convert to one hot vector
        x_one_hot = torch.zeros(n, self.n_classes + 1, h, w).to(device=self.device)
        x_one_hot = x_one_hot.scatter_(1, x, 1)

        return x_one_hot[:, :self.n_classes, :, :].contiguous()

    def forward(self, inputs, target):
        if isinstance(inputs, tuple):
            tuple_len = len(inputs)
            assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                if target.dim() == 3 and self.loss_type == 'bce':
                    target = self.convert_to_one_hot(target)
                loss_ = self.loss_fn(inputs[i], target)
                loss += loss_
        else:
            if target.dim() == 3 and self.loss_type == 'bce':
                target = self.convert_to_one_hot(target)
            return self.loss_fn(inputs, target)
        return loss
