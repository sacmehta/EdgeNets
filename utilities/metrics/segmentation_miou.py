#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import numpy as np
import torch

class MIOU(object):
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def get_iou(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        _, pred = torch.max(output, 1)

        # histc in torch is implemented only for cpu tensors, so move your tensors to CPU
        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()

        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        # shift by 1 so that 255 is 0
        pred += 1
        target += 1

        pred = pred * (target > 0)
        inter = pred * (pred == target)
        area_inter = torch.histc(inter.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_pred = torch.histc(pred.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_mask = torch.histc(target.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_union = area_pred + area_mask - area_inter + self.epsilon

        return area_inter.numpy(), area_union.numpy()


if __name__ == '__main__':
    from utilities.utils import AverageMeter
    inter = AverageMeter()
    union = AverageMeter()
    a = torch.Tensor(1, 21, 224, 224).random_(254, 256)
    b = torch.Tensor(1, 21, 224, 224).random_(254, 256 )

    m = MIOU()
    print(m.get_iou(a, b))
