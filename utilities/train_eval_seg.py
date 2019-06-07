#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
from utilities.utils import AverageMeter
import time
from utilities.metrics.segmentation_miou import MIOU
from utilities.print_utils import *
from torch.nn.parallel import gather

def train_seg(model, dataset_loader, optimizer, criterion, num_classes, epoch, device='cuda'):
    losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    end = time.time()
    model.train()

    miou_class = MIOU(num_classes=num_classes)

    for i, (inputs, target) in enumerate(dataset_loader):
        inputs = inputs.to(device=device)
        target = target.to(device=device)

        outputs = model(inputs)

        if device == 'cuda':
            loss = criterion(outputs, target).mean()
            if isinstance(outputs, (list, tuple)):
                target_dev = outputs[0].device
                outputs = gather(outputs, target_device=target_dev)
        else:
            loss = criterion(outputs, target)

        inter, union = miou_class.get_iou(outputs, target)

        inter_meter.update(inter)
        union_meter.update(union)

        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:  # print after every 100 batches
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                  (epoch, i, len(dataset_loader), batch_time.avg, losses.avg, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100
    return miou, losses.avg


def val_seg(model, dataset_loader, criterion=None, num_classes=21, device='cuda'):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes)

    if criterion:
        losses = AverageMeter()

    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataset_loader):
            inputs = inputs.to(device=device)
            target = target.to(device=device)
            outputs = model(inputs)

            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)
                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion is not None else 0
                print_log_message("[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                      (i, len(dataset_loader), batch_time.avg, loss_, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))
    if criterion:
        return miou, losses.avg
    else:
        return miou, 0