#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

from utilities.utils import AverageMeter
import time
import torch
from utilities.print_utils import *


def train(data_loader, model, criterion, optimizer, device, epoch=-1):
    model.train()

    train_loss = AverageMeter()
    train_cl_loss = AverageMeter()
    train_loc_loss = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    for batch_idx, (images, boxes, labels) in enumerate(data_loader):
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidences, locations = model(images)
        regression_loss, classification_loss = criterion(confidences, locations, labels, boxes)

        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        N = images.size(0)
        train_loss.update(loss.item(), N)
        train_cl_loss.update(classification_loss.item(), N)
        train_loc_loss.update(regression_loss.item(), N)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx and batch_idx % 10 == 0:
            print_log_message(
                "Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tCls Loss: %.4f(%.4f)\t \tLoc Loss: %.4f(%.4f)\t\tTotal Loss: %.4f(%.4f)" %
                (epoch, batch_idx, len(data_loader), batch_time.avg,
                 train_cl_loss.val, train_cl_loss.avg,
                 train_loc_loss.val, train_loc_loss.avg,
                 train_loss.val, train_loss.avg
                 ))

    return train_loss.avg, train_cl_loss.avg, train_loc_loss.avg


def validate(data_loader, model, criterion, device, epoch):
    model.eval()

    val_loss = AverageMeter()
    val_cl_loss = AverageMeter()
    val_loc_loss = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (images, boxes, labels) in enumerate(data_loader):
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            confidences, locations = model(images)
            regression_loss, classification_loss = criterion(confidences, locations, labels, boxes)

            loss = regression_loss + classification_loss

            N = images.size(0)
            val_loss.update(loss.item(), N)
            val_cl_loss.update(classification_loss.item(), N)
            val_loc_loss.update(regression_loss.item(), N)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx and batch_idx % 10 == 0:
                print_log_message(
                    "Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tCls Loss: %.4f(%.4f)\t \tLoc Loss: %.4f(%.4f)\t\tTotal Loss: %.4f(%.4f)" %
                    (epoch, batch_idx, len(data_loader), batch_time.avg,
                     val_cl_loss.val, val_cl_loss.avg,
                     val_loc_loss.val, val_loc_loss.avg,
                     val_loss.val, val_loss.avg
                     ))
        return val_loss.avg, val_cl_loss.avg, val_loc_loss.avg

