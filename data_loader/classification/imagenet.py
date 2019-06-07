#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import os
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from transforms.classification.data_transforms import Lighting, normalize


def train_transforms(inp_size, scale):
    return transforms.Compose([
        transforms.RandomResizedCrop(inp_size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def val_transforms(inp_size):
    return transforms.Compose([
        transforms.Resize(int(inp_size / 0.875)),
        transforms.CenterCrop(inp_size),
        transforms.ToTensor(),
        normalize,
    ])

#helper function for the loading the training data
def train_loader(args):
    traindir = os.path.join(args.data, 'train')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, train_transforms(args.inpSize, scale=args.scale)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    return train_loader


#helper function for the loading the validation data
def val_loader(args):
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms(args.inpSize)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

# helped function for loading trianing and validation data
def data_loaders(args):
    tr_loader = train_loader(args)
    vl_loader = val_loader(args)
    return tr_loader, vl_loader

