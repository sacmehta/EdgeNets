import torch
import argparse
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from data_loader.classification import imagenet as img_loader
import random
import os
from torch.utils.tensorboard import SummaryWriter
import time
from utilities.utils import model_parameters, compute_flops
from utilities.utils import save_checkpoint
import numpy as np
from utilities.print_utils import *
from torch import nn

# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================


def main(args):
    # -----------------------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------------------
    if args.model == 'dicenet':
        from model.classification import dicenet as net
        model = net.CNNModel(args)
    elif args.model == 'espnetv2':
        from model.classification import espnetv2 as net
        model = net.EESPNet(args)
    elif args.model == 'shufflenetv2':
        from model.classification import shufflenetv2 as net
        model = net.CNNModel(args)
    else:
        print_error_message('Model {} not yet implemented'.format(args.model))
        exit()

    if args.finetune:
        # laod the weights for finetuning
        if os.path.isfile(args.weights_ft):
            pretrained_dict = torch.load(args.weights_ft, map_location=torch.device('cpu'))
            print_info_message('Loading pretrained basenet model weights')
            model_dict = model.state_dict()

            overlap_dict = {k: v for k, v in model_dict.items() if k in pretrained_dict}

            total_size_overlap = 0
            for k, v in enumerate(overlap_dict):
                total_size_overlap += torch.numel(overlap_dict[v])

            total_size_pretrain = 0
            for k, v in enumerate(pretrained_dict):
                total_size_pretrain += torch.numel(pretrained_dict[v])

            if len(overlap_dict) == 0:
                print_error_message('No overlaping weights between model file and pretrained weight file. Please check')

            print_info_message('Overlap ratio of weights: {:.2f} %'.format(
                (total_size_overlap * 100.0) / total_size_pretrain))

            model_dict.update(overlap_dict)
            model.load_state_dict(model_dict, strict=False)
            print_info_message('Pretrained basenet model loaded!!')
        else:
            print_error_message('Unable to find the weights: {}'.format(args.weights_ft))

    # -----------------------------------------------------------------------------
    # Writer for logging
    # -----------------------------------------------------------------------------
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    writer = SummaryWriter(log_dir=args.savedir, comment='Training and Validation logs')
    try:
        writer.add_graph(model, input_to_model=torch.randn(1, 3, args.inpSize, args.inpSize))
    except:
        print_log_message("Not able to generate the graph. Likely because your model is not supported by ONNX")

    # network properties
    num_params = model_parameters(model)
    flops = compute_flops(model)
    print_info_message('FLOPs: {:.2f} million'.format(flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    # -----------------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------------

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    best_acc = 0.0
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'
    if args.resume:
        if os.path.isfile(args.resume):
            print_info_message("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], map_location=torch.device(device))
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_info_message("=> loaded checkpoint '{}' (epoch {})"
                               .format(args.resume, checkpoint['epoch']))
        else:
            print_warning_message("=> no checkpoint found at '{}'".format(args.resume))

    # -----------------------------------------------------------------------------
    # Loss Fn
    # -----------------------------------------------------------------------------
    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss()
        acc_metric = 'Top-1'
    elif args.dataset == 'coco':
        criterion = nn.BCEWithLogitsLoss()
        acc_metric = 'F1'
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # -----------------------------------------------------------------------------
    # Data Loaders
    # -----------------------------------------------------------------------------
    # Data loading code
    if args.dataset == 'imagenet':
        train_loader, val_loader = img_loader.data_loaders(args)
        # import the loaders too
        from utilities.train_eval_classification import train, validate
    elif args.dataset == 'coco':
        from data_loader.classification.coco import COCOClassification
        train_dataset = COCOClassification(root=args.data, split='train', year='2017', inp_size=args.inpSize,
                                           scale=args.scale, is_training=True)
        val_dataset = COCOClassification(root=args.data, split='val', year='2017', inp_size=args.inpSize,
                                         is_training=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.workers)

        # import the loaders too
        from utilities.train_eval_classification import train_multi as train
        from utilities.train_eval_classification import validate_multi as validate
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    # -----------------------------------------------------------------------------
    # LR schedulers
    # -----------------------------------------------------------------------------
    if args.scheduler == 'fixed':
        step_sizes = args.steps
        from utilities.lr_scheduler import FixedMultiStepLR
        lr_scheduler = FixedMultiStepLR(base_lr=args.lr, steps=step_sizes, gamma=args.lr_decay)
    elif args.scheduler == 'clr':
        from utilities.lr_scheduler import CyclicLR
        step_sizes = args.steps
        lr_scheduler = CyclicLR(min_lr=args.lr, cycle_len=5, steps=step_sizes, gamma=args.lr_decay)
    elif args.scheduler == 'poly':
        from utilities.lr_scheduler import PolyLR
        lr_scheduler = PolyLR(base_lr=args.lr, max_epochs=args.epochs)
    elif args.scheduler == 'linear':
        from utilities.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(base_lr=args.lr, max_epochs=args.epochs)
    elif args.scheduler == 'hybrid':
        from utilities.lr_scheduler import HybirdLR
        lr_scheduler = HybirdLR(base_lr=args.lr, max_epochs=args.epochs, clr_max=args.clr_max)
    else:
        print_error_message('Scheduler ({}) not yet implemented'.format(args.scheduler))
        exit()

    print_info_message(lr_scheduler)

    # set up the epoch variable in case resuming training
    if args.start_epoch != 0:
        for epoch in range(args.start_epoch):
            lr_scheduler.step(epoch)

    with open(args.savedir + os.sep + 'arguments.json', 'w') as outfile:
        import json
        arg_dict = vars(args)
        arg_dict['model_params'] = '{} '.format(num_params)
        arg_dict['flops'] = '{} '.format(flops)
        json.dump(arg_dict, outfile)

    # -----------------------------------------------------------------------------
    # Training and Val Loop
    # -----------------------------------------------------------------------------

    extra_info_ckpt = args.model + '_' + str(args.s)
    for epoch in range(args.start_epoch, args.epochs):
        lr_log = lr_scheduler.step(epoch)
        # set the optimizer with the learning rate
        # This can be done inside the MyLRScheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_log
        print_info_message("LR for epoch {} = {:.5f}".format(epoch, lr_log))
        train_acc, train_loss = train(data_loader=train_loader, model=model, criteria=criterion, optimizer=optimizer,
                                      epoch=epoch, device=device)
        # evaluate on validation set
        val_acc, val_loss = validate(data_loader=val_loader, model=model, criteria=criterion, device=device)

        # remember best prec@1 and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': weights_dict,
            'best_prec1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.savedir, extra_info_ckpt)

        writer.add_scalar('Classification/LR/learning_rate', lr_log, epoch)
        writer.add_scalar('Classification/Loss/Train', train_loss, epoch)
        writer.add_scalar('Classification/Loss/Val', val_loss, epoch)
        writer.add_scalar('Classification/{}/Train'.format(acc_metric), train_acc, epoch)
        writer.add_scalar('Classification/{}/Val'.format(acc_metric), val_acc, epoch)
        writer.add_scalar('Classification/Complexity/Top1_vs_flops', best_acc, round(flops, 2))
        writer.add_scalar('Classification/Complexity/Top1_vs_params', best_acc, round(num_params, 2))

    writer.close()


if __name__ == '__main__':
    from commons.general_details import classification_models, classification_datasets, classification_exp_choices, \
        classification_schedulers

    parser = argparse.ArgumentParser(description='Training efficient networks')
    # General settings
    parser.add_argument('--workers', default=12, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size (default: 512)')

    # Dataset related settings
    parser.add_argument('--data', default='', help='path to dataset')
    parser.add_argument('--dataset', default='imagenet', help='Name of the dataset', choices=classification_datasets)

    # LR scheduler settings
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--clr-max', default=61, type=int, help='Max. epochs for CLR in Hybrid scheduler')
    parser.add_argument('--steps', default=[51, 101, 131, 161, 191, 221, 251, 281], type=int, nargs="+",
                        help='steps at which lr should be decreased. Only used for Cyclic and Fixed LR')
    parser.add_argument('--scheduler', default='clr', choices=classification_schedulers,
                        help='Learning rate scheduler')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay', default=0.5, type=float, help='factor by which lr should be decreased')

    # optimizer settings
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=4e-5, type=float, help='weight decay (default: 4e-5)')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--savedir', type=str, default='results_classification', help='Location to save the results')

    # Model settings
    parser.add_argument('--s', default=1.0, type=float, help='Factor by which output channels should be scaled (s > 1 '
                                                           'for increasing the dims while < 1 for decreasing)')
    parser.add_argument('--inpSize', default=224, type=int, help='Input image size (default: 224 x 224)')
    parser.add_argument('--scale', default=[0.2, 1.0], type=float, nargs="+", help='Scale for data augmentation')
    parser.add_argument('--model', default='shuffle_vw', choices=classification_models,
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    # DiceNet related settings
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')

    ## Experiment related settings
    parser.add_argument('--exp-type', type=str, choices=classification_exp_choices, default='main',
                        help='Experiment type')
    parser.add_argument('--finetune', action='store_true', default=False, help='Finetune the model')

    args = parser.parse_args()

    assert len(args.scale) == 2
    args.scale = tuple(args.scale)

    random.seed(1882)
    torch.manual_seed(1882)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.savedir = '{}_{}/model_{}_{}/aug_{}_{}/s_{}_inp_{}_sch_{}/{}/'.format(args.savedir, args.exp_type, args.model,
                                                                               args.dataset, args.scale[0],
                                                                               args.scale[1],
                                                                               args.s, args.inpSize, args.scheduler,
                                                                               timestr)

    # if you want to finetune ImageNet model on other dataset, say MS-COCO classification
    if args.finetune:
        print_info_message('Grabbing location of the ImageNet weights from the weight dictionary')
        from model.weight_locations.classification import model_weight_map

        weight_file_key = '{}_{}'.format(args.model, args.s)
        assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
        args.weights_ft = model_weight_map[weight_file_key]


    if args.dataset == 'imagenet':
        args.num_classes = 1000
    elif args.dataset == 'coco':
        from data_loader.classification.coco import COCO_CLASS_LIST
        args.num_classes = len(COCO_CLASS_LIST)

    main(args)
