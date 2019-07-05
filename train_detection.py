import argparse
import os

import torch
from torch.utils.data import DataLoader
from data_loader.detection.augmentation import TrainTransform, ValTransform
from model.detection.match_priors import MatchPrior
from model.detection.generate_priors import PriorBox
from loss_fns.multi_box_loss import MultiBoxLoss
from utilities.train_eval_detect import train, validate
from utilities.utils import save_checkpoint, model_parameters, compute_flops
import math
from torch.utils.tensorboard import SummaryWriter
from utilities.print_utils import *
from model.detection.ssd import ssd

def main(args):
    if args.im_size in [300, 512]:
        from model.detection.ssd_config import get_config
        cfg = get_config(args.im_size)
    else:
        print_error_message('{} image size not supported'.format(args.im_size))

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    train_transform = TrainTransform(cfg.image_size)
    target_transform = MatchPrior(PriorBox(cfg)(), cfg.center_variance, cfg.size_variance, cfg.iou_threshold)
    val_transform = ValTransform(cfg.image_size)

    if args.dataset in ['voc', 'pascal']:
        from data_loader.detection.voc import VOCDataset, VOC_CLASS_LIST
        train_dataset_2007 = VOCDataset(root_dir=args.data_path, transform=train_transform,
                                        target_transform=target_transform,
                                        is_training=True, split="VOC2007")
        train_dataset_2012 = VOCDataset(root_dir=args.data_path, transform=train_transform,
                                        target_transform=target_transform,
                                        is_training=True, split="VOC2012")
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_2007, train_dataset_2012])
        val_dataset = VOCDataset(root_dir=args.data_path, transform=val_transform, target_transform=target_transform,
                                 is_training=False, split="VOC2007")
        num_classes = len(VOC_CLASS_LIST)
    elif args.dataset == 'coco':
        from data_loader.detection.coco import COCOObjectDetection, COCO_CLASS_LIST
        train_dataset = COCOObjectDetection(root_dir=args.data_path, transform=train_transform,
                                            target_transform=target_transform, is_training=True)
        val_dataset = COCOObjectDetection(root_dir=args.data_path, transform=val_transform, target_transform=target_transform, is_training=False)
        num_classes = len(COCO_CLASS_LIST)
    else:
        print_error_message('{} dataset is not supported yet'.format(args.dataset))
        exit()
    cfg.NUM_CLASSES = num_classes

    # -----------------------------------------------------------------------------
    # Dataset loader
    # -----------------------------------------------------------------------------
    print_info_message('Training samples: {}'.format(len(train_dataset)))
    print_info_message('Validation samples: {}'.format(len(val_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model = ssd(args, cfg)
    if args.finetune:
        if os.path.isfile(args.finetune):
            print_info_message('Loading weights for finetuning from {}'.format(args.finetune))
            weight_dict = torch.load(args.finetune, map_location=torch.device(device='cpu'))
            model.load_state_dict(weight_dict)
            print_info_message('Done')
        else:
            print_warning_message('No file for finetuning. Please check.')
    # -----------------------------------------------------------------------------
    # Optimizer and Criterion
    # -----------------------------------------------------------------------------
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    criterion = MultiBoxLoss(neg_pos_ratio=cfg.neg_pos_ratio)

    # writer for logs
    writer = SummaryWriter(log_dir=args.save, comment='Training and Validation logs')
    try:
        writer.add_graph(model, input_to_model=torch.Tensor(1, 3, cfg.image_size, cfg.image_size))
    except:
        print_log_message("Not able to generate the graph. Likely because your model is not supported by ONNX")

    #model stats
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, cfg.image_size, cfg.image_size))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(cfg.image_size, cfg.image_size, flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'

    min_val_loss = float('inf')
    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print_info_message("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            min_val_loss = checkpoint['min_loss']
            start_epoch = checkpoint['epoch']
        else:
            print_warning_message("=> no checkpoint found at '{}'".format(args.resume))

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # -----------------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------------
    if args.lr_type == 'poly':
        from utilities.lr_scheduler import PolyLR
        lr_scheduler = PolyLR(base_lr=args.lr, max_epochs=args.epochs, power=args.power)
    elif args.lr_type == 'hybrid':
        from utilities.lr_scheduler import HybirdLR
        lr_scheduler = HybirdLR(base_lr=args.lr, max_epochs=args.epochs, clr_max=args.clr_max, cycle_len=args.cycle_len)
    elif args.lr_type == 'clr':
        from utilities.lr_scheduler import CyclicLR
        lr_scheduler = CyclicLR(min_lr=args.lr, cycle_len=args.cycle_len, steps=args.steps, gamma=args.gamma, step=True)
    elif args.lr_type == 'cosine':
        from utilities.lr_scheduler import CosineLR
        lr_scheduler = CosineLR(base_lr=args.lr, max_epochs=args.epochs)
    else:
        print_error_message('{} scheduler not yet supported'.format(args.lr_type))
        exit()

    print_info_message(lr_scheduler)

    # -----------------------------------------------------------------------------
    # Training and validation loop
    # -----------------------------------------------------------------------------

    validate(val_loader, model, criterion, device, epoch=-1)
    extra_info_ckpt = '{}_{}'.format(args.model, args.s)
    for epoch in range(start_epoch, args.epochs):
        curr_lr = lr_scheduler.step(epoch)
        optimizer.param_groups[0]['lr'] = curr_lr

        print_info_message('Running epoch {} at LR {}'.format(epoch, curr_lr))
        train_loss, train_cl_loss, train_loc_loss = train(train_loader, model, criterion, optimizer, device, epoch=epoch)
        val_loss, val_cl_loss, val_loc_loss = validate(val_loader, model, criterion, device, epoch=epoch)
        # Save checkpoint
        is_best = val_loss < min_val_loss
        min_val_loss = min(val_loss, min_val_loss)

        weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': weights_dict,
            'min_loss': min_val_loss
        }, is_best, args.save, extra_info_ckpt)

        writer.add_scalar('Detection/LR/learning_rate', round(curr_lr, 6), epoch)
        writer.add_scalar('Detection/Loss/train', train_loss, epoch)
        writer.add_scalar('Detection/Loss/val', val_loss, epoch)
        writer.add_scalar('Detection/Loss/train_cls', train_cl_loss, epoch)
        writer.add_scalar('Detection/Loss/val_cls', val_cl_loss, epoch)
        writer.add_scalar('Detection/Loss/train_loc', train_loc_loss, epoch)
        writer.add_scalar('Detection/Loss/val_loc', val_loc_loss, epoch)
        writer.add_scalar('Detection/Complexity/Flops', min_val_loss, math.ceil(flops))
        writer.add_scalar('Detection/Complexity/Params', min_val_loss, math.ceil(num_params))

    writer.close()


if __name__ == '__main__':
    from commons.general_details import detection_datasets, detection_models, detection_schedulers

    parser = argparse.ArgumentParser(description='Training detection network')
    ### MODEL RELATED PARAMS
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='espnetv2', choices=detection_models, type=str, help='initialized model path')
    parser.add_argument('--s', default=2.0, type=float, help='Model scale factor')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    # dimension wise network related params
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')

    ### General configuration such as dataset path, etc
    parser.add_argument('--save', default='results_detection', type=str, help='results path')
    parser.add_argument('--dataset', default='pascal', choices=detection_datasets, help='Name of the dataset')
    parser.add_argument('--data-path', default='', help='Dataset path')

    #### OPTIMIZER related settings
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--power', default=0.9, type=float, help='Power for Polynomial LR')
    parser.add_argument('--lr-type', default='clr', type=str, choices=detection_schedulers, help='LR scheduler')
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--lr-mult', default=1, type=int, help='Factor by which base lr should be increased')
    # Hybrid LR hyperparameters
    parser.add_argument('--clr-max', default=160, type=int, help='Max CLR epochs (only for hybrid)')
    parser.add_argument('--cycle-len', default=5, type=int, help='Cycle length for CLR')
    # CLR/Multi-step LR related hyper-parameters
    parser.add_argument('--steps', default=[51, 161, 201], type=list,
                        help='steps at which lr should be decreased. Only used for Cyclic and Fixed LR')

    # general training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for laoding data')
    parser.add_argument('--epochs', default=240, type=int, help='Max number of epochs')
    parser.add_argument('--weights', default='', type=str, help='Location of pretrained weights')
    parser.add_argument('--im-size', default=300, type=int, help='Image size for training')
    # finetune the model
    parser.add_argument('--finetune', default='', type=str, help='finetune')

    args = parser.parse_args()

    if not args.weights:
        print('Loading weights using the weight dictionary')
        from model.weight_locations.classification import model_weight_map
        weight_file_key = '{}_{}'.format(args.model, args.s)
        assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
        args.weights = model_weight_map[weight_file_key]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.save = '{}/model_{}_{}/s_{}_sch_{}_im_{}/{}/'.format(args.save, args.model, args.dataset, args.s,
                                                                args.lr_type, args.im_size, timestr)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    main(args)
