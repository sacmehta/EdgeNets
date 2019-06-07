import argparse
import torch
from utilities.utils import model_parameters, compute_flops
from utilities.train_eval_classification import validate
import os
from data_loader.classification.imagenet import val_loader as loader
from utilities.print_utils import *
#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================


def main(args):
    # create model
    if args.model == 'dicenet':
        from model.classification import dicenet as net
        model = net.CNNModel(args)
    elif args.model == 'espnet':
        from model.classification import espnetv2 as net
        model = net.EESPNet(args)
    elif args.model == 'shufflenetv2':
        from model.classification import shufflenetv2 as net
        model = net.CNNModel(args)
    else:
        NotImplementedError('Model {} not yet implemented'.format(args.model))
        exit()

    num_params = model_parameters(model)
    flops = compute_flops(model)
    print_info_message('FLOPs: {:.2f} million'.format(flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))


    if not args.weights:
        print_info_message('Grabbing location of the ImageNet weights from the weight dictionary')
        from model.weight_locations.classification import model_weight_map

        weight_file_key = '{}_{}'.format(args.model, args.s)
        assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
        args.weights = model_weight_map[weight_file_key]

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >=1 else 'cpu'
    weight_dict = torch.load(args.weights, map_location=torch.device(device))
    model.load_state_dict(weight_dict)

    if num_gpus >= 1:
        args.data_parallel = True
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # Data loading code
    val_loader = loader(args)
    validate(val_loader, model, device=device)


if __name__ == '__main__':
    from commons.general_details import classification_models, classification_datasets

    parser = argparse.ArgumentParser(description='Testing efficient networks')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--data', default='', help='path to dataset')
    parser.add_argument('--dataset', default='imagenet', help='Name of the dataset', choices=classification_datasets)

    parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--num-classes', default=1000, type=int, help='# of classes in the dataset')
    parser.add_argument('--s', default=1, type=float, help='Factor by which output channels should be reduced (s > 1 for increasing the dims while < 1 for decreasing)')
    parser.add_argument('--weights', type=str, default='', help='weight file')
    parser.add_argument('--inpSize', default=224, type=int, help='Input size')
    ##Select a model
    parser.add_argument('--model', default='basic', choices=classification_models, help='Which model? basic= basic CNN model, res=resnet style, shuffle=shufflenetv2 style)')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')

    args = parser.parse_args()
    main(args)
