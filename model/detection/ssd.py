# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch import nn
from utilities.print_utils import *
from nn_layers.cnn_utils import CBR
from model.detection.espnetv2 import ESPNetv2SSD


class SSDExtraLayers(nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(SSDExtraLayers, self).__init__()
        self.layer = nn.Sequential(
            CBR(nin, nin, stride=2, kSize=ksize, groups=nin),
            CBR(nin, nout, kSize=1)
        )

    def forward(self, x):
        return self.layer(x)

class SSD300(nn.Module):
    def __init__(self, args, cfg):
        super(SSD300, self).__init__()
        if args.model == 'espnetv2':
            self.base_net = ESPNetv2SSD(args, extra_layer=SSDExtraLayers)
        elif args.model == 'dicenet':
            from model.detection.dicenet import SSDNet300
            self.base_net = SSDNet300(args, extra_layer=SSDExtraLayers)
        else:
            print_error_message('{} model not yet supported'.format(args.model))

        self.num_classes = cfg.NUM_CLASSES

        self.in_channels = self.base_net.config[-6:]
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        num_anchors = cfg.box_per_location

        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], num_anchors[i] * 4, kernel_size=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], num_anchors[i] * self.num_classes, kernel_size=1)]
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        print_info_message('Initializaing Conv Layers with Xavier Unifrom')
        # initializing matters a lot
        # changing to Kaiming He's init functionaity, does not let the model to converge.
        # probably because, smooth function struggles with that initialization.
        # XAVIER Unifrom Rocks here
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        fms = self.base_net(x)
        for i, fm in enumerate(fms):
            loc_pred = self.loc_layers[i](fm)
            cls_pred = self.cls_layers[i](fm)

            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(loc_pred.size(0), -1,4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]

            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(cls_pred.size(0), -1,self.num_classes)  # [N,9*NC,H,W] -> [N,H,W,9*NC] -> [N,H*W*9,NC]

            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        locations = torch.cat(loc_preds, 1)
        confidences = torch.cat(cls_preds, 1)
        return confidences, locations


class SSD512(nn.Module):
    def __init__(self, args, cfg):
        super(SSD512, self).__init__()
        if args.model == 'espnetv2':
            self.base_net = ESPNetv2SSD(args, extra_layer=SSDExtraLayers)
        elif args.model == 'dicenet':
            from model.detection.dicenet import SSDNet512
            self.base_net = SSDNet512(args, extra_layer=SSDExtraLayers)
        else:
            print_error_message('{} model not yet supported'.format(args.model))

        self.num_classes = cfg.NUM_CLASSES

        self.in_channels = self.base_net.config[-6:]
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        num_anchors = cfg.box_per_location

        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], num_anchors[i] * 4, kernel_size=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], num_anchors[i] * self.num_classes, kernel_size=1)]

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        print_info_message('Initializaing Conv Layers with Xavier Unifrom')
        # initializing matters a lot
        # changing to Kaiming He's init functionaity, does not let the model to converge.
        # probably because, smooth function struggles with that initialization.
        # XAVIER Unifrom Rocks here
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        fms = self.base_net(x)
        for i, fm in enumerate(fms):
            loc_pred = self.loc_layers[i](fm)
            cls_pred = self.cls_layers[i](fm)

            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(loc_pred.size(0), -1,4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]

            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(cls_pred.size(0), -1,self.num_classes)  # [N,9*NC,H,W] -> [N,H,W,9*NC] -> [N,H*W*9,NC]

            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        locations = torch.cat(loc_preds, 1)
        confidences = torch.cat(cls_preds, 1)
        return confidences, locations


def ssd(config, cfg, *args, **kwargs):
    weights = config.weights

    if config.im_size == 512:
        model = SSD512(config, cfg)
    elif config.im_size == 300:
        model = SSD300(config, cfg)
    else:
        print_error_message('{} image size not supported'.format(config.im_size))
    if weights:
        import os
        if not os.path.isfile(weights):
            print_error_message('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
            exit(-1)
        num_gpus = torch.cuda.device_count()
        device = 'cuda' if num_gpus >= 1 else 'cpu'
        pretrained_dict = torch.load(weights, map_location=torch.device(device))
        print_info_message('Loading pretrained base model weights')
        basenet_dict = model.base_net.basenet.state_dict()
        model_dict = model.state_dict()
        overlap_dict = {k: v for k, v in pretrained_dict.items() if k in basenet_dict}
        if len(overlap_dict) == 0:
            print_error_message('No overlaping weights between model file and pretrained weight file. Please check')
            exit()
        print_info_message(
            '{:.2f} % of basenet weights copied to detectnet'.format(len(overlap_dict) * 1.0 / len(model_dict) * 100))
        basenet_dict.update(overlap_dict)
        model.base_net.basenet.load_state_dict(basenet_dict)
        print_info_message('Pretrained base model loaded!!')
    else:
        print_warning_message('Training from scratch!!. If you are testing, ignore this message.'
                              ' For testing, we do not load weights here.')
    return model


if __name__ == "__main__":
    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()
    args.s = 2.0
    args.channels = 3
    args.model_width = 224
    args.model_height = 224
    args.model = 'espnetv2'
    args.weights = ''
    args.im_size = 300

    if args.im_size == 512:
        from model.detection.ssd_config import SSD512Configuration as cfg
        cfg.NUM_CLASSES = 81
    elif args.im_size == 300:
        from model.detection.ssd_config import SSD300Configuration as cfg
        cfg.NUM_CLASSES = 81
    else:
        print_error_message('not supported')
    inputs = torch.randn(1, 3, args.im_size, args.im_size)
    net = ssd(args, cfg)
    loc_preds, cls_preds = net(inputs)

    print_info_message(compute_flops(net, input=inputs))
    print_info_message(model_parameters(net))
