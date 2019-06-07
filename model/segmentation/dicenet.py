# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch.nn import init
from nn_layers.espnet_utils import *
from nn_layers.efficient_pyramid_pool import EfficientPyrPool
from nn_layers.efficient_pt import EfficientPWConv
from model.classification.dicenet import CNNModel
from utilities.print_utils import *
from torch.nn import functional as F


class DiCENetSegmentation(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    '''

    def __init__(self, args, classes=21, dataset='pascal'):
        super().__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================
        self.base_net = CNNModel(args) #imagenet model
        del self.base_net.classifier
        del self.base_net.level5

        config = self.base_net.out_channel_map

        #=============================================================
        #                   SEGMENTATION NETWORK
        #=============================================================
        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'coco': 32
        }
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        pyr_plane_proj = min(classes //2, base_dec_planes)

        # dimensions in variable names are shown for an input of 256x256
        self.eff_pool_16x16 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj,
                                               out_planes=dec_planes[0])
        self.eff_pool_32x32 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                               out_planes=dec_planes[1])
        self.eff_pool_64x64 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj,
                                               out_planes=dec_planes[2])
        self.eff_pool_128x128 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                                 out_planes=dec_planes[3], last_layer_br=False)

        self.proj_enc_32x32 = EfficientPWConv(config[2], dec_planes[0])
        self.proj_enc_64x64 = EfficientPWConv(config[1], dec_planes[1])
        self.proj_enc_128x128 = EfficientPWConv(config[0], dec_planes[2])

        self.bu_br_32x32 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                         nn.PReLU(dec_planes[0])
                                         )
        self.bu_br_64x64 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]),
                                         nn.PReLU(dec_planes[1])
                                         )
        self.bu_br_128x128 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]),
                                           nn.PReLU(dec_planes[2])
                                           )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_basenet_params(self):
        modules_base = [self.base_net]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.eff_pool_16x16, self.eff_pool_32x32, self.eff_pool_64x64, self.eff_pool_128x128,
                       self.proj_enc_128x128, self.proj_enc_64x64, self.proj_enc_32x32,
                       self.bu_br_128x128, self.bu_br_64x64, self.bu_br_32x32]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        x_size = (x.size(2), x.size(3))
        # dimensions/names are shown for an input of 256x256
        enc_out_128x128 = self.base_net.level1(x)
        enc_out_64x64 = self.base_net.level2(enc_out_128x128)
        enc_out_32x32 = self.base_net.level3(enc_out_64x64)
        enc_out_16x16 = self.base_net.level4(enc_out_32x32)

        # bottom-up decoding
        bu_out_16x16 = self.eff_pool_16x16(enc_out_16x16)

        # Decoding block
        bu_out_32x32 = self.upsample(bu_out_16x16)
        # project encoder
        enc_out_32x32 = self.proj_enc_32x32(enc_out_32x32)
        # merge encoder and decoder with identity connection
        bu_out_32x32 = enc_out_32x32 + bu_out_32x32
        # normalize
        bu_out_32x32 = self.bu_br_32x32(bu_out_32x32)
        # compute pyramid features
        bu_out_32x32 = self.eff_pool_32x32(bu_out_32x32)

        #decoding block
        bu_out_64x64 = self.upsample(bu_out_32x32)
        # project encoder
        enc_out_64x64 = self.proj_enc_64x64(enc_out_64x64)
        # merge encoder and decoder with identity connection
        bu_out_64x64 = enc_out_64x64 + bu_out_64x64
        # normalize
        bu_out_64x64 = self.bu_br_64x64(bu_out_64x64)
        # compute pyramid features
        bu_out_64x64 = self.eff_pool_64x64(bu_out_64x64)

        # decoding block
        bu_out_128x128 = self.upsample(bu_out_64x64)
        #project encoder
        enc_out_128x128 = self.proj_enc_128x128(enc_out_128x128)
        # merge encoder and decoder with identity connection
        bu_out_128x128 = enc_out_128x128 + bu_out_128x128
        #normalize
        bu_out_128x128 = self.bu_br_128x128(bu_out_128x128)
        # compute pyramid features
        bu_out_128x128  = self.eff_pool_128x128(bu_out_128x128)

        #upsample to the same size as the input
        return F.interpolate(bu_out_128x128, size=x_size, mode='bilinear', align_corners=True) #nn.Upsample(x_size, mode='bilinear', align_corners=True)(bu_out_128x128)


def dicenet_seg(args, classes):
    weights = args.weights
    model = DiCENetSegmentation(args, classes=classes)
    if weights:
        import os
        if os.path.isfile(weights):
            num_gpus = torch.cuda.device_count()
            device = 'cuda' if num_gpus >= 1 else 'cpu'
            pretrained_dict = torch.load(weights, map_location=torch.device(device))
        else:
            print_error_message('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
            exit()
        print_log_message('Loading pretrained basenet model weights')
        basenet_dict = model.base_net.state_dict()
        model_dict = model.state_dict()
        overlap_dict = {k: v for k, v in pretrained_dict.items() if k in basenet_dict}
        if len(overlap_dict) == 0:
            print_error_message('No overlaping weights between model file and pretrained weight file. Please check')
            #exit()
        print_log_message('{:.2f} % of weights copied from basenet to segnet'.format(len(overlap_dict) * 1.0/len(model_dict) * 100))
        basenet_dict.update(overlap_dict)
        model.base_net.load_state_dict(basenet_dict)
        print_log_message('Pretrained basenet model loaded!!')
    else:
        print_warning_message('Training from scratch!!')
    return model

if __name__ == "__main__":

    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()
    from model.weight_locations import segmentation

    args.num_classes = 1000
    args.weights = ''
    args.dataset = 'pascal'
    args.channels = 3
    args.model_width = 224
    args.model_height = 224

    for scale in segmentation.dicenet_scales:
        args.s = scale
        weight_map_dict = segmentation.model_weight_map['dicenet_{}'.format(scale)]
        if args.dataset == 'pascal':
            for size in [256, 384]:
                model = dicenet_seg(args, classes=21)
                input = torch.Tensor(1, 3, size, size)
                print_info_message('Scale: {}, Input: {}, FLOPs: {}, Params: {}'.format(scale, size,
                                                                                        compute_flops(model, input=input),
                                                                                        model_parameters(model)))


        #for size in [224]:
        #input = torch.Tensor(1, 3, 256, 256)
        #model = dicenet_seg(args, classes=21)
        #from utilities.utils import compute_flops, model_parameters
        #print(compute_flops(model, input=input))
        #print(model_parameters(model))
        #out = model(input)
        #print(out.size())

