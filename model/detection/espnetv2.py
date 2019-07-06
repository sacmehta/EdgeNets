# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

from torch.nn import init
import torch.nn.functional as F
import torch
from torch import nn
from model.classification.espnetv2 import EESPNet

class ESPNetv2SSD(nn.Module):

    def __init__(self, args, extra_layer):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super(ESPNetv2SSD, self).__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================

        self.basenet = EESPNet(args)
        # delete the classification layer
        del self.basenet.classifier
        # delte the last layer in level 5
        #del self.basenet.level5[4]
        #del self.basenet.level5[3]


        # retrive the basenet configuration
        base_net_config = self.basenet.config
        config = base_net_config[:4] + [base_net_config[5]]

        # add configuration for SSD version
        config += [512, 256, 128]

        # =============================================================
        #                EXTRA LAYERS for DETECTION
        # =============================================================

        self.extra_level6 = extra_layer(config[4], config[5]) #

        self.extra_level7 = extra_layer(config[5], config[6])

        self.extra_level8 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(config[6], config[7], kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # =============================================================
        #                EXTRA LAYERS for Bottom-up decoding
        # =============================================================

        from nn_layers.efficient_pyramid_pool import EfficientPyrPool

        in_features = config[5] + config[6]
        out_features = config[5]
        red_factor = 5
        self.bu_3x3_5x5 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                           out_planes=out_features, scales=[2.0, 1.0])

        in_features = config[4] + config[5]
        out_features = config[4]
        self.bu_5x5_10x10 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                           out_planes=out_features, scales=[2.0, 1.0, 0.5])

        in_features = config[4] + config[3]
        out_features = config[3]
        self.bu_10x10_19x19 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                             out_planes=out_features, scales=[2.0, 1.0, 0.5, 0.25])

        in_features = config[3] + config[2]
        out_features = config[2]
        self.bu_19x19_38x38 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                               out_planes=out_features, scales=[2.0, 1.0, 0.5, 0.25])

        self.config = config

    def up_sample(self, x, size):
        return F.interpolate(x, size=(size[2], size[3]), align_corners=True, mode='bilinear')

    def forward(self, x, is_train=True):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_150x150 = self.basenet.level1(x)  # 112
        if not self.basenet.input_reinforcement:
            del x
            x = None

        out_75x75 = self.basenet.level2_0(out_150x150, x)  # 56

        out_38x38 = self.basenet.level3_0(out_75x75, x)  # down-sample
        for i, layer in enumerate(self.basenet.level3):
            out_38x38 = layer(out_38x38)

        # Detection network
        out_19x19 = self.basenet.level4_0(out_38x38, x)  # down-sample
        for i, layer in enumerate(self.basenet.level4):
            out_19x19 = layer(out_19x19)

        out_10x10 = self.basenet.level5_0(out_19x19, x)  # down-sample
        for i, layer in enumerate(self.basenet.level5):
            out_10x10 = layer(out_10x10)

        # Detection network's extra layers
        out_5x5 = self.extra_level6(out_10x10)
        out_3x3 = self.extra_level7(out_5x5)
        out_1x1 = self.extra_level8(out_3x3)


        # bottom-up decoding
        ## 3x3 and 5x5
        out_3x3_5x5 = self.up_sample(out_3x3, out_5x5.size())
        out_3x3_5x5 = torch.cat((out_3x3_5x5, out_5x5), dim=1)
        out_5x5_epp = self.bu_3x3_5x5(out_3x3_5x5)

        ## 5x5 and 10x10
        out_5x5_10x10 = self.up_sample(out_5x5_epp, out_10x10.size())
        out_5x5_10x10 = torch.cat((out_5x5_10x10, out_10x10), dim=1)
        out_10x10_epp = self.bu_5x5_10x10(out_5x5_10x10)

        ## 10x10 and 19x19
        out_10x10_19x19 = self.up_sample(out_10x10_epp, out_19x19.size())
        out_10x10_19x19 = torch.cat((out_10x10_19x19, out_19x19), dim=1)
        out_19x19_epp = self.bu_10x10_19x19(out_10x10_19x19)

        ## 19x19 and 38x38
        out_19x19_38x38 = self.up_sample(out_19x19_epp, out_38x38.size())
        out_19x19_38x38 = torch.cat((out_19x19_38x38, out_38x38), dim=1)
        out_38x38_epp = self.bu_19x19_38x38(out_19x19_38x38)

        return out_38x38_epp, out_19x19_epp, out_10x10_epp, out_5x5_epp, out_3x3, out_1x1
