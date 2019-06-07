# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

from torch.nn import init
import torch.nn.functional as F
import torch
from torch import nn
from model.classification.dicenet import CNNModel


class SSDNet300(nn.Module):

    def __init__(self, args, extra_layer):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super(SSDNet300, self).__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================

        self.basenet = CNNModel(args)

        # delete the classification layer
        del self.basenet.classifier

        # retrive the basenet configuration
        base_net_config = self.basenet.out_channel_map
        config = base_net_config[:4] + [base_net_config[5]]

        # add configuration for SSD version
        config += [1024, 512, 256]

        # =============================================================
        #                EXTRA LAYERS for DETECTION
        # =============================================================

        self.extra_level6 = extra_layer(config[4], config[5])
        self.extra_level7 = extra_layer(config[5], config[6])

        self.extra_level8 = nn.Sequential(
            nn.Conv2d(config[6], config[6], kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(config[6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(config[6], config[7], kernel_size=2, stride=2, bias=False),
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
                                           out_planes=out_features)

        in_features = config[4] + config[5]
        out_features = config[4]
        self.bu_5x5_10x10 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                           out_planes=out_features)

        in_features = config[4] + config[3]
        out_features = config[3]
        self.bu_10x10_19x19 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                             out_planes=out_features)

        in_features = config[3] + config[2]
        out_features = config[2]
        self.bu_19x19_38x38 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                               out_planes=out_features)

        self.config = config

    def up_sample(self, x, size):
        return F.interpolate(x, size=(size[2], size[3]), align_corners=True, mode='bilinear')

    def forward(self, x, is_train=True):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        out_150x150 = self.basenet.level1(x)
        out_75x75 = self.basenet.level2(out_150x150)
        out_38x38 = self.basenet.level3(out_75x75)
        out_19x19 = self.basenet.level4(out_38x38)
        out_10x10 = self.basenet.level5(out_19x19)

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



class SSDNet512(nn.Module):

    def __init__(self, args, extra_layer):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super(SSDNet512, self).__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================

        self.basenet = CNNModel(args)

        # delete the classification layer
        del self.basenet.classifier

        # retrive the basenet configuration
        base_net_config = self.basenet.out_channel_map
        config = base_net_config[:4] + [base_net_config[5]]

        # add configuration for SSD version
        config += [1024, 512, 256, 128]

        # =============================================================
        #                EXTRA LAYERS for DETECTION
        # =============================================================

        self.extra_level6 = extra_layer(config[4], config[5])
        self.extra_level7 = extra_layer(config[5], config[6])
        self.extra_level8 = extra_layer(config[6], config[7])

        self.extra_level9 = nn.Sequential(
            nn.Conv2d(config[7], config[8], kernel_size=3, stride=2, bias=False, padding=1),
            nn.ReLU(inplace=True)
        )

        # =============================================================
        #                EXTRA LAYERS for Bottom-up decoding
        # =============================================================

        from nn_layers.efficient_pyramid_pool import EfficientPyrPool

        in_features = config[5] + config[6]
        out_features = config[5]
        red_factor = 5
        self.bu_4x4_8x8 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                           out_planes=out_features)

        in_features = config[4] + config[5]
        out_features = config[4]
        self.bu_8x8_16x16 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                             out_planes=out_features)

        in_features = config[4] + config[3]
        out_features = config[3]
        self.bu_16x16_32x32 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                               out_planes=out_features)

        in_features = config[3] + config[2]
        out_features = config[2]
        self.bu_32x32_64x64 = EfficientPyrPool(in_planes=in_features, proj_planes=out_features // red_factor,
                                               out_planes=out_features)

        self.config = config

    def up_sample(self, x, size):
        return F.interpolate(x, size=(size[2], size[3]), align_corners=True, mode='bilinear')

    def forward(self, x, is_train=True):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        out_256x256 = self.basenet.level1(x)
        out_128x128 = self.basenet.level2(out_256x256)
        out_64x64 = self.basenet.level3(out_128x128)
        out_32x32 = self.basenet.level4(out_64x64)
        out_16x16 = self.basenet.level5(out_32x32)

        # Detection network's extra layers
        out_8x8 = self.extra_level6(out_16x16)
        out_4x4 = self.extra_level7(out_8x8)
        out_2x2 = self.extra_level8(out_4x4)
        out_1x1 = self.extra_level9(out_2x2)

        # bottom-up decoding
        ## 3x3 and 5x5
        out_4x4_8x8 = self.up_sample(out_4x4, out_8x8.size())
        out_4x4_8x8 = torch.cat((out_4x4_8x8, out_8x8), dim=1)
        out_8x8_epp = self.bu_4x4_8x8(out_4x4_8x8)

        ## 5x5 and 10x10
        out_8x8_16x16 = self.up_sample(out_8x8_epp, out_16x16.size())
        out_8x8_16x16 = torch.cat((out_8x8_16x16, out_16x16), dim=1)
        out_16x16_epp = self.bu_8x8_16x16(out_8x8_16x16)

        ## 10x10 and 19x19
        out_16x16_32x32 = self.up_sample(out_16x16_epp, out_32x32.size())
        out_16x16_32x32 = torch.cat((out_16x16_32x32, out_32x32), dim=1)
        out_32x32_epp = self.bu_16x16_32x32(out_16x16_32x32)

        ## 19x19 and 38x38
        out_32x32_64x64 = self.up_sample(out_32x32_epp, out_64x64.size())
        out_32x32_64x64 = torch.cat((out_32x32_64x64, out_64x64), dim=1)
        out_64x64_epp = self.bu_32x32_64x64(out_32x32_64x64)

        return out_64x64_epp, out_32x32_epp, out_16x16_epp, out_8x8_epp, out_4x4, out_2x2, out_1x1
