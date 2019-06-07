#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch.nn import init
from torch import nn
from nn_layers.cnn_utils import CBR, Shuffle
from nn_layers.dice import DICE, StridedDICE
from model.classification import dicenet_config as config_all
from utilities.print_utils import *


class ShuffleDICEBlock(nn.Module):
    def __init__(self, inplanes, outplanes, height, width, c_tag=0.5, groups=2):
        super(ShuffleDICEBlock, self).__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part

        self.layer_right = nn.Sequential(
            CBR(self.right_part_in, self.right_part_out, 1, 1),
            DICE(channel_in=self.right_part_out, channel_out=self.right_part_out, height=height, width=width)
        )

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.groups = groups
        self.shuffle = Shuffle(groups=2)

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]

        right = self.layer_right(right)

        return self.shuffle(torch.cat((left, right), 1))

    def __repr__(self):
        s = '{name}(in_channels={inplanes}, out_channels={outplanes})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()

        # ====================
        # Network configuraiton
        # ====================
        try:
            num_classes = args.num_classes
        except:
            # if not specified, default to 1000 for imageNet
            num_classes = 1000 # 1000 for imagenet

        try:
            channels_in = args.channels
        except:
            # if not specified, default to RGB (3)
            channels_in = 3

        width = args.model_width
        height = args.model_height
        s = args.s
        if not s in config_all.sc_ch_dict.keys():
            print_error_message('Model at scale s={} is not suppoerted yet'.format(s))
            exit(-1)

        out_channel_map = config_all.sc_ch_dict[s]
        reps_at_each_level = config_all.rep_layers

        assert width % 32 == 0, 'Input image width should be divisible by 32'
        assert height % 32 == 0, 'Input image height should be divisible by 32'

        # ====================
        # Network architecture
        # ====================

        # output size will be 112 x 112
        width = int(width / 2)
        height = int(height / 2)
        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)
        width = int(width / 2)
        height = int(height / 2)
        self.level2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # output size will be 28 x 28
        width = int(width / 2)
        height = int(height / 2)
        level3 = nn.ModuleList()
        level3.append(StridedDICE(channel_in=out_channel_map[1], height=height, width=width))
        for i in range(reps_at_each_level[1]):
            if i == 0:
                level3.append(ShuffleDICEBlock(2 * out_channel_map[1], out_channel_map[2], width=width, height=height))
            else:
                level3.append(ShuffleDICEBlock(out_channel_map[2], out_channel_map[2], width=width, height=height))
        self.level3 = nn.Sequential(*level3)

        # output size will be 14 x 14
        level4 = nn.ModuleList()
        width = int(width / 2)
        height = int(height / 2)
        level4.append(StridedDICE(channel_in=out_channel_map[2], width=width, height=height))
        for i in range(reps_at_each_level[2]):
            if i == 0:
                level4.append(ShuffleDICEBlock(2 * out_channel_map[2], out_channel_map[3], width=width, height=height))
            else:
                level4.append(ShuffleDICEBlock(out_channel_map[3], out_channel_map[3], width=width, height=height))
        self.level4 = nn.Sequential(*level4)

        # output size will be 7 x 7
        level5 = nn.ModuleList()
        width = int(width / 2)
        height = int(height / 2)
        level5.append(StridedDICE(channel_in=out_channel_map[3], width=width, height=height))
        for i in range(reps_at_each_level[3]):
            if i == 0:
                level5.append(ShuffleDICEBlock(2 * out_channel_map[3], out_channel_map[4], width=width, height=height))
            else:
                level5.append(ShuffleDICEBlock(out_channel_map[4], out_channel_map[4], width=width, height=height))
        self.level5 = nn.Sequential(*level5)

        # classification layer
        if s > 1:
            self.drop_layer = nn.Dropout(p=0.2)
        else:
            self.drop_layer = nn.Dropout(p=0.1)

        # We use four groups in Grouped linear transformation
        # introduced in Pyramidal Recurrent Unit for Language Modeling
        # https://arxiv.org/abs/1808.09029
        groups = 4

        self.classifier = nn.Sequential(
            nn.Conv2d(out_channel_map[4], out_channel_map[5], kernel_size=1, groups=groups, bias=False),
            self.drop_layer,
            nn.Conv2d(out_channel_map[5], num_classes, 1, padding=0, bias=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.out_channel_map = out_channel_map

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    def forward(self, x):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        x = self.level1(x)  # 112
        x = self.level2(x)  # 56
        x = self.level3(x) # 28
        x = self.level4(x) # 14
        x = self.level5(x) # 7
        x = self.global_pool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == '__main__':

    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse
    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    for scale in config_all.sc_ch_dict.keys():
        for size in [224]:
            #args.num_classes = 1000
            imSz = size
            args.s = scale
            args.channels = 3
            args.model_width = 224
            args.model_height = 224

            model = CNNModel(args)
            input = torch.randn(1, 3, size, size)
            print_info_message('Scale: {}, ImSize: {}'.format(scale, size))
            print_info_message('Flops: {:.2f} million'.format(compute_flops(model, input)))
            print_info_message('Params: {:.2f} million'.format(model_parameters(model)))
            print('\n')
            #exit()
