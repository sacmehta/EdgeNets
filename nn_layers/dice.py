#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch import nn
import torch.nn.functional as F
from nn_layers.cnn_utils import activation_fn, CBR, Shuffle, BR
import math


class DICE(nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True):
        '''
        :param channel_in: # of input channels
        :param channel_out: # of output channels
        :param height: Height of the input volume
        :param width: Width of the input volume
        :param kernel_size: Kernel size. We use the same kernel size of 3 for each dimension. Larger kernel size would increase the FLOPs and Parameters
        :param dilation: It's a list with 3 elements, each element corresponding to a dilation rate for each dimension.
        :param shuffle: Shuffle the feature maps in the volume-wise separable convolutions
        '''
        super().__init__()
        assert len(dilation) == 3
        padding_1 = int((kernel_size - 1) / 2) *dilation[0]
        padding_2 = int((kernel_size - 1) / 2) *dilation[1]
        padding_3 = int((kernel_size - 1) / 2) *dilation[2]
        self.conv_channel = nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, groups=channel_in,
                                      padding=padding_1, bias=False, dilation=dilation[0])
        self.conv_width = nn.Conv2d(width, width, kernel_size=kernel_size, stride=1, groups=width,
                               padding=padding_2, bias=False, dilation=dilation[1])
        self.conv_height = nn.Conv2d(height, height, kernel_size=kernel_size, stride=1, groups=height,
                               padding=padding_3, bias=False, dilation=dilation[2])

        self.br_act = BR(3*channel_in)
        self.weight_avg_layer = CBR(3*channel_in, channel_in, kSize=1, stride=1, groups=channel_in)

        # project from channel_in to Channel_out
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = CBR(channel_in, channel_out, kSize=3, stride=1, groups=groups_proj)
        self.linear_comb_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channel_in, channel_in // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in //4, channel_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.vol_shuffle = Shuffle(3)

        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.shuffle = shuffle
        self.ksize=kernel_size
        self.dilation = dilation

    def forward(self, x):
        '''
        :param x: input of dimension C x H x W
        :return: output of dimension C1 x H x W
        '''
        bsz, channels, height, width = x.size()
        # process across channel. Input: C x H x W, Output: C x H x W
        out_ch_wise = self.conv_channel(x)

        # process across height. Input: H x C x W, Output: C x H x W
        x_h_wise = x.clone()
        if height != self.height:
            if height < self.height:
                x_h_wise = F.interpolate(x_h_wise, mode='bilinear', size=(self.height, width), align_corners=True)
            else:
                x_h_wise = F.adaptive_avg_pool2d(x_h_wise, output_size=(self.height, width))

        x_h_wise = x_h_wise.transpose(1, 2).contiguous()
        out_h_wise = self.conv_height(x_h_wise).transpose(1, 2).contiguous()

        h_wise_height = out_h_wise.size(2)
        if height != h_wise_height:
            if h_wise_height < height:
                out_h_wise = F.interpolate(out_h_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_h_wise = F.adaptive_avg_pool2d(out_h_wise, output_size=(height, width))

        # process across width: Input: W x H x C, Output: C x H x W
        x_w_wise = x.clone()
        if width != self.width:
            if width < self.width:
                x_w_wise = F.interpolate(x_w_wise, mode='bilinear', size=(height, self.width), align_corners=True)
            else:
                x_w_wise = F.adaptive_avg_pool2d(x_w_wise, output_size=(height, self.width))

        x_w_wise = x_w_wise.transpose(1, 3).contiguous()
        out_w_wise = self.conv_width(x_w_wise).transpose(1, 3).contiguous()
        w_wise_width = out_w_wise.size(3)
        if width != w_wise_width:
            if w_wise_width < width:
                out_w_wise = F.interpolate(out_w_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_w_wise = F.adaptive_avg_pool2d(out_w_wise, output_size=(height, width))

        # Merge. Output will be 3C x H X W
        outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise), 1)
        outputs = self.br_act(outputs)
        if self.shuffle:
            outputs = self.vol_shuffle(outputs)
        outputs = self.weight_avg_layer(outputs)
        linear_wts = self.linear_comb_layer(outputs)
        proj_out = self.proj_layer(outputs)
        return proj_out * linear_wts

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, vol_shuffle={shuffle}, ' \
            'width={width}, height={height}, dilation={dilation})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class StridedDICE(nn.Module):
    '''
    This class implements the strided volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, height, width, kernel_size=3, dilation=[1,1,1], shuffle=True):
        '''
        :param channel_in: # of input channels
        :param channel_out: # of output channels
        :param height: Height of the input volume
        :param width: Width of the input volume
        :param kernel_size: Kernel size. We use the same kernel size of 3 for each dimension. Larger kernel size would increase the FLOPs and Parameters
        :param dilation: It's a list with 3 elements, each element corresponding to a dilation rate for each dimension.
        :param shuffle: Shuffle the feature maps in the volume-wise separable convolutions
        '''
        super().__init__()
        assert len(dilation) == 3

        self.left_layer = nn.Sequential(CBR(channel_in, channel_in, 3, stride=2, groups=channel_in),
                                        CBR(channel_in, channel_in, 1, 1)
                                        )
        self.right_layer =  nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            DICE(channel_in, channel_in, height, width, kernel_size=kernel_size, dilation=dilation,
                 shuffle=shuffle),
            CBR(channel_in, channel_in, 1, 1)
        )
        self.shuffle = Shuffle(groups=2)

        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = 2*channel_in
        self.ksize = kernel_size

    def forward(self, x):
        x_left = self.left_layer(x)
        x_right = self.right_layer(x)
        concat = torch.cat([x_left, x_right], 1)
        return self.shuffle(concat)

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, ' \
            'width={width}, height={height})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

if __name__ == '__main__':
    import numpy as np
    channel_in = 3
    channel_out = 30
    width = 112
    height = 112
    bsz = 2
    input = torch.Tensor(bsz, channel_in, height, width).fill_(1)
    model = DICE(channel_in, channel_out, width, height, shuffle=True)
    model.eval()

    input = torch.Tensor(bsz, channel_in, 56, 56).fill_(1)
    out = model(input)

    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Params: {}'.format(n_params))
