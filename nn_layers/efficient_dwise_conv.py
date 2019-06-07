#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch import nn
import torch.nn.functional as F
from nn_layers.cnn_utils import activation_fn, CBR, Shuffle, BR
import math


class EffDWSepConv(nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super().__init__()
        self.conv_channel = CBR(channel_in, channel_in, kSize=kernel_size, stride=1, groups=channel_in)

        # project from channel_in to Channel_out
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = CBR(channel_in, channel_out, kSize=3, stride=1, groups=groups_proj)

        self.linear_comb_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.ksize=kernel_size

    def forward(self, x):
        '''
        :param x: input of dimension C x H x W
        :return: output of dimension C1 x H x W
        '''
        bsz, channels, height, width = x.size()
        x = self.conv_channel(x)
        proj_out  =self.proj_layer(x)
        linear_comb_out = self.linear_comb_layer(x)
        return proj_out * linear_comb_out

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class StridedEffDWise(nn.Module):
    '''
    This class implements the strided volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, kernel_size=3):
        '''
        :param channel_in: # of input channels
        :param channel_out: # of output channels
        :param height: Height of the input volume
        :param width: Width of the input volume
        :param kernel_size: Kernel size. We use the same kernel size of 3 for each dimension. Larger kernel size would increase the FLOPs and Parameters
        :param dilation: It's a list with 3 elements, each element corresponding to a dilation rate for each dimension.
        :param shuffle: Shuffle the feature maps in the volume-wise separable convolutions
        :param weight_avg: Waighted average for fusing the feature maps in volume-wise separable convolutions
        :param res_conn: Residual connection in the volume-wise separabel convolutions
        :param proj: Want to project the feature maps from channel_in to channel_out or not
        '''
        super().__init__()

        self.pool_layer = CBR(channel_in, channel_in, 3, stride=2, groups=channel_in)
        self.dw_layer =  EffDWSepConv(channel_in, channel_in, kernel_size=kernel_size)
        self.channel_in = channel_in
        self.channel_out = 2*channel_in
        self.ksize = kernel_size

    def forward(self, x):
        x = self.pool_layer(x)
        return torch.cat([x, self.dw_layer(x)], 1)

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
    input = torch.Tensor(bsz, channel_in, height, width)._fill_(1)
    model = EffDWSepConv(channel_in, channel_out)
    model.eval()

    input = torch.Tensor(bsz, channel_in, 56, 56)._fill_(1)
    out = model(input)

    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Params: {}'.format(n_params))
