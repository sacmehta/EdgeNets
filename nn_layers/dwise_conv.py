#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch import nn
from nn_layers.cnn_utils import CBR, CB

class Shuffle(nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class DWSepConv(nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out,  kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.dwise_layer = nn.Sequential(
                                CB(channel_in, channel_in, kernel_size, stride=stride, dilation=dilation, groups=channel_in),
                                CBR(channel_in, channel_out, 1, 1, groups=1)
                            )
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.ksize=kernel_size
        self.dilation = dilation

    def forward(self, x):
        return self.dwise_layer(x)

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, dilation={dilation})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class StridedDWise(nn.Module):
    def __init__(self, channel_in, kernel_size=3, dilation=1):
        super().__init__()
        self.pool_layer = CBR(channel_in, channel_in, 3, stride=2, groups=channel_in)
        self.dwise_layer = DWSepConv(channel_in, channel_in, kernel_size=kernel_size, dilation=dilation)
        self.channel_in = channel_in
        self.channel_out = 2*channel_in
        self.ksize = kernel_size

    def forward(self, x):
        x = self.pool_layer(x)
        return torch.cat([x, self.dwise_layer(x)], 1)

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
