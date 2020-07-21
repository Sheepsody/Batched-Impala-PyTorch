import torch.nn as nn
import torch as torch
import numpy as np


def _output_size_conv2d(conv, size):
    """
    Computes the output size of the convolution for an input size
    """
    o_size = np.array(size) + 2 * np.array(conv.padding)
    o_size -= np.array(conv.dilation) * (np.array(conv.kernel_size) - 1)
    o_size -= 1
    o_size = o_size / np.array(conv.stride) + 1
    return np.floor(o_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()

        self.in_ch = in_ch

        self.conv1 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        out += residual
        return out


class BaseBlock(nn.Module):
    """
    Blocks for a residual model for reinforcement learning task as presented in He. and al, 2016
    """

    def __init__(self, in_ch, out_ch):
        super(BaseBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Conv2d(
            in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=3, stride=1
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.residual = ResidualBlock(in_ch=self.out_ch)

        self._body = nn.Sequential(self.conv, self.max_pool, self.residual)

    def forward(self, x):
        x = self._body(x)
        return x

    def output_size(self, size):
        size = _output_size_conv2d(self.conv, size)
        size = _output_size_conv2d(self.max_pool, size)
        return size


class DeepConv(nn.Module):
    """
    Deeper model that uses 12 convolutions with residual blocks
    """

    def __init__(self, c):
        """c is the number of channels in the input tensor"""
        super(DeepConv, self).__init__(self)

        self.block_1 = BaseBlock(c, 16)
        self.block_2 = BaseBlock(16, 32)
        self.block_3 = BaseBlock(32, 32)

        self._body = nn.Sequential(self.block_1, self.block_2, self.block_3)

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = self.block_1.output_size(size)
        size = self.block_2.output_size(size)
        size = self.block_3.output_size(size)
        return size


class ShallowConv(nn.Module):
    """
    Shallow model that uses only 3 convolutions
    """

    def __init__(self, c):
        super(ShallowConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        self._body = nn.Sequential(
            self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3, nn.ReLU()
        )

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = _output_size_conv2d(self.conv1, size)
        size = _output_size_conv2d(self.conv2, size)
        size = _output_size_conv2d(self.conv3, size)
        return size
