import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from abc import ABC, abstractmethod


def _output_size_conv2d(conv, size):
    """
    Computes the output size of the convolution for an input size
    """
    o_size = np.array(size) +  2*np.array(conv.padding)
    o_size -= np.array(conv.dilation) * (np.array(conv.kernel_size) - 1) - 1
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

        self.conv1 = nn.Conv2d(in_channels=self.in_ch,
                               out_channels=self.in_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_ch,
                               out_channels=self.in_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1)
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

        self.conv = nn.Conv2d(in_channels=self.in_ch,
                              out_channels=self.out_ch,
                              kernel_size=3,
                              stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, 
                                     stride=2)
        self.residual = ResidualBlock(in_ch=self.out_ch)

        self._body = nn.Sequential(self.conv, self.max_pool, self.residual)

    def forward(self, x):
        x = self._body(x)
        return x

    def output_size(self, size):
        size = _output_size_conv2d(self.conv, size)
        size = _output_size_conv2d(self.max_pool, size)
        return h, w



class DeepConv(nn.Module):
    """
    Deeper model that uses 12 convolutions with residual blocks
    """
    def __init__(self, c):
        """c is the number of channels in the input tensor"""
        super(DeepConv, self).__init__(self)

        self.block_1 = Baseblock(c, 16)
        self.block_2 = Baseblock(16, 32)
        self.block_3 = Baseblock(32, 32)

        self._body = nn.Sequential(block_1, block_2, block_3)
    
    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = self.block1.output_size(size)
        size = self.block1.output_size(size)
        size = self.block1.output_size(size)
        return size
        


class ShallowConv(nn.Module):
    """
    Shallow model that uses only 3 convolutions
    """
    def __init__(self):
        super(ShallowConv, self).__init__(self)
    
    def forward(x, self):
        pass



class ActorCritic(nn.Module):
    """
    Base class for the Model, with the forward and backward passes
    => Loss methods is abstract and needs to be updated to match a specific algorithm
    """
    def __init__(self, h, w, c, n_outputs, hidden_size=256):
        """You can have several types of body as long as they implement the size function"""
        super(ActorCritic, self).__init__()

        # Keeping some infos
        self.n_outputs = n_outputs
        self.input_size = (c, h, w)
        self.hidden_size = hidden_size

        # Sequential baseblocks

        self.convs = DeepConv(c)

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.conv.output_size(h, w),
                            out_features=256)

        self.body = nn.Sequential(
            self.convs,
            Flatten(),
            nn.ReLU(),
            self.fc,
            nn.ReLU()
        )

        # LSTM (Memory Layer)
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=self.hidden_size,
                            num_layers=1)

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size,
                               out_features=1)

        self.logits = nn.Linear(in_features=256,
                               out_features=self.n_outputs)

        # Using those distributions doesn't affect the gradient calculus
        # see https://arxiv.org/abs/1506.05254 for more infos
        self.dist = Categorical

    def tail(self, x):
        """
        Function used by the trainer, once CNNs and LSTM are computed
        Gives the values and the distributions
        """
        value = self.value(x)
        logits = self.logits(x)

        dist = self.dist(logits=logits)

        return value, dist
    
    def forward(self, inputs):
        x, lstm_hxs = inputs

        x = self.body(x)

        x, _ = self.lstm(x, lstm_hxs)

        value = self.value(x)
        logits = self.logits(x)

        dist = self.dist(logits=logits)

        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return value, action, action_log_prob, dist.probs, entropy
    
    def act(self, x, lstm_hxs):
        x = self.body.forward(x)

        # (seq_len, batch, input_size) formating in-place
        x.unsqueeze_(0)
        x, lstm_hxs = self.lstm(x, lstm_hxs)
        x.squeeze_(0)

        logits = self.logits(x)

        dist = self.dist(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, lstm_hxs