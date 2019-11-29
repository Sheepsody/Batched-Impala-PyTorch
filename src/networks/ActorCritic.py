import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Import predefined networks
from src.networks.utils import *


# Why using Torchscript ?
# nn.Module inheritance
# More efficient execution



class ActorCritic(torch.jit.ScriptModule):
    """
    Base class for the Model, with the forward and backward passes
    => Loss methods is abstract and needs to be updated to match a specific algorithm

    @torch.jit.script_method is used to declare new methods for jit (default is only forward)
    """
    __constants__ = ["input_size", "flatten_dim", "num_action", "hidden_size"]

    def __init__(self, h, w, c, n_outputs):
        """You can have several types of body as long as they implement the size function"""
        super(ActorCritic, self).__init__()

        # Keeping some infos
        self.n_outputs = n_outputs
        self.input_size = (c, h, w)
        self.hidden_size = 256

        # Sequential baseblocks
        self.convs = DeepConv(c)
        self.flatten_dim = self.conv.output_size(h, w)*32

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.flatten_dim,
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
        
        # Allocate tensors as contiguous on GPU memory
        self.lstm.flatten_parameters()

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size,
                               out_features=1)

        self.logits = nn.Linear(in_features=256,
                               out_features=self.n_outputs)

        # Using those distributions doesn't affect the gradient calculus
        # see https://arxiv.org/abs/1506.05254 for more infos
        self.dist = Categorical

    @torch.jit.script_method
    def act(self, obs, lstm_hxs):
        """Performs an one-step prediction with detached gradients"""
        # x : (batch, input_size)
        x = self.body.forward(obs)

        x = x.unsqueeze(0)
        # x : (1, batch, input_size)
        x, lstm_hxs = self.lstm(x, lstm_hxs)
        x = x.squeeze(0)

        logits = self.logits(x)

        dist = self.dist(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach(), lstm_hxs.detach()

    @torch.jit.script_method
    def forward(self, obs, lstm_hxs, reset_mask, behaviour_actions):
        """
        x : (seq, batch, c, h, w)
        reset_mask : (1, batch, length)
        lstm_hxs : (1, batch, hidden)*2
        behaviour_actions : (seq, batch, num_actions)
        """
        # Check the dimentions
        assert obs.dim() == 5
        seq, batch, c, h, w = obs.size()

        assert lstm_hxs[0].dim() == 3
        assert reset_mask.dim() == 3

        # 1. EFFICIENT COMPUTATION ON CNNs (time is folded with batch size)

        obs = obs.view(seq * batch, c, h, w)
        x = self.body.forward(obs)
        x = x.view(seq, batch, self.flatten_dim)

        # 2. LSTM LOOP (same length events but can be resetted)
        
        x = x.unsqueeze(1) # (seq_len, 1, batch, input)
        x_lstm = []

        for i in range(seq):
            # One step pass of lstm
            result, lstm_hxs = self.model.lstm(x[i], lstm_hxs)

            # Zero lstm states is resetted
            lstm_hxs = [(reset_mask[:, :, i]*state) for state in lstm_hxs]
            
            x_lstm.append(result)

        x = torch.cat(tensors=x_lstm, dim=0) # (seq_len, batch, input)
        
        # 3. EFFICIENT COMPUTATION OF LAST LAYERS (time is folded with batch size)

        x = x.view(seq * batch, self.flatten_dim)
        behaviour_actions = x.view(seq * batch, self.n_outputs)

        target_value = self.value(x)
        logits = self.logits(x)
        dist = self.dist(logits=logits)
        target_log_probs = dist.log_prob(behaviour_actions) 
        target_entropy = dist.entropy()

        target_log_probs = target_log_probs.view(seq, batch, 1)
        target_entropy = target_entropy.view(seq, batch, 1)
        target_value = target_value.view(seq, batch, 1)

        return target_log_probs, target_entropy, target_value