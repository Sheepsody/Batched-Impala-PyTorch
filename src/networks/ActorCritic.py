import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Import predefined networks
from src.networks.utils import *

from typing import Dict

try:
    from typing_extensions import Final
except:
    from torch.jit import Final


from enum import Enum


class BodyType(Enum):
    SHALLOW = 1
    DEEP = 2


class ActorCritic(nn.Module):
    flatten_dim: Final[int]
    hidden_size: Final[int]
    n_outputs: Final[int]

    def __init__(self, h, w, c, n_outputs, body=BodyType.SHALLOW):
        """You can have several types of body as long as they implement the size function"""
        super(ActorCritic, self).__init__()

        # Keeping some infos
        self.n_outputs = n_outputs
        self.input_size = (c, h, w)
        self.hidden_size = 256

        # Sequential baseblocks
        if body == BodyType.SHALLOW:
            self.convs = ShallowConv(c)
        elif body == BodyType.DEEP:
            self.convs = DeepConv(c)
        else :
            raise AttributeError("The body type is not valid")

        self.flatten_dim = self.convs.output_size(h, w)*32

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.flatten_dim,
                            out_features=self.hidden_size)

        self.body = nn.Sequential(
            self.convs,
            Flatten(),
            nn.ReLU(),
            self.fc,
            nn.ReLU()
        )

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size,
                               out_features=1)

        self.logits = nn.Linear(in_features=self.hidden_size,
                               out_features=self.n_outputs)

        # Using those distributions doesn't affect the gradient calculus
        # see https://arxiv.org/abs/1506.05254 for more infos
        self.dist = Categorical

    @torch.jit.export
    def act(self, obs, lstm_hxs):
        """Performs an one-step prediction with detached gradients"""
        # x : (batch, input_size)
        x = self.body.forward(obs)

        logits = self.logits(x)

        dist = self.dist(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach(), lstm_hxs.detach()

    def forward(self, obs, behaviour_actions):
        """
        x : (seq, batch, c, h, w)
        behaviour_actions : (seq, batch, num_actions)
        """
        # Check the dimentions
        assert obs.dim() == 5
        seq, batch, c, h, w = obs.size()

        # 1. EFFICIENT COMPUTATION ON CNNs (time is folded with batch size)

        obs = obs.view(seq * batch, c, h, w)
        x = self.body.forward(obs) # x : (seq * batch, flatten_dim)
        
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



class ActorCriticLSTM(nn.Module):
    flatten_dim: Final[int]
    hidden_size: Final[int]
    n_outputs: Final[int]
    sequence_length: Final[int]

    def __init__(self, h, w, c, n_outputs, sequence_length, body=BodyType.SHALLOW):
        """You can have several types of body as long as they implement the size function"""
        super(ActorCriticLSTM, self).__init__()

        # Keeping some infos
        self.n_outputs = n_outputs
        self.input_size = (c, h, w)
        self.hidden_size = 256
        self.sequence_length = sequence_length

        # Sequential baseblocks
        if body == BodyType.SHALLOW:
            self.convs = ShallowConv(c)
        elif body == BodyType.DEEP:
            self.convs = DeepConv(c)
        else :
            raise AttributeError("The body type is not valid")

        self.flatten_dim = self.convs.output_size(h, w)*32

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.flatten_dim,
                            out_features=self.hidden_size)

        self.body = nn.Sequential(
            self.convs,
            Flatten(),
            nn.ReLU(),
            self.fc,
            nn.ReLU()
        )

        # LSTM (Memory Layer)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=1)
        
        # Allocate tensors as contiguous on GPU memory
        self.lstm.flatten_parameters()

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size,
                               out_features=1)

        self.logits = nn.Linear(in_features=self.hidden_size,
                               out_features=self.n_outputs)

        # Using those distributions doesn't affect the gradient calculus
        # see https://arxiv.org/abs/1506.05254 for more infos
        self.dist = Categorical

    @torch.jit.export
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

    @torch.jit.export
    def forward(self, obs, lstm_hxs, mask, behaviour_actions):
        """
        x : (seq, batch, c, h, w)
        mask : (seq, batch)
        lstm_hxs : (1, batch, hidden)*2
        behaviour_actions : (seq, batch, num_actions)
        """
        # Check the dimentions
        assert obs.dim() == 5
        seq, batch, c, h, w = obs.size()
        assert seq==self.sequence_length+1, "Issue with sequence lengths"

        # 1. EFFICIENT COMPUTATION ON CNNs (time is folded with batch size)

        obs = obs.view(seq * batch, c, h, w)
        x = self.body.forward(obs)
        x = x.view(seq, batch, self.flatten_dim)

        # 2. LSTM LOOP (same length events but can be resetted)
        
        mask = mask.unsqueeze(1)

        x = x.unsqueeze(1) # (seq_len, 1, batch, input)
        x_lstm = []

        for i in range(self.sequence_length+1):
            # One step pass of lstm
            result, lstm_hxs = self.model.lstm(x[i], lstm_hxs)

            # Zero lstm states is resetted
            for state in lstm_hxs:
                state = mask[i] * state
            
            x_lstm.append(result)

        x = torch.stack(tensors=x_lstm, dim=0) # (seq_len, batch, input)
        

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