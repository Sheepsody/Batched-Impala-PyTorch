import torch.nn as nn
import torch as torch
from torch.distributions import Categorical
from enum import Enum

# Import predefined networks
from src.networks.utils import ShallowConv, DeepConv, Flatten

# For torch.jit
from typing import Tuple


class BodyType(Enum):
    """Enum to specify the body of out network"""

    SHALLOW = 1
    DEEP = 2


class ActorCriticLSTM(nn.Module):
    """Actor Critic network with an LSTM on top, and accelerated with jit"""

    __constants__ = ["flatten_dim", "hidden_size", "n_outputs", "sequence_length"]

    def __init__(self, h, w, c, n_outputs, sequence_length=1, body=BodyType.SHALLOW):
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
        else:
            raise AttributeError("The body type is not valid")

        conv_out = self.convs.output_size((h, w))
        self.flatten_dim = int(32 * conv_out[0] * conv_out[1])

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.flatten_dim, out_features=self.hidden_size)

        self.body = nn.Sequential(self.convs, Flatten(), nn.ReLU(), self.fc, nn.ReLU())

        # LSTM (Memory Layer)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1
        )

        # Allocate tensors as contiguous on GPU memory
        self.lstm.flatten_parameters()

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

        # Using those distributions doesn't affect the gradient calculus
        # see https://arxiv.org/abs/1506.05254 for more infos
        self.dist = Categorical

    @torch.jit.export
    def act(self, obs: torch.Tensor, lstm_hxs: Tuple[torch.Tensor, torch.Tensor]):
        """Performs an one-step prediction with detached gradients"""
        # x : (batch, input_size)
        x = self.body.forward(obs)

        x = x.unsqueeze(0)
        # x : (1, batch, input_size)

        x, lstm_hxs = self.lstm(x, lstm_hxs)
        x = x.squeeze(0)

        logits = self.logits(x)

        action, log_prob = self._act_dist(logits)

        lstm_hxs[0].detach_()
        lstm_hxs[1].detach_()

        return action.detach(), log_prob.detach(), lstm_hxs

    @torch.jit.ignore
    def _act_dist(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    @torch.jit.export
    def act_greedy(
        self, obs: torch.Tensor, lstm_hxs: Tuple[torch.Tensor, torch.Tensor]
    ):
        """Performs an one-step prediction with detached gradients"""
        # x : (batch, input_size)
        x = self.body.forward(obs)

        x = x.unsqueeze(0)
        # x : (1, batch, input_size)

        x, lstm_hxs = self.lstm(x, lstm_hxs)
        x = x.squeeze(0)

        # logits : (1, batch)
        logits = self.logits(x)
        action = torch.argmax(logits, dim=1)

        lstm_hxs[0].detach_()
        lstm_hxs[1].detach_()

        return action.detach(), lstm_hxs

    def forward(
        self, obs, lstm_hxs: Tuple[torch.Tensor, torch.Tensor], mask, behaviour_actions
    ):
        """
        x : (seq, batch, c, h, w)
        mask : (seq, batch)
        lstm_hxs : (1, batch, hidden)*2
        behaviour_actions : (seq, batch, num_actions)
        """
        # Check the dimentions
        seq, batch, c, h, w = obs.size()

        # 1. EFFICIENT COMPUTATION ON CNNs (time is folded with batch size)

        obs = obs.view(seq * batch, c, h, w)
        x = self.body.forward(obs)
        x = x.view(seq, batch, self.hidden_size)

        # 2. LSTM LOOP (same length events but can be resetted)

        mask = mask.unsqueeze(1)

        x = x.unsqueeze(1)  # (seq_len, 1, batch, input)
        x_lstm = []

        for i in range(self.sequence_length + 1):
            # One step pass of lstm
            result, lstm_hxs = self.lstm(x[i], lstm_hxs)

            # Zero lstm states is resetted
            for state in lstm_hxs:
                state = mask[i] * state

            x_lstm.append(result)

        x = torch.stack(tensors=x_lstm, dim=0)  # (seq_len, batch, input)

        x = x.view(seq * batch, self.hidden_size)
        behaviour_actions = behaviour_actions.view(seq * batch)  # Shape for dist

        target_value = self.value(x)
        logits = self.logits(x)

        target_log_probs, target_entropy = self._forward_dist(
            target_value, logits, behaviour_actions
        )

        target_log_probs = target_log_probs.view(seq, batch, 1)
        target_entropy = target_entropy.view(seq, batch, 1)
        target_value = target_value.view(seq, batch, 1)

        return target_log_probs, target_entropy, target_value

    @torch.jit.ignore
    def _forward_dist(self, value, logits, behaviour_actions):
        dist = self.dist(logits=logits)
        target_log_probs = dist.log_prob(behaviour_actions)
        target_entropy = dist.entropy()
        return target_log_probs, target_entropy
