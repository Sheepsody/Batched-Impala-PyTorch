import torch
from collections import namedtuple

Trajectory = namedtuple(
    "Trajectory",
    [
        'length', 
        'observations',
        'actions',
        'rewards',
        'log_probs',
        'done',
        'lstm_initial_hidden',
        'lstm_initial_cell'
    ]
)

class AgentMemory(object):
    """
    Object to store the trajectories in an optimized way
    """
    def __init__(self, num_steps, observation_shape, lstm_hidden_size, action_space):
        self.observations = torch.zeros(1+num_steps, *observation_shape)
        self.lstm_initial_hidden = torch.zeros(1, 1, lstm_hidden_size)
        self.lstm_initial_cell = torch.zeros(1, 1, lstm_hidden_size)
        self.actions = torch.zeros(1+num_steps, 1)
        self.rewards = torch.zeros(1+num_steps, 1)
        self.log_probs = torch.zeros(1+num_steps, 1)
        self.done = torch.zeros(1+num_steps, 1)
        self.step = 0
    
    def to(self, device):
        self.observations.to(device)
        self.lstm_initial_hidden.to(device)
        self.lstm_initial_cell.to(device)
        self.actions.to(device)
        self.rewards.to(device)
        self.log_probs.to(device)
        self.done.to(device)
    
    # Inplace operation
    def append_(self, observation, action, reward, log_prob, done):
        self.observations[self.step].copy_(observation)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(reward)
        self.log_probs[self.step].copy_(log_prob)
        self.done[self.step].copy_(done)
        self.step += 1
    
    def reset(self, initial_lstm_state):
        # No need to zero the tensors
        self.observations[0].copy_(self.observations[-1])
        self.lstm_initial_hidden.copy_(initial_lstm_state[0])
        self.lstm_initial_cell.copy_(initial_lstm_state[1])
        self.actions[0].copy_(self.actions[-1])
        self.rewards[0].copy_(self.rewards[-1])
        self.log_probs[0].copy_(self.log_probs[-1])
        self.done[0].copy_(self.done[-1])
        # Length of the current trajectory
        self.step = 1
    
    def enqueue(self, device=torch.device("cuda")):
        # Detach ? -> deletes the grad_fn attribute
        # Detach makes sure that we don't record the history of our tensor, not to backprop
        # This should already be done by detach but let's keep things safe !
        return Trajectory(
            length = self.step, 
            observations = self.observations[:self.step].clone().to(device),
            actions = self.actions[:self.step].clone().to(device),
            rewards = self.rewards[:self.step].clone().to(device), 
            log_probs = self.log_probs[:self.step].clone().to(device), 
            done = self.done[:self.step].clone().to(device),
            lstm_initial_hidden = self.lstm_initial_hidden.clone().to(device),
            lstm_initial_cell = self.lstm_initial_cell.clone().to(device)
        )