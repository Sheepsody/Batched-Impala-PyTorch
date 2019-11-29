import random
import numpy as np

from torch.multiprocessing import Queue, Process
from src.GymEnv import make_env
from src.Statistics import SummaryType
import torchvision.transforms as T
from src.Trajectory import Trajectory, AgentMemory


import torch

import time




class Agent(Process):   
    """
    TODO Documentation with IMPALA
    Designed in the case of multiple machines
    TODO function for dynamic batching
    """

    def __init__(self, 
                 id_, 
                 behaviour_policy, 
                 target_policy, 
                 training_queue, 
                 states, 
                 exit_flag, 
                 statistics_queue, 
                 episode_counter, 
                 device="cuda",
                 step_max=5):

        # Calling parent class constructor
        super(Agent, self).__init__()

        self.id = id_

        # We set the worker as a daemon child
        # When the main process stops, it also breaks the workers
        self.daemon = True

        self.device = device

        # Policy followed by the actor during n-steps trajectory
        self.behaviour_policy = behaviour_policy
        self.behaviour_policy.to(self.device)
        # Policy that is being updated
        self.target_policy = target_policy

        self.training_queue = training_queue
        self.stats_queue = statistics_queue
        self.action_queue = Queue(maxsize=1)

        self.states = states

        self.step_max = step_max
        print(f"Outputs {self.behaviour_policy.n_outputs}")
        self.memory = AgentMemory(num_steps=self.step_max, 
                                     observation_shape=self.behaviour_policy.input_size, 
                                     lstm_hidden_size=self.behaviour_policy.hidden_size, 
                                     action_space=self.behaviour_policy.n_outputs)
        self.memory.to(self.device)
            
        self.episode_counter = episode_counter

        # Set exit as global value between processes
        self.exit = exit_flag

    def run(self):

        # Strating the process
        super(Agent, self).run()

        # Counter for the n-step return
        step = 0

        # Create a new environnement
        done = True

        # exit_flag is a shared torch.multiprocessing value
        while not self.exit.value:

            if done :
                # We start a new episode
                # Selecting a random state
                state = random.choice(self.states)
                self.env = make_env(state=state)

                obs = self.env.reset()
                done = False

                # Accumulated reward from the episode
                episode_reward = 0

                # Initialisation of LSTM memory
                # Shape should be num_layers, batch_size, hidden_size
                lstm_hxs = [torch.zeros((1, 1, 256)).to(self.device)]*2

            obs_tensor = torch.tensor(obs, dtype=torch.float) \
                .unsqueeze_(0) \
                .unsqueeze_(0) \
                .to(self.device)

            # Asynchronous prediction
            action, log_prob, lstm_hxs = self.behaviour_policy.act(obs_tensor, lstm_hxs)

            # Receive reward and new state           
            obs, reward, done, info = self.env.step(int(action.item()))

            # Update the trajectory with the latest step
            self.memory.append_(
                observation=obs_tensor.squeeze_(0), 
                action=action, 
                reward=torch.tensor(reward),
                log_prob=log_prob,
                done=torch.tensor(done)
            )

            episode_reward += reward

            # We reset our environnement if the game is done
            if step == self.step_max:

                assert self.memory.step == self.step_max+1, "Length issue"

                # Converting the data before sending
                self.training_queue.put(self.memory.enqueue())

                # Move the last experience
                self.memory.reset(initial_lstm_state=lstm_hxs)

                # Reinialize for next step
                step = 0

                # Updating the model with the latest weights
                self.behaviour_policy.load_state_dict(self.target_policy.state_dict())
                # The model is only used for inferencing
                self.behaviour_policy.eval()     

            # The step counter is placed here because of the first iteration
            # Coincides with the "length" of the trajectory buffer
            step += 1

            # Statistics about the episode
            if done :
                self.episode_counter.value += 1
                episode_duration = info["milliseconds"] + \
                    (info["seconds"] + info["minutes"]*60)*60

                self.stats_queue.put(
                    (SummaryType.SCALAR, "episode/duration", episode_duration))
                self.stats_queue.put(
                    (SummaryType.SCALAR, "episode/cumulated_reward", episode_reward))
                self.stats_queue.put(
                    (SummaryType.SCALAR, "episode/nb_episodes", self.episode_counter.value))

                # Only statistic that is logged
                print(f"Episode n° {self.episode_counter.value} finished \
                    \t Duration: {episode_duration} steps \
                    \t State : {state} \
                    \t Cumulated reward {episode_reward}")

                # Reset the episode
                self.env.close()

        # The background process must be alive for the Trainer
        # Tensors are passed as reference in pytorch
        time.sleep(1)