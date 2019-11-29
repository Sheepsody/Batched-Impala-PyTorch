# TODO
# This would be a callback to check the performancesimport random
import numpy as np
import torch
import cv2
from torch.multiprocessing import Process
import os
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from src.Statistics import SummaryType

from src.GymEnv import make_env

from src.utils import load_inference


# TODO : baseclass for callbacks


class StateCallback(Process):

    def __init__(self, checkpoint_path, step, records_folder, train_set, test_set, statistics_queue):

        super(StateCallback, self).__init__()

        # Child process
        self.daemon = True

        # Init the environnement
        self.game = 'SuperMarioKart-Snes'
        self.train_set = train_set
        self.test_set = test_set

        # Recording folder
        assert os.path.isdir(records_folder), "The folder is not valid"
        self.path = os.path.join(records_folder, f"callback_{step}")
        os.makedirs(self.path, exist_ok=True)

        # Checkpoint path
        print("plop")
        self.model = load_inference(checkpoint_path)
        self.step = step

        # Think about this
        self.maps_dir = "maps"
        self.statistics_queue = statistics_queue

        self.step_max = 10000

        print("plop")
    
    def figure(self, ax, x, y, speed, state):
        # Rescale x and y to figure
        x, y = x/4, y/4

        # Read image and RGB-> BGR
        background_path = os.path.join(self.maps_dir, state+".png")
        background = cv2.imread(background_path)
        background = background[:,:,::-1]
        
        # Create a line of segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(speed.min(), speed.max())
        lc = LineCollection(segments, cmap='winter', norm=norm)

        # Set the values used for colormapping
        lc.set_array(speed)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

        ax.imshow(background, alpha=0.3)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f"State: {state}, episodes: {self.step}")

    def record_state(self, state):

        env = make_env(game='SuperMarioKart-Snes', state=state)
        env.record_movie(os.path.join(self.path, state+".bk2"))
        obs = env.reset()

        x, y, speed = [], [], []

        done = False
        step = 0

        while not done and step < self.step_max:

            obs_tensor = torch.tensor(obs, dtype=torch.float).float().unsqueeze(0)
            action, _ = self.model.select_action(obs_tensor)
            obs, _, done, info = env.step(action)

            x.append(info["x"])
            y.append(info["y"])
            speed.append(info["speed"])

            step += 1

        x, y, speed = np.array(x), np.array(y), np.array(speed)

        env.stop_record()
        env.close()

        return x, y, speed

    def run(self):

        super(StateCallback, self).run()

        print("started")

        # Train set 
        subplot_shape = []
        ncols = ceil(np.sqrt(len(self.train_set)))
        nrows = ceil(len(self.train_set)/ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        fig.gca().invert_yaxis()

        # Fix issue when only one plot
        if nrows == 1 and ncols == 1:
            axs = [axs]

        for ax, state in zip(axs, self.train_set):
            # Recording the state
            x, y, speed = self.record_state(state)
            # Making the figure (modifies ax)
            self.figure(ax, x, y, speed, state)
        
        # Now send it to the tensorboard !
        self.statistics_queue.put((SummaryType.FIGURE, "callback/train_set/", fig))

        # Train set 
        subplot_shape = []
        ncols = ceil(np.sqrt(len(self.test_set)))
        nrows = ceil(len(self.test_set)/subplot_shape[0])
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        fig.gca().invert_yaxis()

        for ax, state in zip(axs, self.test_set):
            # Recording the state
            x, y, speed = self.record_state(state)
            # Making the figure (modifies ax)
            self.figure(ax, x, y, speed, state)
        
        # Now send it to the tensorboard !
        self.statistics_queue.put((SummaryType.FIGURE, "callback/test_set/", fig))

        print("finished")
