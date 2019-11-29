# TODO model replay

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

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

model = load_inference("checkpoint.pt")

env = make_env(game='SuperMarioKart-Snes', state="MarioCircuit.Act1")
obs = env.reset()

out = cv2.VideoWriter('output.avi',fourcc, 20.0, (obs.shape[1],obs.shape[0]))

done = False
step = 0

while not done and step < 1000:

    out.write(obs)

    obs_tensor = torch.tensor(obs, dtype=torch.float).float().unsqueeze(0)
    action, _ = model.select_action(obs_tensor)
    obs, rew, done, info = env.step(action)

    step += 1

env.close()

out.release()
