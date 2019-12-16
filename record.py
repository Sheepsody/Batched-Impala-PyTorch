import os
import argparse
import torch

from src.GymEnv import make_env
from src.utils import load_inference

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--state', type=str, default="SuperMarioKart-Snes/MarioCircuit.Act1")
parser.add_argument('-c', '--checkpoint', type=str, default="checkpoint.pt")

if __name__=="__main__":
    args = parser.parse_args()

    # Check if the files exists
    if (not os.path.isfile(args["state"])) or (not os.path.isfile(args["checkpoint"])):
        raise ValueError("Arguments are not valid")

    # Load model
    model = load_inference(args.checkpoint).float().to("cuda")

    # Environnement with OpenCV recording
    env = make_env(game='SuperMarioKart-Snes', state=args.state, stacks=1, size=(54, 54), record=True)
    obs = env.reset()
    done = False

    lstm_hxs = [torch.zeros((1, 1, 256)).to("cuda")]*2

    while not done:

        obs_tensor = torch.tensor(obs, dtype=torch.float).float().unsqueeze(0).to("cuda")
        action, lstm_hxs = model.act_greedy(obs_tensor, lstm_hxs)
        obs, _, done, _ = env.step(action)

    env.close()

# TODO : could add options from config file