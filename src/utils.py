import numpy as np

import torch
import cv2

from src.networks.ActorCritic import ActorCritic


"""
FUNCTIONS TO SAVE AND RESTORE
Yet supported :
* model
* optimizer
* epoch
"""

def load_checkpoint(checkpoint_path, config):
    """
    Load on GPU the model trained on GPU
    """
    checkpoint = torch.load(checkpoint_path)

    epoch = checkpoint["epoch"]
    
    model = ActorCritic(56, 128, 1, 9).float()
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.SGD(model.parameters(),
                        lr=config["optimizer"]["learning_rate"], 
                        momentum=config["optimizer"]["momentum"],
                        weight_decay=config["optimizer"]["weight_decay"],
                        dampening=config["optimizer"]["dampening"],
                        Nesterov=config["optimizer"]["nesterov"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return epoch, model, optimizer

def load_inference(checkpoint_path, device=torch.device("cpu")):
    """
    Loads a cpu 
    """
    model = ActorCritic(56, 128, 1, 9).float()
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device)["model_state_dict"])
    model.eval()
    model.share_memory()
    
    return model