import torch

from src.networks.ActorCritic import ActorCriticLSTM



def load_checkpoint(checkpoint_path, config):
    """Load on GPU the model trained on GPU"""
    checkpoint = torch.load(checkpoint_path)

    epoch = checkpoint["epoch"]
    
    model = ActorCriticLSTM(54, 54, 1, 9).float()
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
    model = ActorCriticLSTM(54, 54, 1, 9).float()
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device)["model_state_dict"])
    model.eval()
    model.share_memory()
    
    return model
