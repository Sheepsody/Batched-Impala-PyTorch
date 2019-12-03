import torch
import torch.nn as nn

from src.vtrace.VTrace import VTrace


class Impala(nn.Module):
    # Can be constants
    Gamma, rho, multi_steps
    item : Final[type]
    vtrace : Final[nn.Module] # FIXME check that this is authorized
    
    def __init__(self):
        super(Impala, self).__init__()

        # TODO initialize the constants
    
    @torch.jit.export
    def test(self):
        pass
    
    @classmethod
    def clone(cls, model, device):
        cloned = cls(*args) # FIXME
        cloned.load_state_dict(model.state_dict())
        return cloned.to(device)
    
    # TODO : function to copy the weights of the model
    def sync_target_with_online(self):
        
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    @torch.jit.export
    def loss():
        # Divide in several other methods
        pass

    @torch.jit.export
    def act():
        pass

    @torch.jit.export
    def greedy_act():
        pass

