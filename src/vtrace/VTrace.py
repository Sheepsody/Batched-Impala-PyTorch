import torch
import torch.nn as nn

try:
    from typing_extensions import Final
except:
    from torch.jit import Final


class VTrace(nn.Module):
    """
    New PyTorch module for computing v-trace as in https://arxiv.org/abs/1802.01561 (IMPALA)
    On the on-policy case, v-trace reduces to the n-step Bellman target

    Parameters
    ----------
    nn.Module: 
    discount_factor : float
        our objective is to maximize the sum of discounted rewards
    rho : float
        controls the nature of the value function we converge to
    cis : float
        controls the speed of convergence
    
    Notes
    -----
    rho and cis are truncated importance sampling weights 
    As in https://arxiv.org/abs/1802.01561 it is assumed that rho >= cis
    """

    __constants__ = ["rho", "cis", "discount_factor", "sequence_length"]

    def __init__(self, discount_factor, sequence_length, rho=1.0, cis=1.0):
        super(VTrace, self).__init__()

        assert rho >= cis, "Truncation levels do not satify the asumption rho >= cis"

        self.rho = nn.Parameter(
            torch.tensor(rho, dtype=torch.float), requires_grad=False
        )
        self.cis = nn.Parameter(
            torch.tensor(cis, dtype=torch.float), requires_grad=False
        )
        self.discount_factor = discount_factor
        self.sequence_length = sequence_length

    def forward(self, target_value, rewards, target_log_policy, behaviour_log_policy):
        """
        v-trace targets are computed recursively
        v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
        As in the originial (tensorflow) implementation, evaluation is done on the CPU
        
        Parameters
        ----------
        target_value: torch.tensor 
            predicted value by the target policy
        rewards: torch.tensor
            rewards obtained by agents
        target_log_policy: torch.tensor
        behaviour_log_policy: torch.tensor
        
        Returns
        -------
        vtrace : torch.tensor
            vtrace targets for the target value
        rhos : torch.tensor
            truncation levels when computing v-trace

        Notes
        ----- 
        (seq_len, batch, ...)
        unittest can be found to test with python, with original tensorflow code       
        """
        assert rewards.size()[0] == self.sequence_length

        # Copy on the same device at the input tensor
        vtrace = torch.zeros(target_value.size(), device=target_value.device)

        # Computing importance sampling for truncation levels
        importance_sampling = torch.exp(target_log_policy - behaviour_log_policy)
        clipped_rhos = torch.min(self.rho, importance_sampling)
        ciss = torch.min(self.cis, importance_sampling)

        # Recursive calculus
        # Initialisation : v_{-1}
        # v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
        vtrace[-1] = target_value[-1]  # Bootstrapping

        # Computing the deltas
        delta = clipped_rhos * (
            rewards + self.discount_factor * target_value[1:] - target_value[:-1]
        )

        # Pytorch has no funtion such as tf.scan or theano.scan
        # This disgusting is compulsory for jit as reverse is not supported
        for j in range(self.sequence_length):
            i = (self.sequence_length - 1) - j
            vtrace[i] = (
                target_value[i]
                + delta[i]
                + self.discount_factor * ciss[i] * (vtrace[i + 1] - target_value[i + 1])
            )

        # Don't forget to detach !
        # We need to remove the bootstrapping
        return vtrace.detach(), clipped_rhos.detach()
