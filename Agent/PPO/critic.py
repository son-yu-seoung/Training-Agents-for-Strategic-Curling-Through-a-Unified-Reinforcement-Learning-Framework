import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 

import utils


class ValueCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, opt):
        super().__init__()

        self.V = utils.mlp(opt.obs_dim, opt.hidden_dim, 1, opt.hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        # obs: (B, 16, 8) → (B, 128)
        obs = obs.view(obs.shape[0], -1)
        v = self.V(obs)
        self.outputs['v'] = v
        return v
