import torch
import torch.nn as nn
from collections import OrderedDict

class StructuredEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim, num_layers=2, hidden_size=64):
        """
        Constructs φ(x) = [x, φ̃(x)], where φ̃(x) is MLP output of dim (latent_dim - state_dim)
        Args:
            state_dim: Dim of original system state
            latent_dim: Total Koopman space dim (must ≥ state_dim)
            num_layers: Hidden layer count (for φ̃(x))
            hidden_size: Hidden dim size
        """
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.extra_dim = latent_dim - state_dim

        assert self.extra_dim >= 0, "latent_dim must be ≥ state_dim"

        if self.extra_dim > 0:
            dims = [state_dim] + [hidden_size] * num_layers + [self.extra_dim]
            layers = OrderedDict()
            for i in range(len(dims) - 2):
                layers[f"linear_{i}"] = nn.Linear(dims[i], dims[i+1])
                layers[f"relu_{i}"] = nn.ReLU()
            layers["output"] = nn.Linear(dims[-2], dims[-1])
            self.extra_net = nn.Sequential(layers)
        else:
            self.extra_net = None  # No extra dictionary part

    def forward(self, x):
        if self.extra_net is None:
            return x
        extra = self.extra_net(x)
        return torch.cat([x, extra], dim=-1)  # φ(x) = [x, φ̃(x)]



class UEncoder(nn.Module):
    def __init__(self, state_dim, control_dim, num_layers=2, hidden_size=64):
        super().__init__()
        dims = [state_dim] + [hidden_size] * (num_layers - 1) + [control_dim]

        layers = OrderedDict()
        for i in range(len(dims) - 1):
            layers[f"linear_{i}"] = nn.Linear(dims[i], dims[i + 1])
            if i < len(dims) - 2:
                layers[f"relu_{i}"] = nn.ReLU()
        self.bilinear_net = nn.Sequential(layers)

    def forward(self, x, u):
        """
        x: [B, state_dim]
        u: [B, control_dim]
        return: g(x) ⊙ u
        """
        g = self.bilinear_net(x)
        return g * u