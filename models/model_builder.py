import torch
import torch.nn as nn
from .encoder import StructuredEncoder, UEncoder
from .koopman_operator import KoopmanOperator

import torch
import torch.nn as nn
from torchdiffeq import odeint
from .encoder import StructuredEncoder, UEncoder
from .koopman_operator import KoopmanOperator


class NeuralODEModel(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=128, dt=0.02, ode_solver="euler"):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.ode_solver = ode_solver

        # f(x, u) dynamics
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def dynamics(self, t, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

    def forward(self, x0, u_seq):
        """
        Args:
            x0: [B, dx]
            u_seq: [B, T, du]
        Returns:
            x_seq: [B, T, dx]
        """
        B, T, _ = u_seq.shape
        x = x0
        x_seq = []

        for t in range(T):
            u_t = u_seq[:, t, :]

            def dyn(t_dummy, x_):
                return self.dynamics(t_dummy, x_, u_t)

            t_span = torch.tensor([0, self.dt], device=x.device)
            x = odeint(dyn, x, t_span, method=self.ode_solver)[-1]
            x_seq.append(x)

        return torch.stack(x_seq, dim=1)

class MLPModel(nn.Module):
    """
    Discrete-time black-box model:
        x_{t+1} = f_theta([x_t, u_t])
    """
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "tanh",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim

        act = nn.Tanh if activation.lower() == "tanh" else nn.ReLU

        layers = []
        in_dim = state_dim + control_dim
        for i in range(num_layers):
            out_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def step(self, x, u):
        """
        x: [B, dx], u: [B, du] -> x_next: [B, dx]
        """
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

    def forward(self, x0, u_seq):
        """
        Args:
            x0: [B, dx]
            u_seq: [B, T, du]
        Returns:
            x_seq: [B, T, dx]
        """
        B, T, _ = u_seq.shape
        x = x0
        xs = []

        for t in range(T):
            u_t = u_seq[:, t, :]
            x = self.step(x, u_t)
            xs.append(x)
        return torch.stack(xs, dim=1)  # [B, T, dx]


class KoopmanModel(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        latent_dim,
        dt=0.02,
        num_layers=3,
        hidden_size=64,
        koopman_mode="free",
        use_modulated_u=False,
        u_encoder_layers=2,
        u_encoder_hidden=64,
        ode_solver="euler"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.dt = dt
        self.use_modulated_u = use_modulated_u
        self.ode_solver = ode_solver  # e.g., "euler", "dopri5", "rk4", etc.

        # Encoder for x → z
        self.encoder = StructuredEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_size=hidden_size
        )

        # Koopman operator A, B (free or parameterized)
        self.operator = KoopmanOperator(
            latent_dim=latent_dim,
            control_dim=control_dim,
            state_dim=state_dim,
            mode=koopman_mode
        )

        # Optional: u encoder modulated by x (usually x_hat = first few dims of z)
        if self.use_modulated_u:
            self.u_encoder = UEncoder(
                state_dim=state_dim,
                control_dim=control_dim,
                num_layers=u_encoder_layers,
                hidden_size=u_encoder_hidden
            )

    def encode(self, x):
        return self.encoder(x)  # [B, dz]

    def dynamics(self, t, z, u):
        if self.use_modulated_u:
            x_hat = z[:, :self.state_dim]
            u = self.u_encoder(x_hat, u)
        return self.operator.dynamics(t, z, u)  # returns dz/dt

    def forward(self, x0, u_seq):
        """
        Args:
            x0: [B, dx]
            u_seq: [B, T, du]
        Returns:
            z_seq: [B, T, dz]
        """
        B, T, _ = u_seq.shape
        z0 = self.encode(x0)  # [B, dz]
        z = z0
        if z.ndim == 3 and z.shape[1] == 1:
            z = z.squeeze(1)
        z_seq = []

        for t in range(T):
            u_t = u_seq[:, t, :]  # control at t

            # define fixed-control Koopman dynamics
            def koopman_dynamics(t_dummy, z_):
                return self.dynamics(t_dummy, z_, u_t)

            # integrate over [0, dt]
            t_span = torch.tensor([0, self.dt], device=z.device)
            z = odeint(koopman_dynamics, z, t_span, method=self.ode_solver)[-1]
            z_seq.append(z)

        return torch.stack(z_seq, dim=1)  # [B, T, dz]

def build_model(system_config, model_config):
    return KoopmanModel(
        state_dim=system_config["state_dim"],
        control_dim=system_config["umin"].shape[0],
        latent_dim=model_config["latent_dim"],
        dt=system_config["dt"],
        num_layers=model_config["num_layers"],
        hidden_size=model_config["hidden_size"],
        koopman_mode=model_config["koopman_mode"],
        use_modulated_u=model_config["use_modulated_u"],
        u_encoder_layers=model_config["u_encoder_layers"],
        u_encoder_hidden=model_config["u_encoder_hidden"],
        ode_solver=model_config.get("ode_solver", "euler")
    )