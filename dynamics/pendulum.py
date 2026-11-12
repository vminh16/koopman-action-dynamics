import torch
import numpy as np

class PendulumDynamics:
    def __init__(self, g=10.0, l=1.0, m=1.0, dt=0.02):
        self.g = g
        self.l = l
        self.m = m
        self.dt = dt

        # input range
        self.umin = torch.tensor([-8.0])
        self.umax = torch.tensor([8.0])

        self.Nstates = 2

    def dynamics(self, t, x, u):
        """
        Continuous-time pendulum dynamics
        Args:
            x: [B, 2] → (theta, omega), real space
            u: [B, 1] → torque, real space
        Returns:
            dx/dt: [B, 2]
        """
        theta = self.angle_normalize(x[..., 0])
        omega = x[..., 1]
        torque = u[..., 0]

        dtheta = omega
        domega = (3.0 * self.g / (2.0 * self.l)) * torch.sin(theta) + (3.0 / (self.m * self.l ** 2)) * torque

        dxdt = torch.stack([dtheta, domega], dim=-1)
        return dxdt

    def sample_initial_state(self, batch_size=1):
        """
        Sample (theta, omega) in physical space:
            theta ∈ [-π, π], omega ∈ [-1, 1]
        """
        theta = (torch.rand(batch_size) - 0.5) * 2 * torch.pi
        omega = (torch.rand(batch_size) - 0.5) * 2.0
        return torch.stack([theta, omega], dim=-1)

    def angle_normalize(self, theta):
        """
        Wrap theta to [−π, π]
        """
        return ((theta + np.pi) % (2 * np.pi)) - np.pi