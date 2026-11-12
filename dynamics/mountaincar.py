import torch
import numpy as np

class MountainCarDynamics:
    def __init__(self, power=0.0015, dt=1.):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.power = power
        self.dt = dt

        self.umin = torch.tensor([self.min_action])
        self.umax = torch.tensor([self.max_action])
        self.Nstates = 2  # [position, velocity]

    def dynamics(self, t, x, u):
        """
        Continuous-time MountainCar dynamics (no clipping)
        x: [B, 2] position, velocity
        u: [B, 1] action (continuous power coef, typically in [-1,1])
        Returns: dx/dt [B, 2]
        """
        position = x[..., 0]
        velocity = x[..., 1]
        action = u[..., 0]

        # dx/dt for position is velocity
        dpos = velocity
        # dx/dt for velocity
        dvel = action * self.power - 0.0025 * torch.cos(3 * position)
        dxdt = torch.stack([dpos, dvel], dim=-1)
        return dxdt
    #
    # def sample_initial_state(self, batch_size=1):
    #     """
    #     Sample initial state as per gym:
    #         position ∈ [-0.6, -0.4]
    #         velocity = 0
    #     """
    #     position = torch.rand(batch_size) * 0.2 - 0.6
    #     velocity = torch.zeros(batch_size)
    #     return torch.stack([position, velocity], dim=-1)

    def sample_initial_state(self, batch_size=1):
        """
        Sample initial state uniformly over the whole domain:
            position ∈ [-1.2, 0.6]
            velocity ∈ [-0.07, 0.07]
        """
        position = torch.rand(batch_size) * (0.6 - (-1.2)) - 1.2
        velocity = torch.rand(batch_size) * (0.07 - (-0.07)) - 0.07
        return torch.stack([position, velocity], dim=-1)