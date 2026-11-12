import torch
import numpy as np

class CartPoleDynamics:
    def __init__(self, g=9.8, m_cart=1.0, m_pole=0.1, l=0.5, dt=0.02):
        self.g = g
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.total_mass = self.m_cart + self.m_pole
        self.l = l  # half pole length
        self.polemass_length = self.m_pole * self.l
        self.dt = dt

        # input: force range
        self.umin = torch.tensor([-10.0])
        self.umax = torch.tensor([10.0])

        self.Nstates = 4

    def dynamics(self, t, x, u):
        """
        Continuous-time CartPole ODE: [x, x_dot, theta, theta_dot]
        """
        x_pos = x[..., 0]
        x_dot = x[..., 1]
        theta = self.angle_normalize(x[..., 2])
        x[...,2] = theta
        theta_dot = x[..., 3]
        force = u[..., 0]

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.l * (4.0/3.0 - self.m_pole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        dxdt = torch.stack([x_dot, xacc, theta_dot, thetaacc], dim=-1)
        return dxdt

    # def sample_initial_state(self, batch_size=1):
    #     """
    #     Match gym reset:
    #         All 4 states sampled uniformly from [-0.05, 0.05]
    #     """
    #     return (torch.rand(batch_size, self.Nstates) - 0.5) * 0.1

    def sample_initial_state(self, batch_size: int = 1):
        """
        Match gym reset:
            state[2] ~ U(-0.1, 0.1)
            state[0] ~ U(-4, 4)
            others = 0
        """

        x0 = torch.zeros((batch_size, self.Nstates))

        # state[2] ∈ [-0.1, 0.1]
        x0[:, 2] = torch.rand((batch_size,)) * 0.2 - 0.1
        # state[0] ∈ [-4, 4]
        x0[:, 0] = torch.rand((batch_size,)) * 8.0 - 4.0

        return x0

    def angle_normalize(self, theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi