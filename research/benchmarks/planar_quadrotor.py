import torch
import numpy as np


class PlanarQuadrotorDynamics:
    """
    Planar Quadrotor (2D PVTOL) with Orientation-Dependent Thrust Vector.

    State:
        x = [y, z, phi, y_dot, z_dot, phi_dot]^T in R^6
        - y: horizontal position (m)
        - z: vertical altitude (m)
        - phi: pitch angle (rad)
        - y_dot: horizontal velocity (m/s)
        - z_dot: vertical velocity (m/s)
        - phi_dot: angular pitch rate (rad/s)

    Controls:
        u = [u1, u2]^T in R^2
        - u1: left rotor thrust (N)
        - u2: right rotor thrust (N)

    Equations of motion:
        dot(y)       = y_dot
        dot(z)       = z_dot
        dot(phi)     = phi_dot
        ddot(y)      = - (u1 + u2) / m * sin(phi)
        ddot(z)      =   (u1 + u2) / m * cos(phi) - g
        ddot(phi)    =   L * (u2 - u1) / I_yy

    Control-affine form:
        dot(x) = f0(x) + Gx(x) * u
        f0(x) = [y_dot, z_dot, phi_dot, 0, -g, 0]^T
        Gx(x) = [[0,                   0],
                 [0,                   0],
                 [0,                   0],
                 [-sin(phi)/m,         -sin(phi)/m],
                 [ cos(phi)/m,          cos(phi)/m],
                 [-L / I_yy,            L / I_yy]]
    """

    def __init__(
        self,
        m: float = 1.0,
        g: float = 9.81,
        I_yy: float = 0.01,
        L: float = 0.25,
        dt: float = 0.01,
        umax: float = 15.0,
    ):
        self.m = float(m)
        self.g = float(g)
        self.I_yy = float(I_yy)
        self.L = float(L)
        self.dt = float(dt)

        self.Nstates = 6
        self.Ncontrols = 2

        self.umin = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.umax = torch.tensor([umax, umax], dtype=torch.float32)

    def analytic_f0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unforced autonomous drift field: [B, 6] -> [B, 6].
        """
        y_dot = x[..., 3]
        z_dot = x[..., 4]
        phi_dot = x[..., 5]
        zeros = torch.zeros_like(y_dot)
        gravity = torch.full_like(z_dot, -self.g)

        return torch.stack([y_dot, z_dot, phi_dot, zeros, gravity, zeros], dim=-1)

    def analytic_Gx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Orientation-dependent input matrix: [B, 6] -> [B, 6, 2].
        """
        phi = x[..., 2]
        s = torch.sin(phi)
        c = torch.cos(phi)
        zeros = torch.zeros_like(phi)

        ay = -s / self.m
        az = c / self.m
        aphi_left = torch.full_like(phi, -self.L / self.I_yy)
        aphi_right = torch.full_like(phi, self.L / self.I_yy)

        # Column 0: effect of u1
        col0 = torch.stack([zeros, zeros, zeros, ay, az, aphi_left], dim=-1)
        # Column 1: effect of u2
        col1 = torch.stack([zeros, zeros, zeros, ay, az, aphi_right], dim=-1)

        return torch.stack([col0, col1], dim=-1)

    def dynamics(self, t, x=None, u=None) -> torch.Tensor:
        """
        Continuous-time dynamics: dot(x) = f0(x) + Gx(x) * u.
        Supports both dynamics(t, x, u) and dynamics(x, u).
        """
        if u is None:
            # Called as dynamics(x, u)
            x, u = t, x
            t = 0.0

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not isinstance(u, torch.Tensor):
            u = torch.as_tensor(u, dtype=torch.float32)

        f0 = self.analytic_f0(x)
        Gx = self.analytic_Gx(x)  # [..., 6, 2]

        if u.ndim == x.ndim - 1:
            u_val = u.unsqueeze(-1)
        else:
            u_val = u

        # Control-affine contribution: Gx @ u
        gu = torch.matmul(Gx, u_val.unsqueeze(-1)).squeeze(-1)
        return f0 + gu

    def sample_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample initial states:
            y, z in [-1.0, 1.0]
            phi in [-0.2, 0.2] rad (~11.5 deg)
            y_dot, z_dot in [-0.5, 0.5] m/s
            phi_dot in [-0.5, 0.5] rad/s
        """
        pos_yz = (torch.rand(batch_size, 2) - 0.5) * 2.0
        phi = (torch.rand(batch_size, 1) - 0.5) * 0.4
        vel_yz = (torch.rand(batch_size, 2) - 0.5) * 1.0
        phi_dot = (torch.rand(batch_size, 1) - 0.5) * 1.0
        return torch.cat([pos_yz[:, :1], pos_yz[:, 1:2], phi, vel_yz[:, :1], vel_yz[:, 1:2], phi_dot], dim=-1)

