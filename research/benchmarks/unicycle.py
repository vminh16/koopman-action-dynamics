import torch
import numpy as np


class UnicycleDynamics:
    """
    Kinematic Unicycle with Full-State and Heading-Dependent Input Direction.

    State:
        x = [px, py, theta]^T in R^3
        - px: position along x
        - py: position along y
        - theta: heading angle in [-pi, pi]

    Controls:
        u = [v, omega]^T in R^2
        - v: forward linear velocity
        - omega: angular velocity (yaw rate)

    Equations of motion:
        dot(px)    = v * cos(theta)
        dot(py)    = v * sin(theta)
        dot(theta) = omega

    Control-affine form:
        dot(x) = f0(x) + Gx(x) * u
        f0(x) = [0, 0, 0]^T
        Gx(x) = [[cos(theta), 0],
                 [sin(theta), 0],
                 [0,          1]]
    """

    def __init__(
        self,
        dt: float = 0.02,
        v_bounds: tuple[float, float] = (-1.0, 1.0),
        omega_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        self.dt = float(dt)
        self.Nstates = 3
        self.Ncontrols = 2

        self.umin = torch.tensor([v_bounds[0], omega_bounds[0]], dtype=torch.float32)
        self.umax = torch.tensor([v_bounds[1], omega_bounds[1]], dtype=torch.float32)

    def angle_normalize(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Normalize angle theta to [-pi, pi) without in-place tensor mutation.
        """
        return ((theta + torch.pi) % (2.0 * torch.pi)) - torch.pi

    def analytic_f0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unforced autonomous drift field: [B, 3] -> [B, 3] (zeros).
        """
        return torch.zeros_like(x)

    def analytic_Gx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Heading-dependent input matrix: [B, 3] -> [B, 3, 2].
        """
        theta = x[..., 2]
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)

        # Row 0: [cos(theta), 0]
        # Row 1: [sin(theta), 0]
        # Row 2: [0, 1]
        col0 = torch.stack([c, s, zeros], dim=-1)   # [..., 3]
        col1 = torch.stack([zeros, zeros, ones], dim=-1)  # [..., 3]
        return torch.stack([col0, col1], dim=-1)     # [..., 3, 2]

    def dynamics(self, t, x=None, u=None) -> torch.Tensor:
        """
        Continuous-time kinematics: dot(x) = Gx(x) * u.
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

        Gx = self.analytic_Gx(x)  # [..., 3, 2]

        if u.ndim == x.ndim - 1:
            u_val = u.unsqueeze(-1)
        else:
            u_val = u

        # Control-affine contribution: Gx @ u
        dx = torch.matmul(Gx, u_val.unsqueeze(-1)).squeeze(-1)
        return dx

    def sample_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample initial states: px, py in [-2.0, 2.0], theta in [-pi, pi].
        """
        pos = (torch.rand(batch_size, 2) - 0.5) * 4.0
        theta = (torch.rand(batch_size, 1) - 0.5) * (2.0 * torch.pi)
        return torch.cat([pos, theta], dim=-1)

