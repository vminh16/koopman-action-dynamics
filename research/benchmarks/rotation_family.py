import torch
import numpy as np


class RotationFamilyDynamics:
    """
    2D Control-Affine System with State-Dependent Input Direction.

    Dynamics:
        dot(x) = f0(x) + Gx(x) * u

    where:
        f0(x) = [x2, -x1 - 0.2 * x2]^T   (damped harmonic oscillator)
        Gx(x) = [cos(a * x1), sin(a * x1)]^T
        a >= 0 is the rotation coupling strength parameter.

    When a = 0: Gx(x) = [1, 0]^T (constant input direction).
    When a > 0: The actuation direction rotates as a function of state x1.
    """

    def __init__(self, a: float = 1.0, dt: float = 0.02):
        self.a = float(a)
        self.dt = float(dt)
        self.Nstates = 2
        self.Ncontrols = 1

        self.umin = torch.tensor([-2.0])
        self.umax = torch.tensor([2.0])

    def analytic_f0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unforced autonomous drift field: [B, 2] -> [B, 2].
        """
        x1 = x[..., 0]
        x2 = x[..., 1]
        dx1 = x2
        dx2 = -x1 - 0.2 * x2
        return torch.stack([dx1, dx2], dim=-1)

    def analytic_Gx(self, x: torch.Tensor) -> torch.Tensor:
        """
        State-dependent input matrix: [B, 2] -> [B, 2, 1].
        """
        x1 = x[..., 0]
        theta = self.a * x1
        g1 = torch.cos(theta)
        g2 = torch.sin(theta)
        return torch.stack([g1, g2], dim=-1).unsqueeze(-1)

    def dynamics(self, t, x=None, u=None) -> torch.Tensor:
        """
        Continuous-time dynamics dot(x) = f0(x) + Gx(x) * u.
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
        Gx = self.analytic_Gx(x)  # [..., 2, 1]

        # u can be [B, 1] or [B]
        if u.ndim == x.ndim - 1:
            u_val = u.unsqueeze(-1)
        else:
            u_val = u

        # Control-affine contribution: Gx @ u
        gu = torch.matmul(Gx, u_val.unsqueeze(-1)).squeeze(-1)
        return f0 + gu

    def sample_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample initial states x = (x1, x2) in [-1.5, 1.5]^2.
        """
        return (torch.rand(batch_size, 2) - 0.5) * 3.0

