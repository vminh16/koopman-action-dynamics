import torch
import numpy as np

class GRNContinuousDynamics:
    """
    Synthetic Biology Gene Regulatory Network (GRN) Continuous-Time Dynamics
    3 genes, 3 proteins, continuous ODE.
    Based on: Elowitz & Leibler (2000), Strelkowa & Barahona (2010)
    """

    def __init__(self, dt=1.0):
        # ===== Parameters =====
        self.K = 1.0       # Dissociation constant
        self.a = 1.6       # Promoter strength
        self.gamma = 0.16  # mRNA degradation rate
        self.beta = 0.16   # Protein synthesis rate
        self.c = 0.06      # Protein degradation rate

        # ===== System size =====
        self.Nstates = 6           # [x1,x2,x3: mRNA], [x4,x5,x6: proteins]
        self.udim = 3              # 3 control inputs
        self.dt = dt               # Simulation step size

        # ===== Control range =====
        self.umin = torch.tensor([0.0, 0.0, 0.0])
        self.umax = torch.tensor([2.0, 2.0, 2.0])

    # ----------------------
    # Continuous dynamics
    # ----------------------
    def dynamics(self, t, x, u):
        """
        Continuous-time GRN ODE:
            dx/dt = f(x, u)
        Args:
            x: [B, 6] → states
            u: [B, 3] → control inputs for genes 1,2,3
        Returns:
            dx/dt: [B, 6]
        """
        # Ensure batch dims
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)

        # Unpack states
        x1, x2, x3, x4, x5, x6 = [x[..., i] for i in range(6)]
        u1, u2, u3 = [u[..., i] for i in range(3)]

        # ODEs
        dx1 = -self.gamma * x1 + self.a / (self.K + x6**2) + u1
        dx2 = -self.gamma * x2 + self.a / (self.K + x4**2) + u2
        dx3 = -self.gamma * x3 + self.a / (self.K + x5**2) + u3
        dx4 = -self.c * x4 + self.beta * x1
        dx5 = -self.c * x5 + self.beta * x2
        dx6 = -self.c * x6 + self.beta * x3

        dxdt = torch.stack([dx1, dx2, dx3, dx4, dx5, dx6], dim=-1)
        return dxdt

    # ----------------------
    # Initial state sampler
    # ----------------------
    def sample_initial_state(self, batch_size=1):
        """
        Sample initial states for GRN:
            - mRNA (x1,x2,x3)  ∈ [0, 3]
            - Protein (x4,x5,x6) ∈ [5, 7]
        Returns:
            x0: [batch_size, 6]
        """
        x0 = torch.zeros(batch_size, 6)
        x0[:, :3] = 3.0 * torch.rand(batch_size, 3)            # mRNA
        x0[:, 3:] = 5.0 + 2.0 * torch.rand(batch_size, 3)      # proteins
        return x0
