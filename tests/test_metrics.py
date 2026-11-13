# tests/test_metrics.py
import unittest
import torch
import numpy as np
from research.benchmarks.rotation_family import RotationFamilyDynamics
from research.diagnostics.metrics import (
    compute_action_response_error,
    compute_vector_field_mismatches,
    compute_continuous_gramian_and_margin,
    compute_candidate_regret,
)

class MockKoopmanModel(torch.nn.Module):
    def __init__(self, state_dim=2, latent_dim=4, control_dim=1):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.A = torch.nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.B = torch.nn.Parameter(torch.zeros(latent_dim, control_dim))
        self.B.data[0, 0] = 1.0 # acts along first physical coordinate

    def encode(self, x):
        # Linear embedding: [x1, x2, 0, 0]
        pad = torch.zeros(x.shape[0], self.latent_dim - self.state_dim, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def decode(self, z):
        return z[:, :self.state_dim]

    def forward(self, x, u):
        z = self.encode(x)
        dz = torch.matmul(z, self.A.T) + torch.matmul(u, self.B.T)
        return self.decode(dz)

class TestMetrics(unittest.TestCase):
    def test_gramian_quadrature(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.5]], dtype=np.float64)
        B = np.array([[0.0], [1.0]], dtype=np.float64)
        C = np.array([[1.0, 0.0]], dtype=np.float64)
        res = compute_continuous_gramian_and_margin(A, B, C, T=1.0)
        self.assertIn("rho_T", res)
        self.assertIn("lambda_min_Wy", res)
        self.assertIn("lambda_max_Qy", res)
        self.assertGreater(res["rho_T"], 0.0)
        self.assertGreater(res["lambda_min_Wy"], 0.0)

    def test_response_error_calculation(self):
        model = MockKoopmanModel()
        sys = RotationFamilyDynamics(a=0.0)
        x = torch.randn(10, 2)
        u = torch.ones(10, 1)
        u_ref = torch.zeros(10, 1)
        err = compute_action_response_error(model, sys, x, u, u_ref)
        self.assertIsInstance(err, float)
        self.assertGreaterEqual(err, 0.0)

    def test_vector_field_mismatches(self):
        model = MockKoopmanModel()
        sys = RotationFamilyDynamics(a=1.0)
        x = torch.randn(5, 2)
        res = compute_vector_field_mismatches(model, sys, x)
        self.assertIn("E_phys", res)
        self.assertIn("E_lift", res)
        self.assertIn("E_drift", res)
        self.assertIn("d_orient", res)
        self.assertGreater(res["d_orient"], 0.0)

    def test_candidate_regret(self):
        model = MockKoopmanModel()
        sys = RotationFamilyDynamics(a=0.5)
        x = torch.randn(4, 2)
        candidate_actions = torch.tensor([[-1.0], [0.0], [1.0]])
        regret = compute_candidate_regret(model, sys, x, candidate_actions, horizon=3)
        self.assertIsInstance(regret, float)
        self.assertGreaterEqual(regret, 0.0)

if __name__ == "__main__":
    unittest.main()

