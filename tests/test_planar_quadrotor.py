# tests/test_planar_quadrotor.py
import unittest
import torch
import numpy as np
from research.benchmarks.planar_quadrotor import PlanarQuadrotorDynamics
from research.benchmarks import BENCHMARK_REGISTRY, load_benchmark

class TestPlanarQuadrotor(unittest.TestCase):
    def test_dimensions_and_hover_equilibrium(self):
        sys = PlanarQuadrotorDynamics()
        self.assertEqual(sys.Nstates, 6)
        self.assertEqual(sys.Ncontrols, 2)
        # Equilibrium state: origin at rest
        x_hover = torch.zeros((1, 6))
        # Hover thrust = m * g / 2 each
        u_hover = torch.tensor([[sys.m * sys.g / 2.0, sys.m * sys.g / 2.0]])
        dx = sys.dynamics(x_hover, u_hover)
        self.assertTrue(torch.allclose(dx, torch.zeros((1, 6)), atol=1e-5))

    def test_orientation_dependent_lateral_acceleration(self):
        sys = PlanarQuadrotorDynamics()
        # When tilted right (phi > 0), positive thrust pushes laterally in negative y direction:
        # y_ddot = -(u1+u2)/m * sin(phi)
        phi = np.pi / 6.0 # 30 deg
        x_tilted = torch.tensor([[0.0, 1.0, phi, 0.0, 0.0, 0.0]], dtype=torch.float32)
        u_thrust = torch.tensor([[5.0, 5.0]]) # 10 N total
        dx = sys.dynamics(x_tilted, u_thrust)
        expected_y_ddot = -(10.0 / sys.m) * np.sin(phi)
        self.assertAlmostEqual(dx[0, 3].item(), expected_y_ddot, places=4)

    def test_differential_thrust_angular_acceleration(self):
        sys = PlanarQuadrotorDynamics()
        x0 = torch.zeros((1, 6))
        # Differential thrust: u2 > u1 -> positive torque -> phi_ddot > 0
        u = torch.tensor([[2.0, 4.0]]) # diff = 2.0
        dx = sys.dynamics(x0, u)
        expected_phi_ddot = sys.L * (4.0 - 2.0) / sys.I_yy
        self.assertAlmostEqual(dx[0, 5].item(), expected_phi_ddot, places=4)

    def test_control_affine_identity(self):
        sys = PlanarQuadrotorDynamics()
        x = torch.randn(10, 6)
        u = torch.rand(10, 2) * 10.0
        f_direct = sys.dynamics(0.0, x, u)
        f0 = sys.analytic_f0(x)
        Gx = sys.analytic_Gx(x)
        f_affine = f0 + torch.bmm(Gx, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_direct, f_affine, atol=1e-5))

    def test_sample_initial_state(self):
        sys = PlanarQuadrotorDynamics()
        x0 = sys.sample_initial_state(batch_size=7)
        self.assertEqual(x0.shape, (7, 6))

    def test_registry_loader(self):
        sys_dict = load_benchmark("planar_quadrotor")
        self.assertEqual(sys_dict["Nstates"], 6)
        self.assertEqual(sys_dict["Ncontrols"], 2)
        self.assertIn("rotation_family", BENCHMARK_REGISTRY)
        self.assertIn("unicycle", BENCHMARK_REGISTRY)
        self.assertIn("planar_quadrotor", BENCHMARK_REGISTRY)

if __name__ == "__main__":
    unittest.main()

