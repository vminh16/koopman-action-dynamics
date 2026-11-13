# tests/test_unicycle.py
import unittest
import torch
import numpy as np
from research.benchmarks.unicycle import UnicycleDynamics

class TestUnicycle(unittest.TestCase):
    def test_dimensions_and_controls(self):
        sys = UnicycleDynamics()
        self.assertEqual(sys.Nstates, 3)
        self.assertEqual(sys.Ncontrols, 2)
        self.assertEqual(sys.umin.shape, (2,))
        self.assertEqual(sys.umax.shape, (2,))

    def test_heading_dependent_velocity(self):
        sys = UnicycleDynamics()
        # Test heading 0 -> moves in x
        x_east = torch.tensor([[0.0, 0.0, 0.0]])
        u = torch.tensor([[1.0, 0.0]]) # v=1, omega=0
        dx_east = sys.dynamics(x_east, u)
        self.assertTrue(torch.allclose(dx_east, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5))

        # Test heading pi/2 -> moves in y
        x_north = torch.tensor([[0.0, 0.0, np.pi / 2.0]], dtype=torch.float32)
        dx_north = sys.dynamics(x_north, u)
        self.assertTrue(torch.allclose(dx_north, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5))

        # Test pure rotation
        u_rot = torch.tensor([[0.0, 0.5]]) # v=0, omega=0.5
        dx_rot = sys.dynamics(x_east, u_rot)
        self.assertTrue(torch.allclose(dx_rot, torch.tensor([[0.0, 0.0, 0.5]]), atol=1e-5))

    def test_control_affine_identity(self):
        sys = UnicycleDynamics()
        x = torch.randn(10, 3)
        u = torch.randn(10, 2)
        f_direct = sys.dynamics(0.0, x, u)
        f0 = sys.analytic_f0(x)
        Gx = sys.analytic_Gx(x)
        f_affine = f0 + torch.bmm(Gx, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_direct, f_affine, atol=1e-5))

    def test_call_signatures(self):
        sys = UnicycleDynamics()
        x = torch.tensor([[1.0, 2.0, 0.5]])
        u = torch.tensor([[0.8, -0.2]])
        res1 = sys.dynamics(0.0, x, u)
        res2 = sys.dynamics(x, u)
        self.assertTrue(torch.allclose(res1, res2))

    def test_no_inplace_mutation(self):
        sys = UnicycleDynamics()
        x = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)
        u = torch.tensor([[0.5, 0.1]])
        dx = sys.dynamics(x, u)
        loss = dx.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_sample_initial_state(self):
        sys = UnicycleDynamics()
        x0 = sys.sample_initial_state(batch_size=8)
        self.assertEqual(x0.shape, (8, 3))
        # Ensure angles are within [-pi, pi]
        self.assertTrue((x0[:, 2] >= -np.pi).all())
        self.assertTrue((x0[:, 2] <= np.pi).all())

if __name__ == "__main__":
    unittest.main()

