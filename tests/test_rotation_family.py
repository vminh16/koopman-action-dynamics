# tests/test_rotation_family.py
import unittest
import torch
from research.benchmarks.rotation_family import RotationFamilyDynamics

class TestRotationFamily(unittest.TestCase):
    def test_a0_constant_input_direction(self):
        sys = RotationFamilyDynamics(a=0.0)
        self.assertEqual(sys.Nstates, 2)
        self.assertEqual(sys.Ncontrols, 1)
        x = torch.tensor([[0.5, -1.2], [2.0, 3.0]])
        Gx = sys.analytic_Gx(x)
        self.assertEqual(Gx.shape, (2, 2, 1))
        # When a=0, Gx must be exactly [[1.0], [0.0]] for all states
        expected = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        self.assertTrue(torch.allclose(Gx, expected, atol=1e-6))

    def test_rotation_parameter_sweep(self):
        x = torch.tensor([[1.0, 0.0]])
        for a in [0.0, 0.5, 1.0, 2.0]:
            sys = RotationFamilyDynamics(a=a)
            Gx = sys.analytic_Gx(x)
            expected_angle = a * 1.0
            expected_g = torch.tensor([[[torch.cos(torch.tensor(expected_angle))],
                                        [torch.sin(torch.tensor(expected_angle))]]])
            self.assertTrue(torch.allclose(Gx, expected_g, atol=1e-5))

    def test_control_affine_identity(self):
        sys = RotationFamilyDynamics(a=1.5)
        x = torch.randn(10, 2)
        u = torch.randn(10, 1)
        f_direct = sys.dynamics(0.0, x, u)
        f0 = sys.analytic_f0(x)
        Gx = sys.analytic_Gx(x)
        f_affine = f0 + torch.bmm(Gx, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_direct, f_affine, atol=1e-5))

    def test_call_signature_conventions(self):
        sys = RotationFamilyDynamics(a=1.0)
        x = torch.tensor([[1.0, 0.5]])
        u = torch.tensor([[0.2]])
        res1 = sys.dynamics(0.0, x, u)
        res2 = sys.dynamics(x, u)
        self.assertTrue(torch.allclose(res1, res2))

    def test_sample_initial_state(self):
        sys = RotationFamilyDynamics(a=1.0)
        x0 = sys.sample_initial_state(batch_size=5)
        self.assertEqual(x0.shape, (5, 2))

if __name__ == "__main__":
    unittest.main()

