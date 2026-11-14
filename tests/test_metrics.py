# tests/test_metrics.py
import unittest

import numpy as np
import torch

from research.benchmarks.rotation_family import RotationFamilyDynamics
from research.diagnostics.metrics import (
    build_candidate_pool,
    compute_action_response_error,
    compute_candidate_regret,
    compute_continuous_gramian_and_margin,
    compute_gramian_sensitivity,
    compute_h3_feasibility,
    compute_input_map_factorization_distance,
    compute_manifold_error,
    compute_vector_field_mismatches,
)
from research.diagnostics.run_p2_diagnostics import build_official_module


class MockKoopmanModel(torch.nn.Module):
    """Instantaneous latent dynamics z+ = A z + B u with identity head lift."""

    def __init__(self, state_dim=2, latent_dim=4, control_dim=1):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.A = torch.nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.B = torch.nn.Parameter(torch.zeros(latent_dim, control_dim))
        self.B.data[0, 0] = 1.0

    def encode(self, x):
        pad = torch.zeros(x.shape[0], self.latent_dim - self.state_dim, device=x.device)
        return torch.cat([x, pad], dim=-1)


def official_rotation_model(latent_dim=6, a=0.0):
    sys = RotationFamilyDynamics(a=a)
    module = build_official_module(sys, latent_dim=latent_dim)
    module.eval()
    return module, sys


class TestGramian(unittest.TestCase):
    def test_scalar_analytical_margin(self):
        # RRD v0.5 6.4: for scalar dz = a z + b u, C = 1, rho_T = |b| for all T.
        for a in (0.5, 2.0):
            res = compute_continuous_gramian_and_margin(
                np.array([[a]]), np.array([[2.0]]), np.array([[1.0]]), T=1.0
            )
            self.assertAlmostEqual(res["rho_T"], 2.0, places=6)
            self.assertFalse(res["rank_deficient"])
            self.assertEqual(res["numerical_rank"], 1)

    def test_zero_input_authority_is_rank_deficient_not_clipped(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.2]])
        B = np.array([[0.0], [0.0]])
        res = compute_continuous_gramian_and_margin(A, B, np.eye(2), T=1.0)
        self.assertTrue(res["rank_deficient"])
        self.assertIsNone(res["rho_T"], "must not report an artificial positive margin")
        self.assertEqual(res["numerical_rank"], 0)
        self.assertLess(res["lambda_min_Wy"], 1e-12)

    def test_partial_authority_is_rank_deficient(self):
        A = np.zeros((2, 2))
        B = np.array([[1.0], [0.0]])
        res = compute_continuous_gramian_and_margin(A, B, np.eye(2), T=1.0)
        self.assertTrue(res["rank_deficient"])
        self.assertEqual(res["numerical_rank"], 1)
        self.assertIsNone(res["rho_T"])

    def test_quadrature_resolution_sensitivity_is_reported(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.2]])
        B = np.array([[0.0], [1.0]])
        sweep = compute_gramian_sensitivity(
            A, B, np.eye(2), T_values=[0.5, 1.0], num_points_values=[100, 400]
        )
        self.assertEqual(len(sweep), 4)
        for row in sweep:
            self.assertIn("num_points", row)
            self.assertIn("T", row)
            self.assertGreater(row["rho_T"], 0.0)


class TestActionResponse(unittest.TestCase):
    def test_official_module_uses_instantaneous_dynamics(self):
        # Full-trajectory forward(x0, u_seq) must not be used as f_z(z, u).
        module, sys = official_rotation_model()
        x = torch.tensor([[0.3, -0.2], [-0.4, 0.1]])
        u = torch.tensor([[0.5], [-0.7]])
        u_ref = torch.zeros(2, 1)

        res = compute_action_response_error(module, sys, x, u, u_ref)
        A, B = module.model.operator.get_AB()
        C = torch.eye(2, A.shape[0])
        delta_model = (C @ B @ (u - u_ref).T).T
        delta_true = sys.dynamics(0.0, x, u) - sys.dynamics(0.0, x, u_ref)
        expected_raw = float(
        torch.mean(torch.sum((delta_model - delta_true) ** 2, dim=-1)).detach()
    )
        expected_norm = expected_raw / float(
            torch.mean(torch.sum(delta_true ** 2, dim=-1))
        )
        self.assertAlmostEqual(res["raw"], expected_raw, places=6)
        self.assertAlmostEqual(res["normalized"], expected_norm, places=5)
        self.assertIn("relative_error_mean", res)
        self.assertNotIn("nan", str(res).lower())

    def test_exact_model_has_zero_response_error(self):
        sys = RotationFamilyDynamics(a=0.0)
        module = torch.nn.Module()
        module.state_dim, module.latent_dim, module.control_dim = 2, 4, 1
        module.A = torch.nn.Parameter(torch.zeros(4, 4))
        module.B = torch.nn.Parameter(torch.tensor([[1.0], [0.0], [0.0], [0.0]]))
        module.encode = lambda x: torch.cat(
            [x, torch.zeros(x.shape[0], 2)], dim=-1
        )
        x = torch.randn(6, 2)
        u = torch.rand(6, 1) * 2 - 1
        res = compute_action_response_error(module, sys, x, u, torch.zeros(6, 1))
        self.assertLess(res["raw"], 1e-10)


class TestVectorFieldMismatches(unittest.TestCase):
    def test_orientation_mismatch_analytical(self):
        # identity-lift mock: B = e0.
        model = MockKoopmanModel()
        u_zero = torch.zeros(4, 1)
        sys0 = RotationFamilyDynamics(a=0.0)
        sys90 = RotationFamilyDynamics(a=np.pi / 2)
        x = torch.tensor([[1.0, 0.0]] * 4)
        # a=0: Gx = e0, Gz* = e0 -> d_orient = 0.
        res0 = compute_vector_field_mismatches(model, sys0, x)
        self.assertAlmostEqual(res0["d_orient"], 0.0, places=6)
        # a=pi/2, x1=1: Gx = e1, Gz* = e1 -> d_orient = ||B_perp|| = 1.
        res90 = compute_vector_field_mismatches(model, sys90, x)
        self.assertAlmostEqual(res90["d_orient"], 1.0, places=6)

    def test_orientation_metric_is_a_norm(self):
        sys = RotationFamilyDynamics(a=1.0)
        x = torch.randn(5, 2)
        m1 = MockKoopmanModel()
        m2 = MockKoopmanModel()
        with torch.no_grad():
            m2.B.mul_(3.0)
        r1 = compute_vector_field_mismatches(m1, sys, x)
        r2 = compute_vector_field_mismatches(m2, sys, x)
        # d_orient = ||(I - Pi) B|| is a norm, so it scales with ||B||. The
        # dimensionless orientation ratio is what characterises direction.
        self.assertAlmostEqual(r2["d_orient"], 3.0 * r1["d_orient"], places=5)
        # E_lift = ||B - J_phi G_x||_F is affine in B, not linear.
        expected = []
        for xi in x:
            gz = torch.zeros(4, 1)
            gz[:2] = sys.analytic_Gx(xi.unsqueeze(0)).squeeze(0)
            expected.append(float(torch.norm(3.0 * torch.tensor([[1.0], [0.0], [0.0], [0.0]]) - gz, p="fro")))
        self.assertAlmostEqual(r2["E_lift"], float(np.mean(expected)), places=5)

    def test_percentiles_and_drift_are_reported(self):
        model = MockKoopmanModel()
        sys = RotationFamilyDynamics(a=1.0)
        res = compute_vector_field_mismatches(model, sys, torch.randn(7, 2))
        for key in ("E_lift", "E_phys", "E_drift", "d_orient"):
            self.assertIn(key, res)
            self.assertIn(f"{key}_p95", res)
            self.assertGreaterEqual(res[f"{key}_p95"], res[key])

    def test_manifold_error_is_zero_on_manifold(self):
        module, _ = official_rotation_model()
        x = torch.randn(5, 2)
        z = module.model.encoder(x)
        err = compute_manifold_error(module, z)
        self.assertLess(err["mean"], 1e-6)
        z_off = z.clone()
        z_off[:, 2:] += 0.5
        err_off = compute_manifold_error(module, z_off)
        self.assertGreater(err_off["mean"], 1e-3)


class TestFactorisationAndRegret(unittest.TestCase):
    def test_factorization_distance_bounds_orientation(self):
        sys = RotationFamilyDynamics(a=np.pi / 2)
        x = torch.tensor([[1.0, 0.0]])
        target = sys.analytic_Gx(x).squeeze(0)  # G* in physical space e1
        B = np.array([[1.0], [0.0]])
        res = compute_input_map_factorization_distance(
            target, B, sigma_min=0.5, sigma_max=2.0
        )
        # B = e0, G* = e1: no invertible scalar M can align them -> d* >= d_orient = 1.
        self.assertGreaterEqual(res["d_star"], 1.0 - 1e-6)

    def test_candidate_regret_official_module(self):
        module, sys = official_rotation_model()
        x = torch.tensor([[0.2, 0.1], [-0.3, 0.2]])
        actions = torch.tensor([[-1.0], [0.0], [1.0]])
        res = compute_candidate_regret(module, sys, x, actions, horizon=3)
        self.assertIsInstance(res, dict)
        self.assertGreaterEqual(res["mean"], 0.0)
        self.assertIn("p95", res)


    def test_factorization_distance_allows_negative_scaling(self):
        # sigma(M)=|M|; M=-1 is feasible and aligns G=-B exactly.
        B = np.array([[1.0], [0.0]])
        res = compute_input_map_factorization_distance(
            -B, B, sigma_min=0.5, sigma_max=2.0
        )
        self.assertLess(res["d_star"], 1e-12)
        self.assertAlmostEqual(res["M"], -1.0, places=9)
        self.assertTrue(res["feasible"])

    def test_factorization_multi_reports_lower_and_upper_bounds(self):
        G = np.array([[1.0, 0.0], [0.0, 1.0]])
        B_feasible = 0.5 * np.eye(2)  # M = 2 I, sigma in [0.5, 2]
        ok = compute_input_map_factorization_distance(
            G, B_feasible, sigma_min=0.5, sigma_max=2.0
        )
        self.assertAlmostEqual(ok["d_star"], 0.0, places=9)
        self.assertAlmostEqual(ok["upper_bound"], 0.0, places=9)

        B_bad = np.diag([1.0, 0.1])
        res = compute_input_map_factorization_distance(
            G, B_bad, sigma_min=0.5, sigma_max=2.0
        )
        self.assertIn("lower_bound", res)
        self.assertIn("upper_bound", res)
        self.assertIsInstance(res["feasible"], bool)
        self.assertGreaterEqual(res["d_star"], res["lower_bound"] - 1e-9)

    def test_candidate_pool_is_valid_and_axis_exciting(self):
        pool = build_candidate_pool(
            torch.tensor([-1.0, -0.5]), torch.tensor([1.0, 0.5]),
            horizon=10, num_sequences=8, seed=0,
        )
        self.assertEqual(pool.shape[0], 8)
        self.assertTrue(bool((pool[..., 0] >= -1.0).all()))
        self.assertTrue(bool((pool[..., 0] <= 1.0).all()))
        self.assertTrue(bool((pool[..., 1] >= -0.5).all()))
        self.assertTrue(bool((pool[..., 1] <= 0.5).all()))
        # Time variation and independent excitation on both axes.
        self.assertGreater(float(pool[..., 0].std(dim=1).max()), 0.0)
        self.assertGreater(float(pool[..., 1].std(dim=1).max()), 0.0)
        # At least one sequence changes sign on each axis.
        self.assertTrue(bool((pool[..., 0].max(dim=1).values > 0).any()))
        self.assertTrue(bool((pool[..., 1].max(dim=1).values > 0).any()))

    def test_candidate_regret_scores_time_varying_pool(self):
        module, sys = official_rotation_model()
        x = torch.tensor([[0.2, 0.1], [-0.3, 0.2]])
        pool = build_candidate_pool(sys.umin, sys.umax, horizon=3, num_sequences=8, seed=1)
        res = compute_candidate_regret(
            module, sys, x, candidate_sequences=pool, horizon=3,
            target=torch.zeros(2),
        )
        self.assertEqual(res["num_candidates"], 8)
        self.assertIn("exact ZOH", res["model_rollout"])
        self.assertNotIn("reencode", res)
        res_re = compute_candidate_regret(
            module, sys, x, candidate_sequences=pool, horizon=3, reencode=True
        )
        self.assertIn("reencode", res_re)
        self.assertGreaterEqual(res_re["reencode"]["mean"], 0.0)

    def test_h3_reports_orientation_max_and_p95_separately(self):
        module, sys = official_rotation_model()
        x = torch.randn(6, 2)
        A, B = module.model.operator.get_AB()
        res = compute_h3_feasibility(module, sys, x, rho_T=0.05)
        self.assertIn("d_orient_max", res)
        self.assertIn("d_orient_p95", res)
        self.assertGreaterEqual(res["d_orient_max"] + 1e-9, res["d_orient_p95"])
        self.assertEqual(
            res["d_orient_p95_below_rho"], bool(res["d_orient_p95"] < 0.05)
        )
        self.assertEqual(
            res["d_orient_max_below_rho"], bool(res["d_orient_max"] < 0.05)
        )
        self.assertIn("d_star_lower_bound_p95", res)
        self.assertIn("factorization_solver_status", res)


class TestGramianAnalytic(unittest.TestCase):
    def test_expm_matches_scalar_analytic_formula(self):
        for a, b, T in ((0.7, 1.3, 0.4), (-0.3, 2.0, 1.0), (0.0, 0.5, 0.8)):
            res = compute_continuous_gramian_and_margin(
                np.array([[a]]), np.array([[b]]), np.array([[1.0]]), T=T, method="expm"
            )
            integral = T if abs(a) < 1e-12 else (np.exp(2 * a * T) - 1.0) / (2 * a)
            self.assertAlmostEqual(res["lambda_min_Wy"], b * b * integral, places=10)
            self.assertAlmostEqual(res["lambda_max_Qy"], integral, places=10)
            self.assertAlmostEqual(res["rho_T"], abs(b), places=10)

    def test_quadrature_converges_to_expm(self):
        A = np.array([[0.2, 1.0], [-1.0, -0.4]])
        B = np.array([[0.0], [1.0]])
        exact = compute_continuous_gramian_and_margin(A, B, np.eye(2), T=0.9, method="expm")
        fine = compute_continuous_gramian_and_margin(
            A, B, np.eye(2), T=0.9, method="quadrature", num_points=4001
        )
        self.assertAlmostEqual(fine["rho_T"], exact["rho_T"], places=5)
        self.assertAlmostEqual(fine["lambda_min_Wy"], exact["lambda_min_Wy"], places=5)


if __name__ == "__main__":
    unittest.main()
