# tests/test_qp_mpc.py
import unittest
from unittest import mock

import numpy as np
import torch
from scipy.linalg import expm

from research.baseline.data_generator import rk4_step
from research.control.qp_mpc import KoopmanConvexMPC


class DummySystem:
    """Linear plant usable by the MPC simulation."""

    def __init__(self, A, B, dt=0.05):
        self.A = A
        self.B = B
        self.dt = dt
        self.Nstates = A.shape[0]
        self.Ncontrols = B.shape[1]

    def dynamics(self, t, x, u):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            u_np = u.detach().cpu().numpy()
            dx = x_np @ self.A.T + u_np @ self.B.T
            return torch.from_numpy(dx).float()
        return x @ self.A.T + u @ self.B.T


class NonlinearSystem:
    """xdot = [x2, -sin(x1) + u]; Euler and RK4 differ measurably."""

    def __init__(self, dt=0.05):
        self.dt = dt
        self.Nstates = 2
        self.Ncontrols = 1

    def dynamics(self, t, x, u):
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = x2
        dx2 = -torch.sin(x1) + u[..., 0]
        return torch.stack([dx1, dx2], dim=-1)


def double_integrator_mpc(**kwargs):
    params = dict(
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        C=np.eye(2),
        dt=0.05,
        horizon=20,
        Q=np.diag([10.0, 1.0]),
        R=np.array([[0.1]]),
        umin=np.array([-1.5]),
        umax=np.array([1.5]),
    )
    params.update(kwargs)
    return KoopmanConvexMPC(**params)


class TestQPMPC(unittest.TestCase):
    def test_double_integrator_stabilization(self):
        mpc = double_integrator_mpc()
        u_opt, status, info = mpc.solve(np.array([2.0, 0.0]), np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        self.assertTrue(info["success"])
        self.assertFalse(info["fallback_used"])
        self.assertLess(u_opt[0], 0.0)
        self.assertTrue(-1.5 <= u_opt[0] <= 1.5)
        self.assertLessEqual(info["max_input_violation"], 1e-6)
        self.assertGreaterEqual(info["solve_time_s"], 0.0)
        self.assertGreaterEqual(info["iterations"], 0)

    def test_problem_is_convex_dcp(self):
        mpc = double_integrator_mpc()
        self.assertTrue(mpc.problem.is_dcp())
        self.assertIn(mpc.problem.status, [None, "optimal"])

    def test_actuator_saturation_enforcement(self):
        mpc = double_integrator_mpc(horizon=10, umin=np.array([-0.2]), umax=np.array([0.2]))
        u_opt, status, info = mpc.solve(np.array([100.0, 50.0]), np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        self.assertAlmostEqual(u_opt[0], -0.2, places=4)
        self.assertLessEqual(info["max_input_violation"], 1e-6)

    def test_discretization_matches_matrix_exponential(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B = np.array([[0.0], [1.0]])
        dt = 0.05
        mpc = KoopmanConvexMPC(A=A, B=B, C=np.eye(2), dt=dt)
        M = np.zeros((3, 3))
        M[:2, :2] = A
        M[:2, 2:] = B
        E = expm(M * dt)
        self.assertTrue(np.allclose(mpc.Ad, E[:2, :2], atol=1e-12))
        self.assertTrue(np.allclose(mpc.Bd, E[:2, 2:], atol=1e-12))
        euler = KoopmanConvexMPC(A=A, B=B, C=np.eye(2), dt=dt, discretization="euler")
        self.assertTrue(np.allclose(euler.Ad, np.eye(2) + dt * A, atol=1e-12))
        self.assertTrue(np.allclose(euler.Bd, dt * B, atol=1e-12))

    def test_fallback_is_valid_when_zero_is_outside_bounds(self):
        mpc = double_integrator_mpc(umin=np.array([0.5]), umax=np.array([1.5]))
        with mock.patch("cvxpy.Problem.solve", side_effect=RuntimeError("boom")):
            u_opt, status, info = mpc.solve(np.array([1.0, 0.0]), np.zeros(2))
        self.assertFalse(info["success"])
        self.assertTrue(info["fallback_used"])
        self.assertGreaterEqual(u_opt[0], 0.5 - 1e-12)
        self.assertLessEqual(u_opt[0], 1.5 + 1e-12)
        self.assertLessEqual(info["max_input_violation"], 1e-12)
        self.assertEqual(info["fallback_command"], u_opt.tolist())

    def test_residual_above_threshold_triggers_logged_fallback(self):
        mpc = double_integrator_mpc(residual_tolerance=-1.0)
        u_opt, status, info = mpc.solve(np.array([1.0, 0.0]), np.zeros(2))
        if info["primal_residual"] is None:
            self.skipTest("OSQP did not report a primal residual")
        self.assertTrue(info["residual_exceeded"])
        self.assertTrue(info["fallback_used"])
        self.assertEqual(info["fallback_reason"], "residual_tolerance")
        self.assertFalse(info["success"])
        self.assertLessEqual(info["max_input_violation"], 1e-12)

    def test_solver_failure_is_reported_not_hidden(self):
        mpc = double_integrator_mpc()
        with mock.patch("cvxpy.Problem.solve", side_effect=RuntimeError("boom")):
            u_opt, status, info = mpc.solve(np.array([1.0, 0.0]), np.zeros(2))
        self.assertEqual(status, "solver_error")
        self.assertFalse(info["success"])
        self.assertTrue(info["fallback_used"])
        self.assertEqual(info["fallback_reason"], "exception")
        self.assertTrue(np.allclose(u_opt, 0.0))

    def test_plant_integration_uses_rk4_with_benchmark_dt(self):
        sys = NonlinearSystem(dt=0.05)
        mpc = KoopmanConvexMPC(
            A=np.array([[0.0, 1.0], [-1.0, -0.1]]),
            B=np.array([[0.0], [1.0]]),
            C=np.eye(2),
            dt=0.05,
            horizon=5,
        )
        x0 = np.array([1.0, 0.0])
        with mock.patch.object(mpc, "solve", return_value=(np.zeros(1), "optimal", {})):
            traj = mpc.simulate_closed_loop(sys, x0=x0, x_target=np.zeros(2), num_steps=1)
        expected = rk4_step(
            sys.dynamics, 0.0, torch.tensor(x0, dtype=torch.float32).unsqueeze(0),
            torch.zeros(1, 1), sys.dt,
        ).squeeze(0).numpy()
        self.assertTrue(np.allclose(traj["x_history"][1], expected, atol=1e-6),
                        f"plant step {traj['x_history'][1]} != RK4 {expected}")

    def test_dt_mismatch_is_rejected(self):
        mpc = double_integrator_mpc(dt=0.05)
        sys = DummySystem(np.zeros((2, 2)), np.zeros((2, 1)), dt=0.02)
        with self.assertRaises(ValueError):
            mpc.simulate_closed_loop(sys, x0=np.zeros(2), x_target=np.zeros(2), num_steps=2)

    def test_closed_loop_simulation(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B = np.array([[0.0], [1.0]])
        sys = DummySystem(A, B, dt=0.05)
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=np.eye(2), dt=0.05, horizon=15,
            Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]),
            umin=np.array([-2.0]), umax=np.array([2.0]),
        )
        traj = mpc.simulate_closed_loop(
            sys, x0=np.array([1.0, 0.0]), x_target=np.zeros(2), num_steps=25
        )
        self.assertEqual(len(traj["x_history"]), 26)
        self.assertEqual(len(traj["u_history"]), 25)
        self.assertLess(np.linalg.norm(traj["x_history"][-1]), np.linalg.norm(traj["x_history"][0]))
        self.assertEqual(traj["solver"]["num_failures"], 0)
        self.assertGreaterEqual(traj["solver"]["success_rate"], 0.99)
        self.assertIn("latency_median_s", traj["solver"])
        self.assertIn("latency_p95_s", traj["solver"])
        self.assertIn("max_input_violation", traj["solver"])


if __name__ == "__main__":
    unittest.main()
