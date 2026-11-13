# tests/test_qp_mpc.py
import unittest
import numpy as np
import torch
from research.control.qp_mpc import KoopmanConvexMPC

class DummySystem:
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

class TestQPMPC(unittest.TestCase):
    def test_double_integrator_stabilization(self):
        # LTI toy system: x_dot = [x2, u]
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        C = np.eye(2)
        dt = 0.05
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=C, dt=dt, horizon=20,
            Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]),
            umin=np.array([-1.5]), umax=np.array([1.5])
        )
        x0 = np.array([2.0, 0.0])
        u_opt, status, _ = mpc.solve(x0, x_target=np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        # Control must brake/accelerate towards origin (u < 0)
        self.assertLess(u_opt[0], 0.0)
        # Actuator bounds strictly respected
        self.assertTrue(-1.5 <= u_opt[0] <= 1.5)

    def test_actuator_saturation_enforcement(self):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        C = np.eye(2)
        # Narrow bounds [-0.2, 0.2]
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=C, dt=0.05, horizon=10,
            umin=np.array([-0.2]), umax=np.array([0.2])
        )
        x_far = np.array([100.0, 50.0]) # huge error
        u_opt, status, _ = mpc.solve(x_far, x_target=np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        self.assertAlmostEqual(u_opt[0], -0.2, places=4)

    def test_closed_loop_simulation(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B = np.array([[0.0], [1.0]])
        C = np.eye(2)
        dt = 0.05
        sys = DummySystem(A, B, dt)
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=C, dt=dt, horizon=15,
            Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]),
            umin=np.array([-2.0]), umax=np.array([2.0])
        )
        traj = mpc.simulate_closed_loop(sys, x0=np.array([1.0, 0.0]), x_target=np.zeros(2), num_steps=25)
        self.assertEqual(len(traj["x_history"]), 26)
        self.assertEqual(len(traj["u_history"]), 25)
        self.assertLess(np.linalg.norm(traj["x_history"][-1]), np.linalg.norm(traj["x_history"][0]))

if __name__ == "__main__":
    unittest.main()
