import cvxpy as cp
import numpy as np
from scipy.linalg import expm
import torch


class KoopmanConvexMPC:
    """
    Linear Convex Model Predictive Controller for Koopman Operators.

    Optimizes:
        min_{Z, U} sum_{k=0}^{H-1} ( || C z_k - x_ref ||_Q^2 + || u_k ||_R^2 ) + || C z_H - x_ref ||_Qf^2
    subject to:
        z_0 = phi(x_init)
        z_{k+1} = A_d z_k + B_d u_k
        u_min <= u_k <= u_max
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        dt: float,
        horizon: int = 15,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        umin: np.ndarray = None,
        umax: np.ndarray = None,
        Qf: np.ndarray = None,
        encoder_fn=None,
    ):
        self.dt = float(dt)
        self.H = int(horizon)
        self.encoder_fn = encoder_fn

        # Dimensions
        self.dz = A.shape[0]
        self.du = B.shape[1]
        self.dx = C.shape[0]

        # Convert continuous (A, B) to discrete (Ad, Bd) using matrix exponential
        M = np.zeros((self.dz + self.du, self.dz + self.du), dtype=np.float64)
        M[: self.dz, : self.dz] = A
        M[: self.dz, self.dz :] = B
        e_M = expm(M * self.dt)
        self.Ad = np.ascontiguousarray(e_M[: self.dz, : self.dz])
        self.Bd = np.ascontiguousarray(e_M[: self.dz, self.dz :])
        self.C = np.ascontiguousarray(C, dtype=np.float64)

        # Cost weights
        self.Q = np.eye(self.dx) if Q is None else np.ascontiguousarray(Q, dtype=np.float64)
        self.R = 0.1 * np.eye(self.du) if R is None else np.ascontiguousarray(R, dtype=np.float64)
        self.Qf = self.Q if Qf is None else np.ascontiguousarray(Qf, dtype=np.float64)

        # Actuator bounds
        self.umin = None if umin is None else np.asarray(umin, dtype=np.float64).flatten()
        self.umax = None if umax is None else np.asarray(umax, dtype=np.float64).flatten()

        # Build parametric CVXPY problem
        self._build_problem()

    def _build_problem(self):
        self.Z = cp.Variable((self.dz, self.H + 1), name="Z")
        self.U = cp.Variable((self.du, self.H), name="U")

        self.z_init = cp.Parameter(self.dz, name="z_init")
        self.x_ref = cp.Parameter(self.dx, name="x_ref")

        cost = 0.0
        constraints = [self.Z[:, 0] == self.z_init]

        for k in range(self.H):
            # Stage tracking and control effort
            x_k = self.C @ self.Z[:, k]
            cost += cp.quad_form(x_k - self.x_ref, self.Q)
            cost += cp.quad_form(self.U[:, k], self.R)

            # Koopman discrete dynamics constraint
            constraints.append(
                self.Z[:, k + 1] == self.Ad @ self.Z[:, k] + self.Bd @ self.U[:, k]
            )

            # Actuator constraints
            if self.umin is not None:
                constraints.append(self.U[:, k] >= self.umin)
            if self.umax is not None:
                constraints.append(self.U[:, k] <= self.umax)

        # Terminal state cost
        x_term = self.C @ self.Z[:, self.H]
        cost += cp.quad_form(x_term - self.x_ref, self.Qf)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def lift_state(self, x: np.ndarray) -> np.ndarray:
        if self.encoder_fn is not None:
            with torch.no_grad():
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                z_t = self.encoder_fn(x_t).squeeze(0).cpu().numpy()
            return np.asarray(z_t, dtype=np.float64)
        else:
            if len(x) == self.dz:
                return np.asarray(x, dtype=np.float64)
            z = np.zeros(self.dz, dtype=np.float64)
            z[: len(x)] = x
            return z

    def solve(self, x_current: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, str, dict]:
        x_cur_np = np.asarray(x_current, dtype=np.float64).flatten()
        x_tgt_np = np.asarray(x_target, dtype=np.float64).flatten()

        z0 = self.lift_state(x_cur_np)
        self.z_init.value = z0
        self.x_ref.value = x_tgt_np

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            self.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)

        status = self.problem.status
        if status in ["optimal", "optimal_inaccurate"] and self.U.value is not None:
            u_opt = self.U.value[:, 0].copy()
            if self.umin is not None:
                u_opt = np.maximum(u_opt, self.umin)
            if self.umax is not None:
                u_opt = np.minimum(u_opt, self.umax)
            info = {
                "status": status,
                "cost": float(self.problem.value),
                "Z": self.Z.value.copy(),
                "U": self.U.value.copy(),
            }
        else:
            # Fallback zero control
            u_opt = np.zeros(self.du)
            info = {"status": status, "cost": np.nan, "Z": None, "U": None}

        return u_opt, status, info

    def step(self, x_current: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        u_opt, _, _ = self.solve(x_current, x_target)
        return u_opt

    def simulate_closed_loop(
        self,
        sys,
        x0: np.ndarray,
        x_target: np.ndarray,
        num_steps: int = 50,
    ) -> dict:
        x_cur = np.asarray(x0, dtype=np.float64).flatten()
        x_history = [x_cur.copy()]
        u_history = []
        cost_history = []

        cur_t = 0.0
        for _ in range(num_steps):
            u_t = self.step(x_cur, x_target)
            u_history.append(u_t)

            # Step true plant
            if hasattr(sys, "dynamics"):
                x_t_tensor = torch.from_numpy(x_cur).float().unsqueeze(0)
                u_t_tensor = torch.from_numpy(u_t).float().unsqueeze(0)
                dx = sys.dynamics(cur_t, x_t_tensor, u_t_tensor).squeeze(0).detach().cpu().numpy()
                x_next = x_cur + sys.dt * dx
            else:
                x_next = x_cur + self.dt * (x_cur @ sys.A.T + u_t @ sys.B.T)

            x_cur = x_next
            x_history.append(x_cur.copy())
            cur_t += self.dt

            err = x_cur - x_target
            step_cost = float(err @ self.Q @ err + u_t @ self.R @ u_t)
            cost_history.append(step_cost)

        return {
            "x_history": x_history,
            "u_history": u_history,
            "cost_history": cost_history,
            "total_cost": float(np.sum(cost_history)),
            "final_error": float(np.linalg.norm(x_history[-1] - x_target)),
        }
