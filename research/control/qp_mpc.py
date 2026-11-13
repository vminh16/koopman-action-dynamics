import time

import cvxpy as cp
import numpy as np
import torch
from scipy.linalg import expm

from research.baseline.data_generator import rk4_step


class KoopmanConvexMPC:
    """
    Linear Convex Model Predictive Controller for Koopman Operators.

    Optimizes:
        min_{Z, U} sum_{k=0}^{H-1} ( || C z_k - x_ref ||_Q^2 + || u_k ||_R^2 ) + || C z_H - x_ref ||_Qf^2
    subject to:
        z_0 = phi(x_init)
        z_{k+1} = A_d z_k + B_d u_k
        u_min <= u_k <= u_max

    Solver regime: *exact baseline QP* for the lifted LTI surrogate. It is not an
    exact nonlinear MPC for a state-dependent plant.
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
        # (exact zero-order hold for the lifted LTI model).
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
            x_k = self.C @ self.Z[:, k]
            cost += cp.quad_form(x_k - self.x_ref, self.Q)
            cost += cp.quad_form(self.U[:, k], self.R)

            constraints.append(
                self.Z[:, k + 1] == self.Ad @ self.Z[:, k] + self.Bd @ self.U[:, k]
            )

            if self.umin is not None:
                constraints.append(self.U[:, k] >= self.umin)
            if self.umax is not None:
                constraints.append(self.U[:, k] <= self.umax)

        x_term = self.C @ self.Z[:, self.H]
        cost += cp.quad_form(x_term - self.x_ref, self.Qf)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def lift_state(self, x: np.ndarray) -> np.ndarray:
        if self.encoder_fn is not None:
            with torch.no_grad():
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                z_t = self.encoder_fn(x_t).squeeze(0).cpu().numpy()
            return np.asarray(z_t, dtype=np.float64)
        if len(x) == self.dz:
            return np.asarray(x, dtype=np.float64)
        z = np.zeros(self.dz, dtype=np.float64)
        z[: len(x)] = x
        return z

    def _input_violation(self, u: np.ndarray) -> float:
        violation = 0.0
        if self.umin is not None:
            violation = max(violation, float(np.max(self.umin - u)))
        if self.umax is not None:
            violation = max(violation, float(np.max(u - self.umax)))
        return max(0.0, violation)

    def solve(self, x_current: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, str, dict]:
        x_cur_np = np.asarray(x_current, dtype=np.float64).flatten()
        x_tgt_np = np.asarray(x_target, dtype=np.float64).flatten()

        self.z_init.value = self.lift_state(x_cur_np)
        self.x_ref.value = x_tgt_np

        fallback_used = False
        fallback_reason = None
        t0 = time.perf_counter()
        exception_text = None
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as exc:  # noqa: BLE001 - solver-specific failures
            exception_text = repr(exc)
            try:
                self.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)
            except Exception as exc2:  # noqa: BLE001
                exception_text = repr(exc2)
        solve_time = time.perf_counter() - t0
        status = self.problem.status

        stats = getattr(self.problem, "solver_stats", None)
        iterations = int(getattr(stats, "num_iters", 0) or 0)
        extra = getattr(stats, "extra_stats", None) or {}

        if status in ("optimal", "optimal_inaccurate") and self.U.value is not None:
            u_raw = np.asarray(self.U.value[:, 0], dtype=np.float64).copy()
            violation = self._input_violation(u_raw)
            # Remove only floating-point noise; do not hide a real bound violation.
            u_opt = u_raw.copy()
            if violation <= 1e-7:
                if self.umin is not None:
                    u_opt = np.maximum(u_opt, self.umin)
                if self.umax is not None:
                    u_opt = np.minimum(u_opt, self.umax)
                violation = self._input_violation(u_opt)
            success = True
            cost = float(self.problem.value)
        else:
            fallback_used = True
            fallback_reason = "exception" if exception_text else str(status)
            if exception_text:
                fallback_reason = "exception"
            u_opt = np.zeros(self.du)
            violation = 0.0
            cost = float("nan")
            success = False

        info = {
            "status": status if status is not None else "solver_error",
            "success": success,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "exception": exception_text,
            "cost": cost,
            "solve_time_s": float(solve_time),
            "iterations": iterations,
            "primal_residual": _extra_stat(extra, "pri_res"),
            "dual_residual": _extra_stat(extra, "dua_res"),
            "max_input_violation": float(violation),
            "Z": self.Z.value.copy() if self.Z.value is not None else None,
            "U": self.U.value.copy() if self.U.value is not None else None,
        }
        return u_opt, info["status"], info

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
        plant_dt = float(getattr(sys, "dt", self.dt))
        if not np.isclose(plant_dt, self.dt, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"MPC dt ({self.dt}) must match plant dt ({plant_dt}); "
                "training, prediction and closed-loop integration must use the "
                "same benchmark time step"
            )

        x_cur = np.asarray(x0, dtype=np.float64).flatten()
        x_history = [x_cur.copy()]
        u_history = []
        cost_history = []
        solver_info = []

        cur_t = 0.0
        for _ in range(num_steps):
            u_t, _, info = self.solve(x_cur, x_target)
            solver_info.append(
                {
                    "status": info.get("status", "unknown"),
                    "success": bool(info.get("success", False)),
                    "fallback_used": bool(info.get("fallback_used", False)),
                    "solve_time_s": float(info.get("solve_time_s", 0.0)),
                    "iterations": int(info.get("iterations", 0) or 0),
                    "primal_residual": info.get("primal_residual"),
                    "dual_residual": info.get("dual_residual"),
                    "max_input_violation": float(info.get("max_input_violation", 0.0)),
                }
            )
            u_history.append(u_t)

            x_tensor = torch.from_numpy(x_cur).float().unsqueeze(0)
            u_tensor = torch.from_numpy(u_t).float().unsqueeze(0)
            if hasattr(sys, "dynamics"):
                x_next = (
                    rk4_step(sys.dynamics, cur_t, x_tensor, u_tensor, self.dt)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:  # discrete LTI fallback (legacy)
                x_next = x_cur + self.dt * (x_cur @ sys.A.T + u_t @ sys.B.T)

            x_cur = x_next
            x_history.append(x_cur.copy())
            cur_t += self.dt

            err = x_cur - x_target
            cost_history.append(float(err @ self.Q @ err + u_t @ self.R @ u_t))

        latencies = np.array([s["solve_time_s"] for s in solver_info])
        violations = np.array([s["max_input_violation"] for s in solver_info])
        successes = np.array([s["success"] for s in solver_info])
        pri = [s["primal_residual"] for s in solver_info if s["primal_residual"] is not None]
        dua = [s["dual_residual"] for s in solver_info if s["dual_residual"] is not None]
        return {
            "x_history": x_history,
            "u_history": u_history,
            "cost_history": cost_history,
            "total_cost": float(np.sum(cost_history)),
            "final_error": float(np.linalg.norm(x_history[-1] - x_target)),
            "solver": {
                "num_steps": int(num_steps),
                "num_failures": int(np.sum(~successes)),
                "success_rate": float(np.mean(successes)),
                "num_fallbacks": int(sum(s["fallback_used"] for s in solver_info)),
                "latency_median_s": float(np.median(latencies)),
                "latency_p95_s": float(np.percentile(latencies, 95)),
                "max_iterations": int(max(s["iterations"] for s in solver_info)),
                "max_input_violation": float(violations.max()),
                "primal_residual_max": float(np.max(pri)) if pri else None,
                "dual_residual_max": float(np.max(dua)) if dua else None,
                "statuses": [s["status"] for s in solver_info],
            },
        }


def _extra_stat(extra, key):
    try:
        value = extra.get(key)
    except AttributeError:
        return None
    return None if value is None else float(value)
