"""P2 diagnostic metrics, following the definitions in EXPERIMENT.md §4 and RRD v0.5 §6.

Numerical conventions
---------------------
* Gramians are computed exactly (up to ``scipy.linalg.expm``) with a Van Loan
  block-exponential in float64. A trapezoidal quadrature implementation is
  retained only to report integration-resolution sensitivity.
* A Gramian whose numerical rank is below the output dimension is reported as
  ``rank_deficient`` with ``rho_T = None``. It is never clipped to a positive
  eigenvalue: that would fabricate a controllability margin.
* ``d_star`` is the pointwise constrained factorization distance from RRD §6.6.
  It is solved exactly for single-input systems; for multi-input systems it is a
  documented numerical approximation (SLSQP), not a global feasibility proof.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar

DEFAULT_RANK_TOL = 1e-10
EPS = 1e-12


# ---------------------------------------------------------------------------
# model adapters
# ---------------------------------------------------------------------------
def extract_latent_model(model) -> dict:
    """Return A, B, encoder and the *instantaneous* latent dynamics.

    ``KoopmanLitModule.forward`` integrates a full trajectory; it must never be
    used as ``f_z(z, u)``. This adapter uses ``model.model.dynamics`` (the ODE
    right-hand side) and the operator's transformed (A, B).
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "operator"):
        A, B = inner.operator.get_AB()
        state_dim = int(getattr(inner, "state_dim", B.shape[0]))
        return {
            "A": A,
            "B": B,
            "encode": inner.encoder,
            "instant": lambda z, u: inner.dynamics(0.0, z, u),
            "state_dim": state_dim,
            "latent_dim": int(A.shape[0]),
        }
    if all(hasattr(model, key) for key in ("A", "B", "encode")):
        A, B = model.A, model.B
        state_dim = int(getattr(model, "state_dim", B.shape[0]))
        return {
            "A": A,
            "B": B,
            "encode": model.encode,
            "instant": lambda z, u: z @ A.T + u @ B.T,
            "state_dim": state_dim,
            "latent_dim": int(A.shape[0]),
        }
    raise AttributeError("Unrecognized model structure for latent diagnostics")


# ---------------------------------------------------------------------------
# controllability diagnostics
# ---------------------------------------------------------------------------
def _van_loan_gramian(A: np.ndarray, Q: np.ndarray, T: float) -> np.ndarray:
    """Exact integral int_0^T exp(As) Q exp(A^T s) ds."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n), dtype=np.float64)
    M[:n, :n] = -A
    M[:n, n:] = Q
    M[n:, n:] = A.T
    E = expm(M * T)
    W = E[n:, n:].T @ E[:n, n:]
    return 0.5 * (W + W.T)


def _quadrature_gramians(A, B, C, T, num_points):
    t_grid = np.linspace(0.0, float(T), int(num_points))
    dt = t_grid[1] - t_grid[0]
    ny = C.shape[0]
    Wy = np.zeros((ny, ny))
    Qy = np.zeros((ny, ny))
    for i, s in enumerate(t_grid):
        weight = dt if (i > 0 and i < num_points - 1) else 0.5 * dt
        e_As = expm(A * s)
        M_w = C @ e_As @ B
        Wy += weight * (M_w @ M_w.T)
        M_q = C @ e_As
        Qy += weight * (M_q @ M_q.T)
    return Wy, Qy


def compute_continuous_gramian_and_margin(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    T: float,
    num_points: int = 200,
    rank_tol: float = DEFAULT_RANK_TOL,
    method: str = "expm",
) -> dict:
    """Output reachability/observability Gramians and the RRD §6.4 margin.

    rho_T = sqrt(lambda_min(W_y(T))) / sqrt(lambda_max(Q_y(T))).
    Returns ``rho_T=None`` whenever the Gramian is numerically rank deficient.
    """
    A_64 = np.asarray(A, dtype=np.float64)
    B_64 = np.asarray(B, dtype=np.float64)
    C_64 = np.asarray(C, dtype=np.float64)
    if T <= 0:
        raise ValueError("T must be positive")
    ny = C_64.shape[0]

    if method == "expm":
        Wy = C_64 @ _van_loan_gramian(A_64, B_64 @ B_64.T, T) @ C_64.T
        Qy = C_64 @ _van_loan_gramian(A_64, np.eye(A_64.shape[0]), T) @ C_64.T
    elif method == "quadrature":
        Wy, Qy = _quadrature_gramians(A_64, B_64, C_64, T, num_points)
        Wy = 0.5 * (Wy + Wy.T)
        Qy = 0.5 * (Qy + Qy.T)
    else:
        raise ValueError(f"unknown gramian method: {method}")

    eig_Wy = np.linalg.eigvalsh(Wy)
    eig_Qy = np.linalg.eigvalsh(Qy)
    lambda_max_Wy = float(np.max(eig_Wy))
    lambda_min_Wy = float(np.min(eig_Wy))
    lambda_max_Qy = float(np.max(eig_Qy))
    scale = max(1.0, abs(lambda_max_Wy))
    numerical_rank = int(np.sum(eig_Wy > rank_tol * scale))
    rank_deficient = numerical_rank < ny

    if rank_deficient or lambda_max_Qy <= 0.0:
        rho_T = None
        cond_Wy = None
    else:
        rho_T = float(np.sqrt(lambda_min_Wy) / np.sqrt(lambda_max_Qy))
        cond_Wy = float(lambda_max_Wy / lambda_min_Wy)

    return {
        "rho_T": rho_T,
        "lambda_min_Wy": lambda_min_Wy,
        "lambda_max_Wy": lambda_max_Wy,
        "lambda_max_Qy": lambda_max_Qy,
        "cond_Wy": cond_Wy,
        "rank_deficient": rank_deficient,
        "numerical_rank": numerical_rank,
        "output_dim": int(ny),
        "rank_tolerance": rank_tol,
        "T": float(T),
        "method": method,
        "num_points": int(num_points) if method == "quadrature" else None,
        "float_dtype": "float64",
    }


def compute_gramian_sensitivity(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    T_values,
    num_points_values,
    rank_tol: float = DEFAULT_RANK_TOL,
) -> list[dict]:
    """rho_T sensitivity to integration resolution and horizon (quadrature mode)."""
    rows = []
    for T in T_values:
        for num_points in num_points_values:
            res = compute_continuous_gramian_and_margin(
                A, B, C, T=float(T), num_points=int(num_points),
                rank_tol=rank_tol, method="quadrature",
            )
            rows.append(
                {
                    "T": float(T),
                    "num_points": int(num_points),
                    "rho_T": res["rho_T"],
                    "lambda_min_Wy": res["lambda_min_Wy"],
                    "cond_Wy": res["cond_Wy"],
                    "rank_deficient": res["rank_deficient"],
                }
            )
    return rows


def compute_controllability_rank(A, B, C=None, max_power: int | None = None) -> dict:
    """Rank of the reachability matrix, optionally after output projection C."""
    A_t = torch.as_tensor(np.asarray(A), dtype=torch.float64)
    B_t = torch.as_tensor(np.asarray(B), dtype=torch.float64)
    N = A_t.shape[0]
    powers = N if max_power is None else int(max_power)
    blocks = [B_t]
    for _ in range(1, powers):
        blocks.append(A_t @ blocks[-1])
    reach = torch.cat(blocks, dim=1)
    full_rank = int(torch.linalg.matrix_rank(reach).item())
    out = {"latent_dim": N, "full_reachability_rank": full_rank}
    if C is not None:
        C_t = torch.as_tensor(np.asarray(C), dtype=torch.float64)
        out["output_reachability_rank"] = int(
            torch.linalg.matrix_rank(C_t @ reach).item()
        )
        out["output_dim"] = int(C_t.shape[0])
    return out


# ---------------------------------------------------------------------------
# action response
# ---------------------------------------------------------------------------
def compute_action_response_error(
    model,
    sys,
    x_samples: torch.Tensor,
    u_samples: torch.Tensor,
    u_ref: torch.Tensor,
    eps: float = 1e-12,
) -> dict:
    """Same-state action response (EXPERIMENT §4.2).

    Uses the instantaneous latent dynamics f_z(z,u) = dz/dt, not the trajectory
    rollout. Returns raw MSE, epsilon-normalised error and a relative error.
    """
    spec = extract_latent_model(model)
    n = spec["state_dim"]
    model.eval() if hasattr(model, "eval") else None

    with torch.no_grad():
        f_true_u = sys.dynamics(0.0, x_samples, u_samples)
        f_true_ref = sys.dynamics(0.0, x_samples, u_ref)
        delta_true = f_true_u - f_true_ref

        z = spec["encode"](x_samples).detach()
        dz_u = spec["instant"](z, u_samples)
        dz_ref = spec["instant"](z, u_ref)
        delta_model = dz_u[:, :n] - dz_ref[:, :n]

        error_sq = torch.sum((delta_model - delta_true) ** 2, dim=-1)
        raw = float(torch.mean(error_sq).item())

        # epsilon-normalised definition from EXPERIMENT §4.2
        model_delta_sq = torch.sum((delta_model - delta_true) ** 2, dim=-1)
        normalizer = torch.mean(torch.sum(delta_true ** 2, dim=-1)) + eps
        normalized = float(torch.mean(model_delta_sq).item() / normalizer.item())

        rel = torch.linalg.norm(delta_model - delta_true, dim=-1)
        rel_den = torch.linalg.norm(delta_true, dim=-1) + eps
        relative = rel / rel_den

    return {
        "raw": raw,
        "normalized": normalized,
        "relative_error_mean": float(relative.mean().item()),
        "relative_error_p95": float(torch.quantile(relative, 0.95).item())
        if relative.numel() > 1 else float(relative.item()),
        "delta_true_energy_mean": float(torch.mean(torch.sum(delta_true ** 2, dim=-1)).item()),
        "num_samples": int(x_samples.shape[0]),
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# lifted map / manifold diagnostics
# ---------------------------------------------------------------------------
def _summary(values: list[float], prefix: str) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        prefix: float(arr.mean()),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_max": float(arr.max()),
    }


def compute_vector_field_mismatches(model, sys, x_samples: torch.Tensor) -> dict:
    """Lifted/physical input-map and drift mismatches (EXPERIMENT §4.3)."""
    spec = extract_latent_model(model)
    A = spec["A"].detach()
    B = spec["B"].detach()
    encode_fn = spec["encode"]
    n = sys.Nstates
    latent_dim = spec["latent_dim"]
    C = torch.eye(n, latent_dim, dtype=A.dtype, device=A.device)

    E_phys_list, E_lift_list, E_drift_list, d_orient_list = [], [], [], []

    for i in range(x_samples.shape[0]):
        xi = x_samples[i].clone().detach().requires_grad_(True)
        phi_i = encode_fn(xi.unsqueeze(0)).squeeze(0)
        J_phi = torch.zeros(latent_dim, n, dtype=A.dtype, device=A.device)
        for j in range(latent_dim):
            grad_out = torch.zeros(latent_dim, dtype=A.dtype, device=A.device)
            grad_out[j] = 1.0
            (grad_xi,) = torch.autograd.grad(
                phi_i, xi, grad_outputs=grad_out, retain_graph=True
            )
            J_phi[j] = grad_xi

        with torch.no_grad():
            Gx_i = sys.analytic_Gx(xi.detach().unsqueeze(0)).squeeze(0)
            f0_i = sys.analytic_f0(xi.detach().unsqueeze(0)).squeeze(0)

        CB = C @ B
        E_phys_list.append(float(torch.norm(CB - Gx_i, p="fro").item()))

        Gz_star = J_phi @ Gx_i
        E_lift_list.append(float(torch.norm(B - Gz_star, p="fro").item()))

        drift_model = A @ phi_i.detach()
        drift_true = J_phi @ f0_i
        E_drift_list.append(float(torch.norm(drift_model - drift_true, p=2).item()))

        # d_orient = ||(I - Pi_{col Gz*}) B||_2, stable SVD projector.
        U_gz, S_gz, _ = torch.linalg.svd(Gz_star, full_matrices=False)
        if S_gz.numel() > 0 and float(S_gz[0]) > 0.0:
            rank = int((S_gz > 1e-12 * S_gz[0]).sum().item())
            U_span = U_gz[:, :rank]
            Pi = U_span @ U_span.T
            B_perp = (torch.eye(latent_dim, dtype=A.dtype, device=A.device) - Pi) @ B
            d_orient_list.append(float(torch.linalg.matrix_norm(B_perp, ord=2).item()))
        else:
            d_orient_list.append(float(torch.linalg.matrix_norm(B, ord=2).item()))

    out = {}
    out.update(_summary(E_phys_list, "E_phys"))
    out.update(_summary(E_lift_list, "E_lift"))
    out.update(_summary(E_drift_list, "E_drift"))
    out.update(_summary(d_orient_list, "d_orient"))
    out["num_samples"] = int(x_samples.shape[0])
    out["d_orient_convention"] = "||(I - Pi_col(Gz*)) B||_2, SVD rank tol 1e-12"
    return out


def compute_manifold_error(model, z_samples: torch.Tensor) -> dict:
    """E_manifold(z) = ||z - phi(C z)||_2 on predicted latents."""
    spec = extract_latent_model(model)
    n = spec["state_dim"]
    with torch.no_grad():
        z = z_samples if z_samples.ndim == 2 else z_samples.reshape(-1, z_samples.shape[-1])
        z_back = spec["encode"](z[:, :n].detach())
        err = torch.linalg.norm(z.detach() - z_back, dim=-1)
    return {
        "mean": float(err.mean().item()),
        "p95": float(torch.quantile(err, 0.95).item()) if err.numel() > 1 else float(err.item()),
        "max": float(err.max().item()),
        "num_samples": int(err.numel()),
    }


# ---------------------------------------------------------------------------
# H3 constrained factorization distance
# ---------------------------------------------------------------------------
def _factorization_distance_single(G: np.ndarray, B: np.ndarray, lo: float, hi: float) -> dict:
    g = G[:, 0]
    b = B[:, 0]
    if np.linalg.norm(g) < EPS:
        return {"d_star": float(np.linalg.norm(b)), "feasible": True, "approximate": False}
    obj = lambda s: float(np.linalg.norm(g / s - b))
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
    return {"d_star": float(res.fun), "feasible": True, "approximate": False}


def _factorization_distance_multi(G: np.ndarray, B: np.ndarray, lo: float, hi: float) -> dict:
    """Approximate min_{M: sigma in [lo,hi]} ||G M^{-1} - B||_2.

    Parameterise N = M^{-1} (so spectrum(N) in [1/hi, 1/lo]) and minimise the
    linear residual ||G N - B||_2 with SLSQP. This is a numerical diagnostic, not
    a global feasibility certificate.
    """
    m = G.shape[1]
    # Least-squares residual is a lower bound and is feasible when its spectrum
    # respects the bounds.
    N_ls, *_ = np.linalg.lstsq(G, B, rcond=None)
    sv = np.linalg.svd(N_ls, compute_uv=False)
    if sv.min() >= 1.0 / hi - 1e-9 and sv.max() <= 1.0 / lo + 1e-9:
        return {
            "d_star": float(np.linalg.norm(G @ N_ls - B, ord=2)),
            "feasible": True,
            "approximate": False,
        }

    def objective(nvec):
        N = nvec.reshape(m, m)
        return float(np.linalg.norm(G @ N - B, ord=2))

    def sv_constraint(nvec):
        N = nvec.reshape(m, m)
        s = np.linalg.svd(N, compute_uv=False)
        # positive when inside [1/hi, 1/lo]
        return min(s.min() - 1.0 / hi, 1.0 / lo - s.max())

    x0 = N_ls.flatten()
    cons = [{"type": "ineq", "fun": sv_constraint}]
    res = minimize(objective, x0, method="SLSQP", constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    feasible = bool(sv_constraint(res.x) >= -1e-8)
    return {"d_star": float(res.fun), "feasible": feasible, "approximate": True}


def compute_input_map_factorization_distance(
    target: torch.Tensor | np.ndarray,
    B: torch.Tensor | np.ndarray,
    sigma_min: float,
    sigma_max: float,
) -> dict:
    """Pointwise constrained factorization distance d_star (RRD v0.5 §6.6).

    target = G_z^*(x) [N x m], B [N x m]; M is m x m with
    sigma_min <= sigma(M) <= sigma_max. Exact scalar solve for m=1, approximate
    SLSQP for m>1.
    """
    G = np.asarray(target.detach().cpu() if isinstance(target, torch.Tensor) else target, dtype=np.float64)
    B_np = np.asarray(B.detach().cpu() if isinstance(B, torch.Tensor) else B, dtype=np.float64)
    if G.ndim == 1:
        G = G[:, None]
    if B_np.ndim == 1:
        B_np = B_np[:, None]
    if G.shape != B_np.shape:
        raise ValueError(f"shape mismatch: G {G.shape} vs B {B_np.shape}")
    if not (0 < sigma_min <= sigma_max):
        raise ValueError("require 0 < sigma_min <= sigma_max")

    if G.shape[1] == 1:
        res = _factorization_distance_single(G, B_np, sigma_min, sigma_max)
    else:
        res = _factorization_distance_multi(G, B_np, sigma_min, sigma_max)
    res.update({"sigma_min": float(sigma_min), "sigma_max": float(sigma_max)})
    return res


def compute_h3_feasibility(
    model,
    sys,
    x_samples: torch.Tensor,
    rho_T: float | None,
    sigma_min: float = 0.5,
    sigma_max: float = 2.0,
) -> dict:
    """H3 feasibility diagnostics: d_star, rank and orientation vs rho_T."""
    spec = extract_latent_model(model)
    A = spec["A"].detach()
    B = spec["B"].detach()
    n = sys.Nstates
    N = spec["latent_dim"]
    C = torch.eye(n, N, dtype=A.dtype, device=A.device)

    d_star, d_orient, feasible_flags, approx_flags = [], [], [], []
    for i in range(x_samples.shape[0]):
        xi = x_samples[i].clone().detach().requires_grad_(True)
        phi_i = spec["encode"](xi.unsqueeze(0)).squeeze(0)
        J_phi = torch.zeros(N, n, dtype=A.dtype, device=A.device)
        for j in range(N):
            grad_out = torch.zeros(N, dtype=A.dtype, device=A.device)
            grad_out[j] = 1.0
            (grad_xi,) = torch.autograd.grad(phi_i, xi, grad_outputs=grad_out, retain_graph=True)
            J_phi[j] = grad_xi
        with torch.no_grad():
            Gx_i = sys.analytic_Gx(xi.detach().unsqueeze(0)).squeeze(0)
            Gz = J_phi @ Gx_i
            U, S, _ = torch.linalg.svd(Gz, full_matrices=False)
            if S.numel() and float(S[0]) > 0:
                rank = int((S > 1e-12 * S[0]).sum().item())
                Pi = U[:, :rank] @ U[:, :rank].T
                d_orient.append(float(torch.linalg.matrix_norm((torch.eye(N, dtype=A.dtype, device=A.device) - Pi) @ B, ord=2)))
            else:
                d_orient.append(float(torch.linalg.matrix_norm(B, ord=2)))
        res = compute_input_map_factorization_distance(Gz, B, sigma_min, sigma_max)
        d_star.append(res["d_star"])
        feasible_flags.append(res["feasible"])
        approx_flags.append(res["approximate"])

    rank_info = compute_controllability_rank(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy())
    d_star_np = np.asarray(d_star)
    d_orient_np = np.asarray(d_orient)
    margin_defined = rho_T is not None
    return {
        "d_star_mean": float(d_star_np.mean()),
        "d_star_p95": float(np.quantile(d_star_np, 0.95)),
        "d_star_max": float(d_star_np.max()),
        "d_orient_mean": float(d_orient_np.mean()),
        "d_orient_p95": float(np.quantile(d_orient_np, 0.95)),
        "sigma_bounds": [float(sigma_min), float(sigma_max)],
        "factorization_feasible_fraction": float(np.mean(feasible_flags)),
        "multinput_used_approximation": bool(any(approx_flags)),
        "rank": rank_info,
        "rho_T": rho_T,
        "margin_defined": margin_defined,
        # Necessary orientation condition from RRD §6.6 (not sufficient).
        "d_orient_p95_below_rho": bool(margin_defined and d_orient_np.max() < rho_T),
        "d_star_p95_below_rho": bool(margin_defined and float(np.quantile(d_star_np, 0.95)) < rho_T),
        "note": "d_star is pointwise numerical feasibility, not a global certificate",
    }


# ---------------------------------------------------------------------------
# candidate regret
# ---------------------------------------------------------------------------
def compute_candidate_regret(
    model,
    sys,
    x_samples: torch.Tensor,
    candidate_actions: torch.Tensor,
    horizon: int = 3,
    control_weight: float = 0.1,
) -> dict:
    """Candidate-set regret R_cand (EXPERIMENT §4.5), true plant via RK4."""
    from research.baseline.data_generator import rk4_step

    spec = extract_latent_model(model)
    n = spec["state_dim"]
    dt = float(getattr(sys, "dt", getattr(model, "dt", 0.02)))
    num_samples = x_samples.shape[0]
    num_candidates = candidate_actions.shape[0]
    regrets = []
    plant_costs, model_costs = [], []

    for i in range(num_samples):
        x0 = x_samples[i].clone()
        true_costs = []
        predicted_costs = []
        for c_idx in range(num_candidates):
            u_cand = candidate_actions[c_idx]

            x_true = x0.clone()
            cost_true = 0.0
            for _ in range(horizon):
                x_true = rk4_step(
                    sys.dynamics, 0.0, x_true.unsqueeze(0), u_cand.unsqueeze(0), dt
                ).squeeze(0)
                cost_true += float(
                    torch.norm(x_true).item() ** 2
                    + control_weight * torch.norm(u_cand).item() ** 2
                )
            true_costs.append(cost_true)

            x_model = x0.clone()
            cost_model = 0.0
            with torch.no_grad():
                for _ in range(horizon):
                    z = spec["encode"](x_model.unsqueeze(0))
                    dz = spec["instant"](z, u_cand.unsqueeze(0))
                    x_model = x_model + dt * dz[0, :n]
                    cost_model += float(
                        torch.norm(x_model).item() ** 2
                        + control_weight * torch.norm(u_cand).item() ** 2
                    )
            predicted_costs.append(cost_model)

        best_model_idx = int(np.argmin(predicted_costs))
        true_cost_of_best_model = true_costs[best_model_idx]
        optimal_true_cost = float(np.min(true_costs))
        regrets.append(max(0.0, true_cost_of_best_model - optimal_true_cost))
        plant_costs.append(optimal_true_cost)
        model_costs.append(predicted_costs[best_model_idx])

    reg = np.asarray(regrets)
    return {
        "mean": float(reg.mean()),
        "p95": float(np.quantile(reg, 0.95)) if reg.size > 1 else float(reg[0]),
        "max": float(reg.max()),
        "horizon": int(horizon),
        "num_samples": int(num_samples),
        "num_candidates": int(num_candidates),
        "dt": dt,
        "plant_step": "rk4",
        "cost": "||x||^2 + %.3f ||u||^2" % control_weight,
    }
