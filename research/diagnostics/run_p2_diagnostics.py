"""Phase P2: controlled stress tests with the original Koopman baseline.

Scientific contract:
* The trained model is the upstream ``KoopmanLitModule`` (Miao et al. baseline),
  including its structured controllable operator, encoder losses, controllability
  regularizer and ODE integration. No simplified surrogate is used.
* Transitions are ``(x_t, u_t) -> x_{t+1}`` including the initial state ``x_0``.
* One time step ``sys.dt`` is propagated to data generation, training,
  prediction and closed-loop evaluation. Ground truth is the RK4 simulator.
* Gate verdicts are derived from the pre-registered acceptance criteria below;
  they are never hard-coded.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from research.baseline.common import (
    environment_provenance,
    git_provenance,
    sha256_file,
)
from research.baseline.data_generator import generate_benchmark_dataset, rk4_step
from research.benchmarks import load_benchmark
from research.control.qp_mpc import KoopmanConvexMPC
from research.diagnostics.metrics import (
    compute_action_response_error,
    compute_candidate_regret,
    compute_continuous_gramian_and_margin,
    compute_gramian_sensitivity,
    compute_h3_feasibility,
    compute_manifold_error,
    compute_vector_field_mismatches,
)

HORIZONS = (15, 30, 50)
EVAL_SEED = 4242
EVAL_DT = {"rotation_family": 0.02, "unicycle": 0.02, "planar_quadrotor": 0.01}
LATENT_DIMS = {"rotation_family": 6, "unicycle": 8, "planar_quadrotor": 12}

# One locked configuration for every P2 baseline run. Only `latent_dim` changes
# with the system/ablation; everything else is identical to the upstream loss and
# operator parameterization.
P2_BASELINE_CONFIG = {
    "model_class": "KoopmanLitModule",
    "koopman_mode": "controllable",
    "control_loss_flag": True,
    "encoder_loss_flag": True,
    "use_modulated_u": False,
    "num_layers": 3,
    "hidden_size": 64,
    "u_encoder_layers": 2,
    "u_encoder_hidden": 64,
    "lambda_encoder_pred": 0.5,
    "lambda_control": 0.005,
    "prediction_gamma": 0.8,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "window_len": 15,
    "max_windows_per_traj": 20,
}

ACCEPTANCE_CRITERIA = {
    "h2_response_ratio_min": 1.5,
    "h2_require_monotone_response": True,
    "max_input_covariance_condition": 100.0,
    "max_normalized_state_distance_p95": 3.0,
    "max_one_step_mse": 0.05,
    "mpc_success_rate_min": 0.95,
    "mpc_latency_p95_max_s": 0.05,
    "max_input_violation": 1e-6,
    "h3_sigma_bounds": [0.5, 2.0],
}


# ---------------------------------------------------------------------------
# official baseline adapter
# ---------------------------------------------------------------------------
def _model_config(latent_dim: int, config: dict | None = None) -> dict:
    cfg = dict(P2_BASELINE_CONFIG if config is None else config)
    return {
        "koopman_mode": cfg["koopman_mode"],
        "control_loss_flag": cfg["control_loss_flag"],
        "encoder_loss_flag": cfg["encoder_loss_flag"],
        "use_modulated_u": cfg["use_modulated_u"],
        "latent_dim": int(latent_dim),
        "num_layers": cfg["num_layers"],
        "hidden_size": cfg["hidden_size"],
        "u_encoder_layers": cfg["u_encoder_layers"],
        "u_encoder_hidden": cfg["u_encoder_hidden"],
        "lambda_encoder_pred": cfg["lambda_encoder_pred"],
        "lambda_control": cfg["lambda_control"],
        "ode_solver": cfg.get("ode_solver", "euler"),
    }


def build_official_module(sys_inst, latent_dim: int, config: dict | None = None):
    """Construct the upstream Koopman module for a benchmark instance."""
    from training.train import KoopmanLitModule

    cfg = dict(P2_BASELINE_CONFIG if config is None else config)
    system_config = {
        "dynamics": sys_inst.dynamics,
        "sample_initial_state": sys_inst.sample_initial_state,
        "umin": sys_inst.umin,
        "umax": sys_inst.umax,
        "dt": float(sys_inst.dt),
        "state_dim": int(sys_inst.Nstates),
    }
    module = KoopmanLitModule(system_config, _model_config(latent_dim, cfg), lr=cfg["lr"])
    if float(module.model.dt) != float(sys_inst.dt):
        raise ValueError("model dt does not match benchmark dt")
    return module


def build_training_dataset(
    trajectories, window_len: int = 15, max_windows_per_traj: int = 20, seed: int = 42
):
    """Window sampler that keeps (x_t, u_t) aligned and includes x_0."""
    from my_utils.dataset import SampledMultiTrajectoryDataset

    return SampledMultiTrajectoryDataset(
        trajectories,
        window_len=window_len,
        max_windows_per_traj=max_windows_per_traj,
        seed=seed,
    )


def train_official_baseline(
    trajectories,
    sys_inst,
    latent_dim: int,
    epochs: int,
    seed: int,
    window_len: int = 15,
    max_windows_per_traj: int = 20,
    batch_size: int = 64,
    checkpoint_path: str | None = None,
    dataset_sha256: str | None = None,
    config: dict | None = None,
):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    cfg = dict(P2_BASELINE_CONFIG if config is None else config)
    pl.seed_everything(seed, workers=True)
    module = build_official_module(sys_inst, latent_dim, cfg)
    dataset = build_training_dataset(
        trajectories,
        window_len=window_len,
        max_windows_per_traj=max_windows_per_traj,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=False,
        devices=1,
        accelerator="cpu",
    )
    trainer.fit(module, loader)
    module.eval()

    prov = {
        "model_class": type(module).__name__,
        "module_entrypoint": "training.train:KoopmanLitModule",
        "parameterization": type(module.model.operator.core).__name__,
        "config": _model_config(latent_dim, cfg),
        "latent_dim": int(latent_dim),
        "state_dim": int(sys_inst.Nstates),
        "control_dim": int(sys_inst.Ncontrols),
        "dt": float(module.dt),
        "ode_solver": module.model.ode_solver,
        "seed": int(seed),
        "epochs": int(epochs),
        "parameter_count": int(sum(p.numel() for p in module.parameters())),
        "dataset": {
            "sha256": dataset_sha256,
            "num_trajectories": len(trajectories),
            "windows": len(dataset),
            "window_len": int(window_len),
            "max_windows_per_traj": int(max_windows_per_traj),
        },
        "trainer": {
            "optimizer": "AdamW",
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "final_train_loss": _finite_or_none(
                trainer.callback_metrics.get("train_total_loss")
            ),
            "global_step": int(trainer.global_step),
        },
        "state_dict_shapes": {k: list(v.shape) for k, v in module.state_dict().items()},
    }
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(module.state_dict(), checkpoint_path)
        prov["checkpoint"] = {
            "path": checkpoint_path,
            "sha256": sha256_file(checkpoint_path),
        }
    return module, prov


def predict_one_step(module, x: torch.Tensor, u: torch.Tensor, dt: float | None = None):
    """One Euler step of the official latent ODE at the benchmark dt."""
    step = float(module.model.dt if dt is None else dt)
    z = module.model.encode(x)
    dz = module.model.dynamics(0.0, z, u)
    return (z + step * dz)[:, : module.state_dim]


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def _stack_transitions(data):
    """(x_t, u_t, x_{t+1}) for t=0..T-1, including x_0."""
    x_list, u_list, next_list = [], [], []
    for traj in data:
        x_seq = traj["x_seq"] if traj["x_seq"].ndim == 2 else traj["x_seq"].squeeze(1)
        u_seq = traj["u_seq"]
        x0 = traj["x0"].reshape(1, -1)
        x_t = torch.cat([x0, x_seq[:-1]], dim=0)
        x_list.append(x_t)
        u_list.append(u_seq)
        next_list.append(x_seq)
    return torch.cat(x_list), torch.cat(u_list), torch.cat(next_list)


def _mse(pred, true):
    return float(torch.mean((pred - true) ** 2).item())


def _finite_or_none(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _verify_transition_replay(data, sys_inst, num_checks: int = 5) -> dict:
    """Re-simulate stored transitions with RK4 to verify indexing and dt."""
    devs = []
    checks = 0
    for traj in data:
        x = traj["x0"].reshape(-1)
        for t in range(min(num_checks, traj["u_seq"].shape[0])):
            x_next = rk4_step(
                sys_inst.dynamics, 0.0, x.unsqueeze(0), traj["u_seq"][t : t + 1], sys_inst.dt
            ).squeeze(0)
            devs.append(float(torch.max(torch.abs(x_next - traj["x_seq"][t])).item()))
            checks += 1
            x = traj["x_seq"][t]
    return {
        "transitions_checked": int(checks),
        "max_abs_deviation": float(max(devs)) if devs else None,
        "indexing": "x_t,u_t -> x_{t+1}, initial x_0 included",
        "integrator": "rk4",
    }


def _coverage_diagnostics(train, eval, sys_inst, num_bins: int = 5) -> dict:
    def states(data):
        return torch.cat(
            [torch.cat([t["x0"].reshape(1, -1), t["x_seq"]], dim=0) for t in data], dim=0
        )

    train_x, eval_x = states(train), states(eval)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    train_z = (train_x - mean) / std
    eval_z = (eval_x - mean) / std
    d = torch.cdist(eval_z, train_z)
    nearest = d.min(dim=1).values

    all_u = torch.cat([t["u_seq"] for t in train], dim=0)
    if all_u.shape[1] == 1:
        eig = np.asarray([float(all_u.var(unbiased=False).item())], dtype=np.float64)
        cond = 1.0 if eig[0] > 0 else None
    else:
        cov_np = np.cov(all_u.numpy(), rowvar=False)
        eig = np.linalg.eigvalsh(cov_np)
        cond = float(eig.max() / max(eig.min(), 1e-12)) if eig.min() > 0 else None
    bin_occupancy = []
    for j in range(all_u.shape[1]):
        lo, hi = float(sys_inst.umin[j]), float(sys_inst.umax[j])
        hist, _ = np.histogram(all_u[:, j].numpy(), bins=num_bins, range=(lo, hi))
        bin_occupancy.append(float(hist.min() / max(hist.sum(), 1)))
    return {
        "state_distance_p95": float(torch.quantile(nearest, 0.95).item()),
        "state_distance_max": float(nearest.max().item()),
        "input_covariance_condition": cond,
        "input_covariance_min_eig": float(eig.min()),
        "min_action_bin_fraction": float(min(bin_occupancy)),
        "num_train_states": int(train_x.shape[0]),
        "num_eval_states": int(eval_x.shape[0]),
        "normalization": "train mean/std",
    }


def evaluate_baseline(
    module,
    sys_inst,
    eval_path: str,
    horizons=HORIZONS,
    response_samples: int = 64,
    diagnostic_samples: int = 8,
    seed: int = 0,
    sigma_bounds=(0.5, 2.0),
) -> dict:
    data = torch.load(eval_path)
    X, U, X_next = _stack_transitions(data)
    n = sys_inst.Nstates

    module.eval()
    with torch.no_grad():
        pred = predict_one_step(module, X, U)
        one_step_mse = _mse(pred, X_next)
        one_step_rmse = float(np.sqrt(one_step_mse))

        z_all = module.model.encode(X)
        dz_all = module.model.dynamics(0.0, z_all, U)
        f_true = sys_inst.dynamics(0.0, X, U)
        derivative_err = dz_all[:, :n] - f_true
        derivative_mse = float(torch.mean(torch.sum(derivative_err ** 2, dim=-1)).item() / n)
        derivative_rel = float(
            torch.mean(
                torch.linalg.norm(derivative_err, dim=-1)
                / (torch.linalg.norm(f_true, dim=-1) + 1e-12)
            ).item()
        )

    # Held-out open-loop prediction, no re-encoding.
    x0 = torch.cat([t["x0"].reshape(1, -1) for t in data], dim=0)
    open_loop = {}
    manifold = {}
    for H in horizons:
        u_seq = torch.stack([t["u_seq"][:H] for t in data])
        truth = torch.stack([t["x_seq"][:H] for t in data])
        with torch.no_grad():
            z_seq = module(x0, u_seq)
        pred_seq = z_seq[:, :, :n]
        open_loop[str(H)] = {
            "mse": _mse(pred_seq, truth),
            "final_step_mse": _mse(pred_seq[:, -1], truth[:, -1]),
            "free_running": True,
            "reencode": False,
        }
        if H == horizons[-1]:
            k = min(3, z_seq.shape[0])
            manifold = compute_manifold_error(module, z_seq[:k])

    torch.manual_seed(seed)
    idx = torch.randperm(X.shape[0])[:response_samples]
    u_rand = torch.rand(response_samples, sys_inst.Ncontrols)
    u_samples = sys_inst.umin + u_rand * (sys_inst.umax - sys_inst.umin)
    u_ref = torch.zeros_like(u_samples)
    response = compute_action_response_error(
        module, sys_inst, X[idx], u_samples, u_ref
    )
    response["E_response"] = response["raw"]

    diag_idx = torch.randperm(X.shape[0])[: max(diagnostic_samples, 1)]
    vf = compute_vector_field_mismatches(module, sys_inst, X[diag_idx])

    A, B = module.model.operator.get_AB()
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    C = np.eye(n, A.shape[0])
    gramian = compute_continuous_gramian_and_margin(
        A, B, C, T=1.0, method="expm"
    )
    gramian_sensitivity = compute_gramian_sensitivity(
        A, B, C,
        T_values=[0.2, 0.5, 1.0, 2.0],
        num_points_values=[50, 200, 800],
    )

    h3 = compute_h3_feasibility(
        module, sys_inst, X[diag_idx], gramian["rho_T"],
        sigma_min=sigma_bounds[0], sigma_max=sigma_bounds[1],
    )

    candidate_actions = torch.stack(
        [
            sys_inst.umin + t * (sys_inst.umax - sys_inst.umin)
            for t in torch.linspace(0.0, 1.0, 5)
        ],
        dim=0,
    )
    regret = compute_candidate_regret(
        module, sys_inst, X[diag_idx], candidate_actions, horizon=3
    )

    alignment = _verify_transition_replay(data, sys_inst)

    return {
        "num_transitions": int(X.shape[0]),
        "one_step_mse": one_step_mse,
        "one_step_rmse": one_step_rmse,
        "derivative_error": {
            "mse": derivative_mse,
            "relative_mean": derivative_rel,
            "space": "physical dx/dt",
        },
        "open_loop_mse": open_loop,
        "manifold_error": manifold,
        "response": response,
        "E_phys": vf["E_phys"],
        "E_lift": vf["E_lift"],
        "E_drift": vf["E_drift"],
        "d_orient": vf["d_orient"],
        "d_orient_p95": vf["d_orient_p95"],
        "vector_field": vf,
        "gramian": gramian,
        "gramian_sensitivity": gramian_sensitivity,
        "h3": h3,
        "candidate_regret": regret,
        "alignment": alignment,
    }


# ---------------------------------------------------------------------------
# MPC evaluation
# ---------------------------------------------------------------------------
def _linearization(sys_inst, x0):
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    f0 = lambda xi: sys_inst.analytic_f0(xi.unsqueeze(0)).squeeze(0)
    A = torch.autograd.functional.jacobian(f0, x).numpy()
    B = sys_inst.analytic_Gx(x.unsqueeze(0)).squeeze(0).detach().numpy()
    return A, B


def _zero_control_cost(sys_inst, x0, target, steps, Q, R):
    x = np.asarray(x0, dtype=np.float64).copy()
    cost = 0.0
    for _ in range(steps):
        u = np.zeros(sys_inst.Ncontrols)
        x = (
            rk4_step(
                sys_inst.dynamics, 0.0,
                torch.from_numpy(x).float().unsqueeze(0),
                torch.from_numpy(u).float().unsqueeze(0),
                sys_inst.dt,
            )
            .squeeze(0).numpy().astype(np.float64)
        )
        err = x - target
        cost += float(err @ Q @ err + u @ R @ u)
    return {
        "total_cost": float(cost),
        "final_error": float(np.linalg.norm(x - target)),
        "final_state": x.tolist(),
    }


def _mpc_eval(sys_inst, A, B, C, encoder_fn, x0, target, steps, horizon, seed):
    torch.manual_seed(seed)
    Q = np.diag([10.0, 1.0]) if sys_inst.Nstates == 2 else np.eye(sys_inst.Nstates)
    R = 0.1 * np.eye(sys_inst.Ncontrols)
    mpc = KoopmanConvexMPC(
        A=A, B=B, C=C, dt=sys_inst.dt, horizon=horizon, Q=Q, R=R,
        umin=sys_inst.umin.numpy(), umax=sys_inst.umax.numpy(),
        encoder_fn=encoder_fn,
    )
    traj = mpc.simulate_closed_loop(
        sys_inst, x0=np.asarray(x0), x_target=np.asarray(target), num_steps=steps
    )
    return {
        "total_cost": float(traj["total_cost"]),
        "final_error": float(traj["final_error"]),
        "u_abs_mean": float(np.mean(np.abs(np.asarray(traj["u_history"])))),
        "final_state": np.asarray(traj["x_history"][-1]).tolist(),
        "solver": traj["solver"],
        "settings": {
            "horizon": horizon,
            "num_steps": steps,
            "Q": np.diag(Q).tolist(),
            "R": np.diag(R).tolist(),
            "umin": sys_inst.umin.tolist(),
            "umax": sys_inst.umax.tolist(),
            "dt": float(sys_inst.dt),
            "seed": int(seed),
            "solver": "OSQP",
            "regime": "exact lifted-LTI QP",
        },
    }


# ---------------------------------------------------------------------------
# aggregation and gates
# ---------------------------------------------------------------------------
def _aggregate(seed_entries: dict, keys) -> dict:
    out = {}
    for key in keys:
        vals = [e[key] for e in seed_entries.values() if e.get(key) is not None]
        if vals:
            out[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return out


def _derive_gates(report: dict) -> dict:
    crit = ACCEPTANCE_CRITERIA
    sweep = report["rotation_sweep"]
    a_keys = sorted(sweep, key=lambda k: sweep[k]["a"])
    a_values = [sweep[k]["a"] for k in a_keys]

    common_seeds = None
    for key in a_keys:
        seeds = set(sweep[key]["seeds"])
        common_seeds = seeds if common_seeds is None else common_seeds & seeds
    common_seeds = sorted(common_seeds or [])

    ratios = []
    monotone_flags = []
    for seed in common_seeds:
        norm = [sweep[k]["seeds"][seed]["response"]["normalized"] for k in a_keys]
        ratios.append(norm[-1] / max(norm[0], 1e-12))
        monotone_flags.append(
            all(norm[i] <= norm[i + 1] * (1 + 1e-6) for i in range(len(norm) - 1))
        )
    ratio_min = float(min(ratios)) if ratios else None
    response_growth = (
        ratio_min is not None
        and ratio_min >= crit["h2_response_ratio_min"]
        and (all(monotone_flags) if crit["h2_require_monotone_response"] else True)
    )

    coverage_ok = True
    coverage_issues = []
    optimization_ok = True
    optimization_issues = []
    for key in a_keys:
        for seed, entry in sweep[key]["seeds"].items():
            cov = entry["coverage"]
            if cov["input_covariance_condition"] is None or cov[
                "input_covariance_condition"
            ] > crit["max_input_covariance_condition"]:
                coverage_ok = False
                coverage_issues.append(f"{key}/seed{seed}: input covariance condition")
            if cov["state_distance_p95"] > crit["max_normalized_state_distance_p95"]:
                coverage_ok = False
                coverage_issues.append(f"{key}/seed{seed}: state distance p95")
            train_loss = entry["training"]["trainer"]["final_train_loss"]
            if train_loss is None or not np.isfinite(train_loss):
                optimization_ok = False
                optimization_issues.append(f"{key}/seed{seed}: non-finite train loss")
            if entry["one_step_mse"] > crit["max_one_step_mse"]:
                optimization_ok = False
                optimization_issues.append(
                    f"{key}/seed{seed}: one-step MSE {entry['one_step_mse']:.3g}"
                )

    learned_mpc = [
        ev["learned"]["solver"]
        for ev in report["mpc_evaluations"].values()
        if "learned" in ev
    ]
    solver_ok = bool(learned_mpc) and all(
        s["success_rate"] >= crit["mpc_success_rate_min"]
        and s["max_input_violation"] <= crit["max_input_violation"]
        for s in learned_mpc
    )
    latency_ok = bool(learned_mpc) and all(
        s["latency_p95_s"] <= crit["mpc_latency_p95_max_s"] for s in learned_mpc
    )

    alignment_ok = True
    dt_consistent = True
    for key in a_keys:
        entry = sweep[key]
        dt_consistent &= abs(entry["budget"]["dt"] - entry["datasets"]["eval"]["dt"]) < 1e-12
        for ds in entry["datasets"]["train"].values():
            dt_consistent &= abs(entry["budget"]["dt"] - ds["dt"]) < 1e-12
        for seed_entry in entry["seeds"].values():
            dev = seed_entry["alignment"]["max_abs_deviation"]
            if dev is None or dev > 1e-5:
                alignment_ok = False

    provenance_ok = bool(
        report["provenance"]["git_head"] and report["provenance"]["environment"]["torch"]
    )
    # Compare against the pre-registered protocol, not against the observed keys.
    expected_configs = len(
        report.get("protocol", {}).get("budget", {}).get("a_values", a_values)
    )
    p2_checks = {
        "official_baseline_used": report["baseline"]["model_class"],
        "transition_alignment_verified": bool(alignment_ok),
        "dt_consistent": bool(dt_consistent),
        "provenance_recorded": bool(provenance_ok),
        "rotation_configs_evaluated": len(a_keys),
        "expected_rotation_configs": expected_configs,
    }
    p2_pass = (
        p2_checks["official_baseline_used"] == "KoopmanLitModule"
        and alignment_ok
        and dt_consistent
        and provenance_ok
        and len(a_keys) == expected_configs >= 1
    )

    fixed_b = (
        "CONFIRMED"
        if (response_growth and coverage_ok and optimization_ok)
        else ("INCONCLUSIVE" if not coverage_ok or not optimization_ok else "NOT_CONFIRMED")
    )
    h2_go = fixed_b == "CONFIRMED" and coverage_ok
    h2_readiness = {
        "verdict": "GO" if h2_go else "NO-GO",
        "criteria": {
            "response_growth_ratio_min": ratio_min,
            "response_ratio_threshold": crit["h2_response_ratio_min"],
            "monotone_response": bool(all(monotone_flags)) if monotone_flags else False,
            "coverage_ok": bool(coverage_ok),
            "optimization_ok": bool(optimization_ok),
            "matched_class_comparison_feasible": True,
            "note": "fixed-B limitation alone does not validate H2; this is a readiness decision",
        },
    }

    h3_entries = [
        entry["h3"] for key in a_keys for entry in sweep[key]["seeds"].values()
    ]
    margin_defined = any(e["margin_defined"] for e in h3_entries)
    certificate_window = any(
        e["margin_defined"] and e["d_star_p95_below_rho"] and e["d_orient_p95_below_rho"]
        for e in h3_entries
    )
    h3_go = bool(certificate_window and solver_ok and latency_ok)
    gates = {
        "p2": {
            "verdict": "PASS" if p2_pass else "FAIL",
            "criteria": p2_checks,
        },
        "h2_fixed_b_limitation": {
            "verdict": fixed_b,
            "criteria": {
                "response_ratio_min": ratio_min,
                "response_ratio_threshold": crit["h2_response_ratio_min"],
                "coverage_ok": bool(coverage_ok),
                "coverage_issues": coverage_issues,
                "optimization_ok": bool(optimization_ok),
                "optimization_issues": optimization_issues,
                "solver_ok": bool(solver_ok),
            },
        },
        "h2_readiness": h2_readiness,
        "h3": {
            "verdict": "GO" if h3_go else "NO-GO",
            "criteria": {
                "margin_defined": bool(margin_defined),
                "certificate_window_exists": bool(certificate_window),
                "mpc_solver_ok": bool(solver_ok),
                "mpc_latency_ok": bool(latency_ok),
                "latency_budget_p95_s": crit["mpc_latency_p95_max_s"],
                "note": (
                    "H3 requires a usable margin window AND tractable bounded-input MPC; "
                    "a fixed-B failure does not validate H3"
                ),
            },
        },
    }
    return gates


def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, str):
        return str(obj)  # normalise str subclasses such as torch.TorchVersion
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


# ---------------------------------------------------------------------------
# main protocol
# ---------------------------------------------------------------------------
def _budget(quick: bool) -> dict:
    if quick:
        return {
            "num_trajectories": 4,
            "eval_trajectories": 3,
            "T_train": 20,
            "T_eval": 60,
            "epochs": 2,
            "seeds": [42],
            "a_values": [0.0, 1.0],
            "max_windows_per_traj": 3,
            "batch_size": 8,
            "latent_sweep": False,
            "mpc_steps": 8,
            "mpc_horizon": 6,
        }
    return {
        "num_trajectories": 30,
        "eval_trajectories": 10,
        "T_train": 50,
        "T_eval": 60,
        "epochs": 25,
        "seeds": [42, 43, 44],
        "a_values": [0.0, 0.5, 1.0, 2.0],
        "max_windows_per_traj": 20,
        "batch_size": 64,
        "latent_sweep": True,
        "mpc_steps": 20,
        "mpc_horizon": 12,
    }


def _generate(sys_name, num_trajectories, T, dt, seed, save_dir, sys_kwargs=None):
    return generate_benchmark_dataset(
        sys_name=sys_name,
        num_trajectories=num_trajectories,
        T=T,
        dt=dt,
        seed=seed,
        save_dir=save_dir,
        sys_kwargs=sys_kwargs or {},
    )


def run_p2_stress_diagnosis(
    output_dir: str = "outputs/diagnosis",
    quick_mode: bool = False,
) -> str:
    budget = _budget(quick_mode)
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    started = time.time()
    report = {
        "phase": "P2",
        "description": "Controlled stress tests with the original Koopman baseline",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "quick_mode": bool(quick_mode),
        "baseline": {
            "model_class": "KoopmanLitModule",
            "module_entrypoint": "training.train:KoopmanLitModule",
            "model_config": _model_config(LATENT_DIMS["rotation_family"]),
            "locked_config": P2_BASELINE_CONFIG,
            "losses": "prediction_loss + encoder consistency + controllability_loss (upstream)",
            "ode_integration": "upstream KoopmanModel.forward (torchdiffeq)",
        },
        "protocol": {
            "budget": budget,
            "horizons": list(HORIZONS),
            "eval_seed": EVAL_SEED,
            "ground_truth": "RK4 simulator at sys.dt",
            "candidate_pool": "5 actions on the umin->umax diagonal",
        },
        "acceptance_criteria": dict(ACCEPTANCE_CRITERIA),
        "rotation_sweep": {},
        "benchmark_evaluations": {},
        "mpc_evaluations": {},
        "h3_feasibility": {},
        "failure_mode_separation": {},
        "provenance": {
            "git_head": git_provenance()["git_head"],
            "git": git_provenance(),
            "environment": environment_provenance(),
            "source": "working tree at run time (see git block)",
        },
    }

    trained_models = {}
    eval_cache = {}

    # ------------------------------------------------------------------
    # E06: controlled rotation sweep
    # ------------------------------------------------------------------
    for a in budget["a_values"]:
        sys_dict = load_benchmark("rotation_family", a=a)
        sys_inst = sys_dict["instance"]
        dt = float(sys_inst.dt)
        assert abs(dt - EVAL_DT["rotation_family"]) < 1e-12, "benchmark dt drift"

        eval_path, eval_hash, eval_meta = _generate(
            "rotation_family", budget["eval_trajectories"], budget["T_eval"],
            dt, EVAL_SEED, data_dir, sys_kwargs={"a": a},
        )
        eval_data = torch.load(eval_path)
        eval_cache[a] = eval_path

        entry = {
            "a": float(a),
            "budget": {
                "num_trajectories": budget["num_trajectories"],
                "T_train": budget["T_train"],
                "T_eval": budget["T_eval"],
                "epochs": budget["epochs"],
                "window_len": P2_BASELINE_CONFIG["window_len"],
                "max_windows_per_traj": budget["max_windows_per_traj"],
                "batch_size": budget["batch_size"],
                "latent_dim": LATENT_DIMS["rotation_family"],
                "seeds": budget["seeds"],
                "dt": dt,
            },
            "datasets": {
                "eval": {
                    "path": eval_path, "sha256": eval_hash, "seed": EVAL_SEED,
                    "num_trajectories": budget["eval_trajectories"],
                    "T": budget["T_eval"], "dt": eval_meta["dt"],
                },
                "train": {},
            },
            "seeds": {},
        }

        for seed in budget["seeds"]:
            train_path, train_hash, train_meta = _generate(
                "rotation_family", budget["num_trajectories"], budget["T_train"],
                dt, seed, data_dir, sys_kwargs={"a": a},
            )
            entry["datasets"]["train"][str(seed)] = {
                "path": train_path, "sha256": train_hash, "seed": seed,
                "num_trajectories": budget["num_trajectories"],
                "T": budget["T_train"], "dt": train_meta["dt"],
            }
            train_data = torch.load(train_path)
            module, prov = train_official_baseline(
                train_data, sys_inst,
                latent_dim=LATENT_DIMS["rotation_family"],
                epochs=budget["epochs"], seed=seed,
                window_len=P2_BASELINE_CONFIG["window_len"],
                max_windows_per_traj=budget["max_windows_per_traj"],
                batch_size=budget["batch_size"],
                checkpoint_path=os.path.join(
                    ckpt_dir, f"rotation_a{a}_latent{LATENT_DIMS['rotation_family']}_seed{seed}.pt"
                ),
                dataset_sha256=train_hash,
            )
            trained_models[(a, seed)] = module
            metrics = evaluate_baseline(
                module, sys_inst, eval_path, horizons=HORIZONS, seed=seed,
                sigma_bounds=tuple(ACCEPTANCE_CRITERIA["h3_sigma_bounds"]),
            )
            metrics["training"] = prov
            metrics["coverage"] = _coverage_diagnostics(train_data, eval_data, sys_inst)
            entry["seeds"][str(seed)] = metrics

        entry["aggregate"] = _aggregate(
            entry["seeds"],
            [
                "one_step_mse", "one_step_rmse", "E_phys", "E_lift", "E_drift",
                "d_orient", "d_orient_p95",
            ],
        )
        entry["aggregate"]["E_response"] = _mean_std(
            [e["response"]["raw"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["E_response_normalized"] = _mean_std(
            [e["response"]["normalized"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["candidate_regret"] = _mean_std(
            [e["candidate_regret"]["mean"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["open_loop_mse"] = {
            H: _mean_std([e["open_loop_mse"][H]["mse"] for e in entry["seeds"].values()])
            for H in [str(h) for h in HORIZONS]
        }
        entry["aggregate"]["rho_T"] = _mean_std(
            [e["gramian"]["rho_T"] for e in entry["seeds"].values()
             if e["gramian"]["rho_T"] is not None]
        )
        report["rotation_sweep"][f"a_{a}"] = entry

    # Latent-dimension sensitivity at the strongest rotation (full protocol).
    report["latent_dim_sensitivity"] = {}
    if budget["latent_sweep"]:
        sys_dict = load_benchmark("rotation_family", a=1.0)
        sys_inst = sys_dict["instance"]
        eval_path = eval_cache[1.0]
        train_path = report["rotation_sweep"]["a_1.0"]["datasets"]["train"]["42"]["path"]
        train_data = torch.load(train_path)
        for latent_dim in [4, 6, 8]:
            if latent_dim == LATENT_DIMS["rotation_family"]:
                gramian = report["rotation_sweep"]["a_1.0"]["seeds"]["42"]["gramian"]
            else:
                module, prov = train_official_baseline(
                    train_data, sys_inst, latent_dim=latent_dim,
                    epochs=budget["epochs"], seed=42,
                    window_len=P2_BASELINE_CONFIG["window_len"],
                    max_windows_per_traj=budget["max_windows_per_traj"],
                    batch_size=budget["batch_size"],
                    checkpoint_path=os.path.join(
                        ckpt_dir, f"rotation_a1.0_latent{latent_dim}_seed42.pt"
                    ),
                    dataset_sha256=report["rotation_sweep"]["a_1.0"]["datasets"]["train"]["42"]["sha256"],
                )
                A, B = module.model.operator.get_AB()
                gramian = compute_continuous_gramian_and_margin(
                    A.detach().numpy(), B.detach().numpy(),
                    np.eye(2, A.shape[0]), T=1.0,
                )
            report["latent_dim_sensitivity"][str(latent_dim)] = {
                "rho_T": gramian["rho_T"],
                "rank_deficient": gramian["rank_deficient"],
                "cond_Wy": gramian["cond_Wy"],
                "lambda_min_Wy": gramian["lambda_min_Wy"],
            }

    # ------------------------------------------------------------------
    # E07: unicycle and planar quadrotor (same original baseline)
    # ------------------------------------------------------------------
    for sys_name in ("unicycle", "planar_quadrotor"):
        sys_dict = load_benchmark(sys_name)
        sys_inst = sys_dict["instance"]
        dt = float(sys_inst.dt)
        assert abs(dt - EVAL_DT[sys_name]) < 1e-12, "benchmark dt drift"
        latent_dim = LATENT_DIMS[sys_name]

        eval_path, eval_hash, eval_meta = _generate(
            sys_name, budget["eval_trajectories"], budget["T_eval"], dt,
            EVAL_SEED, data_dir,
        )
        eval_data = torch.load(eval_path)
        entry = {
            "budget": {
                "num_trajectories": budget["num_trajectories"],
                "T_train": budget["T_train"],
                "T_eval": budget["T_eval"],
                "epochs": budget["epochs"],
                "window_len": P2_BASELINE_CONFIG["window_len"],
                "max_windows_per_traj": budget["max_windows_per_traj"],
                "batch_size": budget["batch_size"],
                "latent_dim": latent_dim,
                "seeds": budget["seeds"],
                "dt": dt,
            },
            "datasets": {
                "eval": {
                    "path": eval_path, "sha256": eval_hash, "seed": EVAL_SEED,
                    "num_trajectories": budget["eval_trajectories"],
                    "T": budget["T_eval"], "dt": eval_meta["dt"],
                },
                "train": {},
            },
            "seeds": {},
        }
        for seed in budget["seeds"]:
            train_path, train_hash, train_meta = _generate(
                sys_name, budget["num_trajectories"], budget["T_train"], dt, seed, data_dir
            )
            entry["datasets"]["train"][str(seed)] = {
                "path": train_path, "sha256": train_hash, "seed": seed,
                "num_trajectories": budget["num_trajectories"],
                "T": budget["T_train"], "dt": train_meta["dt"],
            }
            train_data = torch.load(train_path)
            module, prov = train_official_baseline(
                train_data, sys_inst, latent_dim=latent_dim,
                epochs=budget["epochs"], seed=seed,
                window_len=P2_BASELINE_CONFIG["window_len"],
                max_windows_per_traj=budget["max_windows_per_traj"],
                batch_size=budget["batch_size"],
                checkpoint_path=os.path.join(
                    ckpt_dir, f"{sys_name}_latent{latent_dim}_seed{seed}.pt"
                ),
                dataset_sha256=train_hash,
            )
            if sys_name == "unicycle":
                trained_models[(sys_name, seed)] = module
            metrics = evaluate_baseline(
                module, sys_inst, eval_path, horizons=HORIZONS, seed=seed,
                sigma_bounds=tuple(ACCEPTANCE_CRITERIA["h3_sigma_bounds"]),
            )
            metrics["training"] = prov
            metrics["coverage"] = _coverage_diagnostics(train_data, eval_data, sys_inst)
            entry["seeds"][str(seed)] = metrics
        entry["aggregate"] = _aggregate(
            entry["seeds"],
            ["one_step_mse", "one_step_rmse", "E_phys", "E_lift", "E_drift", "d_orient"],
        )
        entry["aggregate"]["E_response"] = _mean_std(
            [e["response"]["raw"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["candidate_regret"] = _mean_std(
            [e["candidate_regret"]["mean"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["open_loop_mse"] = {
            H: _mean_std([e["open_loop_mse"][H]["mse"] for e in entry["seeds"].values()])
            for H in [str(h) for h in HORIZONS]
        }
        report["benchmark_evaluations"][sys_name] = entry

    # ------------------------------------------------------------------
    # E08: closed-loop QP MPC, learned model vs LTI-oracle and zero control
    # ------------------------------------------------------------------
    for a_val in ([0.0, 1.0] if quick_mode else [0.0, 1.0]):
        sys_dict = load_benchmark("rotation_family", a=a_val)
        sys_inst = sys_dict["instance"]
        for seed in budget["seeds"]:
            module = trained_models[(a_val, seed)]
            A, B = module.model.operator.get_AB()
            x0 = np.array([1.0, 0.5])
            target = np.zeros(2)
            learned = _mpc_eval(
                sys_inst, A.detach().numpy(), B.detach().numpy(),
                np.eye(2, A.shape[0]), module.model.encoder,
                x0, target, budget["mpc_steps"], budget["mpc_horizon"], seed,
            )
            A_lin, B_lin = _linearization(sys_inst, target)
            oracle = _mpc_eval(
                sys_inst, A_lin, B_lin, np.eye(2), None,
                x0, target, budget["mpc_steps"], budget["mpc_horizon"], seed,
            )
            zero = _zero_control_cost(
                sys_inst, x0, target, budget["mpc_steps"],
                np.diag([10.0, 1.0]), 0.1 * np.eye(1),
            )
            comparable = (
                learned["solver"]["success_rate"] >= ACCEPTANCE_CRITERIA["mpc_success_rate_min"]
                and oracle["solver"]["success_rate"] >= ACCEPTANCE_CRITERIA["mpc_success_rate_min"]
            )
            report["mpc_evaluations"][f"rotation_a_{a_val}_seed{seed}"] = {
                "learned": learned,
                "oracle_linearization": oracle,
                "zero_control": zero,
                "comparison": {
                    "same_plant": True,
                    "same_target": True,
                    "same_action_bounds": True,
                    "same_cost_weights": True,
                    "same_dt": True,
                    "learned_vs_oracle_cost_ratio": (
                        learned["total_cost"] / oracle["total_cost"]
                        if oracle["total_cost"] > 0 else None
                    ),
                    "learned_vs_zero_cost_ratio": (
                        learned["total_cost"] / zero["total_cost"]
                        if zero["total_cost"] > 0 else None
                    ),
                    "attribution": (
                        "comparable: both controllers solved all steps"
                        if comparable
                        else "NOT_ATTRIBUTABLE: solver success differs or failed"
                    ),
                },
            }

    # ------------------------------------------------------------------
    # H3 summary
    # ------------------------------------------------------------------
    by_rotation = {}
    for key, entry in report["rotation_sweep"].items():
        by_rotation[key] = [e["h3"] for e in entry["seeds"].values()]
    report["h3_feasibility"] = {
        "declared_sigma_bounds": ACCEPTANCE_CRITERIA["h3_sigma_bounds"],
        "by_rotation": by_rotation,
        "note": (
            "d_star is a pointwise constrained factorization distance; for multi-input "
            "systems it is an SLSQP approximation. It is not a global H3 certificate."
        ),
    }

    # ------------------------------------------------------------------
    # Failure-mode separation (measurements only; verdicts live in gates)
    # ------------------------------------------------------------------
    report["failure_mode_separation"] = {
        "structural_model_class": {
            "measure": "E_response (normalized) vs rotation a",
            "values": {
                key: entry["aggregate"]["E_response_normalized"]
                for key, entry in report["rotation_sweep"].items()
            },
            "E_lift": {
                key: entry["aggregate"]["E_lift"]
                for key, entry in report["rotation_sweep"].items()
            },
            "d_orient": {
                key: entry["aggregate"]["d_orient"]
                for key, entry in report["rotation_sweep"].items()
            },
        },
        "data_coverage": {
            "per_config": {
                key: {
                    seed: {
                        "state_distance_p95": e["coverage"]["state_distance_p95"],
                        "input_covariance_condition": e["coverage"]["input_covariance_condition"],
                        "min_action_bin_fraction": e["coverage"]["min_action_bin_fraction"],
                    }
                    for seed, e in entry["seeds"].items()
                }
                for key, entry in report["rotation_sweep"].items()
            }
        },
        "optimization": {
            "per_config": {
                key: {
                    seed: {
                        "final_train_loss": e["training"]["trainer"]["final_train_loss"],
                        "one_step_mse": e["one_step_mse"],
                        "open_loop_mse_50": e["open_loop_mse"]["50"]["mse"],
                    }
                    for seed, e in entry["seeds"].items()
                }
                for key, entry in report["rotation_sweep"].items()
            }
        },
        "solver_numerical": {
            key: ev["learned"]["solver"]
            for key, ev in report["mpc_evaluations"].items()
        },
    }

    report["gates"] = _derive_gates(report)
    report["summary"] = {
        "wall_time_s": float(time.time() - started),
        "verdicts": {k: v["verdict"] for k, v in report["gates"].items()},
        "interpretation": (
            "P2 verdicts are derived from measured metrics and the pre-registered "
            "criteria in acceptance_criteria. H2 and H3 remain independent tracks."
        ),
    }

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    a_pts = [report["rotation_sweep"][k]["a"] for k in sorted(report["rotation_sweep"], key=lambda k: report["rotation_sweep"][k]["a"])]
    keys = sorted(report["rotation_sweep"], key=lambda k: report["rotation_sweep"][k]["a"])

    def _mean_series(metric):
        vals = []
        errs = []
        for k in keys:
            agg = report["rotation_sweep"][k]["aggregate"].get(metric)
            if isinstance(agg, dict):
                vals.append(agg["mean"])
                errs.append(agg["std"])
            else:
                vals.append(np.nan)
                errs.append(0.0)
        return np.asarray(vals), np.asarray(errs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mse, mse_e = _mean_series("one_step_mse")
    resp, resp_e = _mean_series("E_response_normalized")
    axes[0].errorbar(a_pts, mse, yerr=mse_e, marker="o", label="One-step prediction MSE")
    axes[0].errorbar(a_pts, resp, yerr=resp_e, marker="s", linestyle="--", label="Normalized $E_{response}$")
    axes[0].set_xlabel("Rotation parameter $a$")
    axes[0].set_ylabel("Error")
    axes[0].set_yscale("log")
    axes[0].set_title("Prediction vs action-response error")
    axes[0].grid(True)
    axes[0].legend()

    dor, dor_e = _mean_series("d_orient")
    axes[1].errorbar(a_pts, dor, yerr=dor_e, marker="^", label="Orientation mismatch $d_{orient}$")
    axes[1].set_xlabel("Rotation parameter $a$")
    axes[1].set_ylabel("$d_{orient}$")
    axes[1].set_title("Input-direction subspace mismatch")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "rotation_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    report = _sanitize(report)
    report_path = os.path.join(output_dir, "p2_failure_report.yaml")
    with open(report_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return report_path


def _mean_std(values) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": len(vals),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/diagnosis")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out_file = run_p2_stress_diagnosis(output_dir=args.output_dir, quick_mode=args.quick)
    print(f"P2 diagnosis report saved to {out_file}")
