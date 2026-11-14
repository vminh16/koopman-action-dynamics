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
import hashlib
import os
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from research.baseline.common import (
    ROOT,
    effective_hyperparameters,
    environment_provenance,
    git_provenance,
    sha256_file,
)
from research.baseline.data_generator import generate_benchmark_dataset, rk4_step
from research.benchmarks import load_benchmark
from research.control.qp_mpc import KoopmanConvexMPC
from research.diagnostics.metrics import (
    build_candidate_pool,
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
EXPERIMENTS = ("E05", "E06", "E07", "E08", "E09")

# Source identity: a dirty working tree is allowed only when this snapshot hash
# plus the recorded command make the run reproducible.
SOURCE_FILES = (
    "research/diagnostics/run_p2_diagnostics.py",
    "research/diagnostics/metrics.py",
    "research/control/qp_mpc.py",
    "research/baseline/data_generator.py",
    "research/baseline/common.py",
    "training/train.py",
    "training/loss.py",
    "my_utils/dataset.py",
)

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
# provenance and run identity
# ---------------------------------------------------------------------------
def source_snapshot() -> dict:
    """Hash the exact files that produced a run, so a dirty tree is identifiable."""
    entries = {}
    for rel in SOURCE_FILES:
        path = ROOT / rel
        entries[rel] = sha256_file(path) if path.exists() else None
    blob = "\n".join(f"{k}:{v}" for k, v in sorted(entries.items()))
    return {
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "files": entries,
        "command": " ".join([sys.executable, *sys.argv]),
    }


def _run_identity(output_dir: str, quick_mode: bool) -> tuple[str, str, str, str]:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head = git_provenance()["git_head"][:7] or "nogit"
    run_id = f"P2-{'quick' if quick_mode else 'full'}-{stamp}-{head}"
    run_dir = os.path.join(output_dir, "runs", run_id)
    data_dir = os.path.join(run_dir, "data")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return run_id, run_dir, data_dir, ckpt_dir


def _archive_previous_report(output_dir: str) -> None:
    """Keep earlier reports as history; never overwrite the only copy."""
    path = os.path.join(output_dir, "p2_failure_report.yaml")
    if not os.path.exists(path):
        return
    hist = os.path.join(output_dir, "history")
    os.makedirs(hist, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(hist, f"p2_failure_report_{stamp}.yaml")
    if os.path.exists(dest):
        return
    with open(path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())


def _artifact(kind: str, experiment: str, path: str, sha256: str | None = None) -> dict:
    return {
        "kind": kind,
        "experiment": experiment,
        "path": os.path.abspath(path),
        "sha256": sha256 or (sha256_file(path) if os.path.exists(path) else None),
        "exists": os.path.exists(path),
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - optional dependency
        return None



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
    eff = effective_hyperparameters(module, cfg)

    prov = {
        "model_class": type(module).__name__,
        "module_entrypoint": "training.train:KoopmanLitModule",
        "parameterization": type(module.model.operator.core).__name__,
        "config": _model_config(latent_dim, cfg),
        "effective": eff,
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
            "optimizer": eff["optimizer"],
            "lr": eff["lr"],
            "weight_decay": eff["weight_decay"],
            "device": "cpu",
            "final_train_loss": _finite_or_none(
                trainer.callback_metrics.get("train_total_loss")
            ),
            "global_step": int(trainer.global_step),
        },
        "expectation_horizon_seconds": 15 * float(module.dt),
        "state_dict_shapes": {k: list(v.shape) for k, v in module.state_dict().items()},
    }
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(module.state_dict(), checkpoint_path)
        prov["checkpoint"] = {
            "path": os.path.abspath(checkpoint_path),
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
def _stack_transitions(data, with_ids: bool = False):
    """(x_t, u_t, x_{t+1}) for t=0..T-1, including x_0."""
    x_list, u_list, next_list, ids = [], [], [], []
    for i, traj in enumerate(data):
        x_seq = traj["x_seq"] if traj["x_seq"].ndim == 2 else traj["x_seq"].squeeze(1)
        u_seq = traj["u_seq"]
        x0 = traj["x0"].reshape(1, -1)
        x_t = torch.cat([x0, x_seq[:-1]], dim=0)
        x_list.append(x_t)
        u_list.append(u_seq)
        next_list.append(x_seq)
        ids.extend([i] * int(u_seq.shape[0]))
    out = (torch.cat(x_list), torch.cat(u_list), torch.cat(next_list))
    if with_ids:
        return (*out, ids)
    return out


def _ci(values) -> dict:
    """Mean/std/95% CI across trajectories or seeds."""
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _per_trajectory_mse(pred, true, ids) -> dict:
    grouped = defaultdict(list)
    for i, traj_id in enumerate(ids):
        grouped[traj_id].append(float(torch.mean((pred[i] - true[i]) ** 2).item()))
    return _ci([np.mean(v) for v in grouped.values()])


def _state_stats(data) -> dict:
    x = torch.cat(
        [torch.cat([t["x0"].reshape(1, -1), t["x_seq"]], dim=0) for t in data], dim=0
    )
    return {
        "mean": x.mean(dim=0),
        "std": x.std(dim=0).clamp_min(1e-6),
    }


def build_response_probe(data, sys_inst, num_samples: int = 64, seed: int = EVAL_SEED) -> dict:
    """One (x, u, u0) probe shared by every rotation strength and seed."""
    X, _, _, ids = _stack_transitions(data, with_ids=True)
    torch.manual_seed(seed)
    idx = torch.randperm(X.shape[0])[: min(num_samples, X.shape[0])]
    u_rand = torch.rand(idx.numel(), sys_inst.Ncontrols)
    u = sys_inst.umin + u_rand * (sys_inst.umax - sys_inst.umin)
    return {
        "x": X[idx],
        "u": u,
        "u_ref": torch.zeros_like(u),
        "indices": idx.tolist(),
        "trajectory_ids": [ids[i] for i in idx.tolist()],
        "seed": int(seed),
    }


def _fixed_b_reference(module, sys_inst, probe, device=None) -> dict:
    """Best constant input map on the same probe (fixed-B ceiling)."""
    from research.diagnostics.metrics import extract_latent_model, physical_input_map

    spec = extract_latent_model(module)
    x = probe["x"]
    _, Gx = physical_input_map(sys_inst, x)
    n = int(sys_inst.Nstates)
    C = torch.eye(n, spec["latent_dim"])
    CB = (C @ spec["B"].detach().cpu()).numpy()
    Gx_np = Gx.detach().cpu().numpy()
    best_const = Gx_np.mean(axis=0)
    learned_err = float(np.sum((CB - best_const) ** 2))
    oracle0_err = float(np.sum((Gx_np - best_const[None, :, :]) ** 2) / max(len(x), 1))
    return {
        "probe_size": int(x.shape[0]),
        "learned_CB_frobenius_sq_to_best_constant": learned_err,
        "best_constant_Gx_mean": best_const.tolist(),
        "oracle_a0_variance_floor": oracle0_err,
        "note": "lower is better; oracle at a=0 has a constant Gx, so its floor is 0",
    }


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


def _state_array(data) -> torch.Tensor:
    rows = []
    for t in data:
        x_seq = t["x_seq"] if t["x_seq"].ndim == 2 else t["x_seq"].squeeze(1)
        rows.append(torch.cat([t["x0"].reshape(1, -1), x_seq], dim=0))
    return torch.cat(rows, dim=0)


def _state_action_rows(data) -> tuple[torch.Tensor, torch.Tensor]:
    """State rows aligned with the action applied at that state (drops x_T)."""
    xs, us = [], []
    for t in data:
        x_seq = t["x_seq"] if t["x_seq"].ndim == 2 else t["x_seq"].squeeze(1)
        xs.append(torch.cat([t["x0"].reshape(1, -1), x_seq[:-1]], dim=0))
        us.append(t["u_seq"])
    return torch.cat(xs, dim=0), torch.cat(us, dim=0)


def _local_input_stats(train_x, train_u, query_x, k: int = 32, num_bins: int = 5, sys_inst=None) -> dict:
    """Local design excitation at query states (EXPERIMENT §4.6)."""
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    d = torch.cdist((query_x - mean) / std, (train_x - mean) / std)
    kk = min(int(k), int(train_x.shape[0]))
    nearest = d.topk(kk, dim=1, largest=False)
    U = train_u.detach().cpu().numpy()
    conds, sv_mins, counts = [], [], []
    for i in range(query_x.shape[0]):
        idx = nearest.indices[i].numpy()
        local = U[idx]
        counts.append(int(local.shape[0]))
        if local.shape[1] == 1:
            var = float(np.var(local, ddof=1)) if local.shape[0] > 1 else 0.0
            sv = np.asarray([np.sqrt(max(var, 0.0))])
            cond = 1.0 if var > 1e-12 else None
        else:
            cov = np.cov(local, rowvar=False)
            sv = np.linalg.svd(cov, compute_uv=False)
            cond = float(sv.max() / sv.min()) if sv.min() > 1e-12 else None
        sv_mins.append(float(np.min(sv)))
        conds.append(cond)
    valid = [c for c in conds if c is not None]
    out = {
        "num_queries": int(query_x.shape[0]),
        "neighbors_per_query": int(kk),
        "samples_per_local_covariance": {"min": int(min(counts)), "mean": float(np.mean(counts))},
        "nearest_train_distance_p95": float(torch.quantile(nearest.values[:, -1], 0.95)),
        "nearest_train_distance_max": float(nearest.values[:, -1].max()),
        "local_singular_value_min": {
            "min": float(np.min(sv_mins)), "mean": float(np.mean(sv_mins)),
        },
        "local_covariance_condition": {
            "mean": float(np.mean(valid)) if valid else None,
            "p95": float(np.quantile(valid, 0.95)) if valid else None,
            "max": float(np.max(valid)) if valid else None,
            "undefined_rank_deficient": int(len(conds) - len(valid)),
        },
    }
    return out


def _coverage_diagnostics(train, eval, sys_inst, num_bins: int = 5, query_x=None) -> dict:
    train_x, eval_x = _state_array(train), _state_array(eval)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    train_z = (train_x - mean) / std
    eval_z = (eval_x - mean) / std
    nearest = torch.cdist(eval_z, train_z).min(dim=1).values

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
    out = {
        "state_distance_p95": float(torch.quantile(nearest, 0.95).item()),
        "state_distance_max": float(nearest.max().item()),
        "input_covariance_condition": cond,
        "input_covariance_min_eig": float(eig.min()),
        "input_covariance_singular_values": np.sqrt(np.clip(eig, 0.0, None)).tolist(),
        "min_action_bin_fraction": float(min(bin_occupancy)),
        "num_train_states": int(train_x.shape[0]),
        "num_eval_states": int(eval_x.shape[0]),
        "normalization": "train mean/std",
        "visited_states_eval_to_train": {
            "p95": float(torch.quantile(nearest, 0.95).item()),
            "max": float(nearest.max().item()),
        },
        "visited_states_train_self": {
            "p95": float(torch.quantile(torch.cdist(train_z, train_z).min(dim=1).values, 0.95).item()),
        },
    }
    if query_x is not None:
        local_x, local_u = _state_action_rows(train)
        out["query_states"] = _local_input_stats(
            local_x, local_u, query_x, sys_inst=sys_inst
        )
    return out


def evaluate_baseline(
    module,
    sys_inst,
    eval_path: str,
    horizons=HORIZONS,
    response_samples: int = 64,
    diagnostic_samples: int = 8,
    seed: int = 0,
    sigma_bounds=(0.5, 2.0),
    probe: dict | None = None,
    train_stats: dict | None = None,
    gramian_T: float | None = None,
) -> dict:
    data = torch.load(eval_path)
    X, U, X_next, traj_ids = _stack_transitions(data, with_ids=True)
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
        per_traj = ((pred_seq - truth) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        entry = {
            "mse": _mse(pred_seq, truth),
            "final_step_mse": _mse(pred_seq[:, -1], truth[:, -1]),
            "mse_ci95_by_trajectory": _ci(per_traj),
            "free_running": True,
            "reencode": False,
        }
        if train_stats is not None:
            std = train_stats["std"].to(pred_seq.dtype)
            std_pred = (pred_seq - train_stats["mean"].to(pred_seq.dtype)) / std
            std_true = (truth - train_stats["mean"].to(pred_seq.dtype)) / std
            entry["standardized_mse"] = _mse(std_pred, std_true)
        open_loop[str(H)] = entry

        if H == horizons[-1]:
            k = min(3, z_seq.shape[0])
            manifold = compute_manifold_error(module, z_seq[:k])

    if probe is not None:
        p_x, p_u, p_u0 = probe["x"], probe["u"], probe["u_ref"]
    else:
        torch.manual_seed(seed)
        idx = torch.randperm(X.shape[0])[: min(response_samples, X.shape[0])]
        p_x = X[idx]
        u_rand = torch.rand(p_x.shape[0], sys_inst.Ncontrols)
        p_u = sys_inst.umin + u_rand * (sys_inst.umax - sys_inst.umin)
        p_u0 = torch.zeros_like(p_u)
    response = compute_action_response_error(module, sys_inst, p_x, p_u, p_u0)
    response["E_response"] = response["raw"]
    response["probe"] = {
        "shared": probe is not None,
        "size": int(p_x.shape[0]),
        "seed": int(probe["seed"]) if probe is not None else int(seed),
        "action_pair": "u sampled in [umin,umax], u0=0",
    }
    if probe is not None:
        response["probe"]["trajectory_ids"] = probe.get("trajectory_ids")

    torch.manual_seed(seed)
    diag_idx = torch.randperm(X.shape[0])[: max(diagnostic_samples, 1)]
    vf = compute_vector_field_mismatches(module, sys_inst, X[diag_idx])

    A, B = module.model.operator.get_AB()
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    C = np.eye(n, A.shape[0])
    T_eval = float(gramian_T) if gramian_T is not None else 15.0 * float(sys_inst.dt)
    gramian = compute_continuous_gramian_and_margin(A, B, C, T=T_eval, method="expm")
    gramian["horizon_convention"] = (
        "T = MPC horizon * dt" if gramian_T is not None else "T = upstream 15*dt"
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
    h3["rho_T_horizon_T"] = T_eval

    pool = build_candidate_pool(
        sys_inst.umin, sys_inst.umax, horizon=3, num_sequences=8, seed=seed
    )
    regret = compute_candidate_regret(
        module, sys_inst, X[diag_idx], candidate_sequences=pool,
        horizon=3, target=torch.zeros(n),
    )
    regret_re = compute_candidate_regret(
        module, sys_inst, X[diag_idx], candidate_sequences=pool,
        horizon=3, target=torch.zeros(n), reencode=True,
    )
    regret["reencode"] = regret_re["reencode"]

    alignment = _verify_transition_replay(data, sys_inst)
    fixed_b = _fixed_b_reference(module, sys_inst, probe) if probe is not None else None

    return {
        "num_transitions": int(X.shape[0]),
        "one_step_mse": one_step_mse,
        "one_step_rmse": one_step_rmse,
        "one_step_ci95_by_trajectory": _per_trajectory_mse(pred, X_next, traj_ids),
        "derivative_error": {
            "mse": derivative_mse,
            "relative_mean": derivative_rel,
            "space": "physical dx/dt",
        },
        "open_loop_mse": open_loop,
        "manifold_error": manifold,
        "response": response,
        "fixed_b_reference": fixed_b,
        "E_phys": vf["E_phys"],
        "E_lift": vf["E_lift"],
        "E_drift": vf["E_drift"],
        "d_orient": vf["d_orient"],
        "d_orient_p95": vf["d_orient_p95"],
        "d_orient_max": vf["d_orient_max"],
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


def _cost_weights(sys_inst):
    n = int(sys_inst.Nstates)
    q = np.ones(n)
    q[: min(2, n)] = 10.0
    return np.diag(q), 0.1 * np.eye(int(sys_inst.Ncontrols))


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


def _mpc_eval(sys_inst, A, B, C, encoder_fn, x0, target, steps, horizon, seed,
              discretization: str = "zoh"):
    torch.manual_seed(seed)
    Q, R = _cost_weights(sys_inst)
    mpc = KoopmanConvexMPC(
        A=A, B=B, C=C, dt=sys_inst.dt, horizon=horizon, Q=Q, R=R,
        umin=sys_inst.umin.numpy(), umax=sys_inst.umax.numpy(),
        encoder_fn=encoder_fn, discretization=discretization,
    )
    traj = mpc.simulate_closed_loop(
        sys_inst, x0=np.asarray(x0), x_target=np.asarray(target), num_steps=steps
    )
    return {
        "total_cost": float(traj["total_cost"]),
        "final_error": float(traj["final_error"]),
        "u_abs_mean": float(np.mean(np.abs(np.asarray(traj["u_history"])))),
        "tracking_rmse": float(
            np.sqrt(np.mean([
                np.sum((np.asarray(x) - np.asarray(target)) ** 2)
                for x in traj["x_history"]
            ]))
        ),
        "final_state": np.asarray(traj["x_history"][-1]).tolist(),
        "x_history": np.asarray(traj["x_history"]).tolist(),
        "u_history": np.asarray(traj["u_history"]).tolist(),
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
            "discretization": discretization,
        },
    }


def _lifted_step_consistency(module, sys_inst, x_history, u_history, dt: float) -> dict:
    """Teacher-forced one-step RK4 vs exact-ZOH and Euler lifted predictions."""
    from scipy.linalg import expm

    A, B = module.model.operator.get_AB()
    A = A.detach().cpu().numpy().astype(np.float64)
    B = B.detach().cpu().numpy().astype(np.float64)
    n = int(sys_inst.Nstates)
    Mz = np.zeros((A.shape[0] + B.shape[1],) * 2)
    Mz[: A.shape[0], : A.shape[0]] = A
    Mz[: A.shape[0], A.shape[0] :] = B
    E = expm(Mz * dt)
    Ad_z, Bd_z = E[: A.shape[0], : A.shape[0]], E[: A.shape[0], A.shape[0] :]
    Ad_e, Bd_e = np.eye(A.shape[0]) + dt * A, dt * B
    err_z, err_e = [], []
    with torch.no_grad():
        for x, u in zip(x_history[:-1], u_history):
            xt = torch.as_tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)
            ut = torch.as_tensor(np.asarray(u), dtype=torch.float32).unsqueeze(0)
            z = module.model.encode(xt).squeeze(0).double().numpy()
            u_np = np.asarray(u, dtype=np.float64)
            x_true = rk4_step(sys_inst.dynamics, 0.0, xt, ut, dt).squeeze(0).numpy()
            z_z = Ad_z @ z + Bd_z @ u_np
            z_e = Ad_e @ z + Bd_e @ u_np
            err_z.append(float(np.linalg.norm(z_z[:n] - x_true)))
            err_e.append(float(np.linalg.norm(z_e[:n] - x_true)))
    return {
        "zoh_rmse": float(np.sqrt(np.mean(np.square(err_z)))) if err_z else None,
        "euler_rmse": float(np.sqrt(np.mean(np.square(err_e)))) if err_e else None,
        "training_integration": "euler",
        "note": "one-step teacher-forced physical-state error on the MPC trajectory",
    }


# ---------------------------------------------------------------------------
# E05: checkpoint margin/Jacobian diagnostics on verified checkpoints
# ---------------------------------------------------------------------------
E05_CONFIGS = {
    "pendulum": "configs/research/pendulum_controllable_win15_frac0.05_seed42.yaml",
}


def _find_assets(pattern: str) -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob(pattern))


def _e05_missing(reason: str) -> dict:
    return {"status": "missing_checkpoint", "reason": reason}


def evaluate_e05_system(system: str, config_path: str | None, checkpoint_path: str | None) -> dict:
    if not config_path or not checkpoint_path:
        return _e05_missing(f"no verified config/checkpoint registered for {system}")
    from dynamics import SYSTEM_REGISTRY
    from research.baseline.common import build_module, load_config

    cfg = load_config(ROOT / config_path)
    ckpt = ROOT / checkpoint_path
    if not ckpt.exists():
        return _e05_missing(f"checkpoint not found: {checkpoint_path}")
    sys_inst = SYSTEM_REGISTRY[cfg["system"]]()
    module = build_module(cfg)
    module.load_state_dict(torch.load(ckpt, weights_only=True), strict=True)
    module.eval()

    hashes = {}
    for name in ("train", "test"):
        rel = cfg.get(f"data_{name}")
        if rel is None:
            continue
        actual = sha256_file(ROOT / rel)
        expected = cfg.get(f"data_{name}_sha256")
        hashes[name] = {
            "path": rel, "sha256": actual, "expected_sha256": expected,
            "matches": expected is None or actual == expected,
        }
    if any(h["matches"] is False for h in hashes.values()):
        return {"status": "hash_mismatch", "dataset_hashes": hashes}

    data = torch.load(ROOT / cfg["data_test"])
    X = _state_array(data)
    torch.manual_seed(EVAL_SEED)
    idx = torch.randperm(X.shape[0])[:64]
    vf = compute_vector_field_mismatches(module, sys_inst, X[idx])
    A, B = module.model.operator.get_AB()
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    T_train = 15.0 * float(sys_inst.dt)
    gramian = compute_continuous_gramian_and_margin(
        A, B, np.eye(sys_inst.Nstates, A.shape[0]), T=T_train, method="expm"
    )
    return {
        "status": "verified",
        "config": config_path,
        "checkpoint": _artifact("checkpoint", "E05", str(ckpt)),
        "dataset_hashes": hashes,
        "state_samples": int(idx.numel()),
        "rho_T": gramian["rho_T"],
        "rank_deficient": gramian["rank_deficient"],
        "gramian": {k: gramian[k] for k in ("rho_T", "lambda_min_Wy", "cond_Wy", "T")},
        "E_lift": vf["E_lift"],
        "E_lift_p95": vf["E_lift_p95"],
        "E_phys": vf["E_phys"],
        "d_orient": vf["d_orient"],
        "d_orient_p95": vf["d_orient_p95"],
        "d_orient_max": vf["d_orient_max"],
        "G_z_star_convention": vf["d_orient_convention"],
    }


def run_e05_diagnostics() -> dict:
    systems = {}
    for system, cfg_path in E05_CONFIGS.items():
        cfg = yaml.safe_load((ROOT / cfg_path).read_text())
        systems[system] = evaluate_e05_system(system, cfg_path, cfg.get("checkpoint"))
    # CartPole is only registered if a config + checkpoint actually exist.
    cart_cfg = _find_assets("*cartpole*.yaml")
    cart_ckpt = sorted(p.name for p in (ROOT / "checkpoints").glob("*cartpole*.pt"))
    if cart_cfg and cart_ckpt:
        cfg = yaml.safe_load((ROOT / cart_cfg[0]).read_text())
        systems["cartpole"] = evaluate_e05_system(
            "cartpole", cart_cfg[0], cfg.get("checkpoint", f"checkpoints/{cart_ckpt[0]}")
        )
    else:
        systems["cartpole"] = _e05_missing(
            "no verified CartPole config/checkpoint in configs/ or checkpoints/"
        )

    verified = [k for k, v in systems.items() if v.get("status") == "verified"]
    status = "verified" if len(verified) == len(systems) else (
        "partial" if verified else "missing_checkpoint"
    )
    return {
        "experiment": "E05",
        "status": status,
        "verified_systems": verified,
        "missing_systems": [k for k in systems if k not in verified],
        "systems": systems,
    }


def _reduced_excitation_data(train_data, sys_inst, seed: int = 0) -> list[dict]:
    """Build a dataset whose inputs are nearly rank-one (local excitation loss)."""
    torch.manual_seed(seed)
    m = int(sys_inst.Ncontrols)
    out = []
    for traj in train_data:
        T = traj["u_seq"].shape[0]
        if m == 1:
            val = float(sys_inst.umin[0] + 0.95 * (sys_inst.umax[0] - sys_inst.umin[0]))
            u_seq = torch.full((T, 1), val)
        else:
            u0 = sys_inst.umin[0] + torch.rand(T) * (sys_inst.umax[0] - sys_inst.umin[0])
            u_seq = u0[:, None].repeat(1, m)
            u_seq[:, 1:] *= 0.9
            u_seq = torch.max(torch.min(u_seq, sys_inst.umax), sys_inst.umin)
        x = traj["x0"].reshape(-1)
        x_seq = []
        for t in range(T):
            x = rk4_step(
                sys_inst.dynamics, 0.0, x.unsqueeze(0), u_seq[t : t + 1], sys_inst.dt
            ).squeeze(0)
            x_seq.append(x)
        out.append({"x0": traj["x0"], "u_seq": u_seq, "x_seq": torch.stack(x_seq)})
    return out



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


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _derive_verification(report: dict, budget: dict, quick_mode: bool) -> dict:
    """Full-evidence gate. Quick runs are explicitly not verification runs."""
    if quick_mode:
        return {
            "verdict": "SMOKE_ONLY",
            "criteria": {"reason": "quick mode is a wiring smoke test, not evidence"},
        }

    checks = {}
    exp = report.get("experiments", {})
    checks["experiments_declared"] = set(EXPERIMENTS) == set(exp)
    checks["experiment_status"] = {k: exp.get(k, {}).get("status") for k in EXPERIMENTS}

    # E06/E07 seed and a-set coverage, alignment, dt, finite metrics.
    missing_seeds, bad_alignment, bad_dt, nonfinite = [], [], [], []
    for key, entry in report.get("rotation_sweep", {}).items():
        expected = {str(s) for s in budget["seeds"]}
        if set(entry.get("seeds", {})) != expected:
            missing_seeds.append(f"E06/{key}")
        for seed, e in entry.get("seeds", {}).items():
            if not _finite(e.get("one_step_mse")) or not _finite(e.get("E_lift")) \
                    or not _finite(e.get("response", {}).get("raw")):
                nonfinite.append(f"E06/{key}/seed{seed}")
            dev = e.get("alignment", {}).get("max_abs_deviation")
            if dev is None or dev > 1e-5:
                bad_alignment.append(f"E06/{key}/seed{seed}")
            if abs(float(entry["budget"]["dt"]) - float(e["training"]["dt"])) > 1e-12:
                bad_dt.append(f"E06/{key}/seed{seed}/model_dt")
            if abs(float(entry["budget"]["dt"]) - float(entry["datasets"]["eval"]["dt"])) > 1e-12:
                bad_dt.append(f"E06/{key}/eval_dt")
        for seed, ds in entry.get("datasets", {}).get("train", {}).items():
            if abs(float(entry["budget"]["dt"]) - float(ds["dt"])) > 1e-12:
                bad_dt.append(f"E06/{key}/train{seed}_dt")
    for name, entry in report.get("benchmark_evaluations", {}).items():
        if set(entry.get("seeds", {})) != {str(s) for s in budget["seeds"]}:
            missing_seeds.append(f"E07/{name}")
        for seed, e in entry.get("seeds", {}).items():
            if not _finite(e.get("one_step_mse")) or not _finite(e.get("response", {}).get("raw")):
                nonfinite.append(f"E07/{name}/seed{seed}")
            dev = e.get("alignment", {}).get("max_abs_deviation")
            if dev is None or dev > 1e-5:
                bad_alignment.append(f"E07/{name}/seed{seed}")
            if abs(float(entry["budget"]["dt"]) - float(e["training"]["dt"])) > 1e-12:
                bad_dt.append(f"E07/{name}/seed{seed}/model_dt")

    checks["experiment_seed_coverage"] = not missing_seeds
    checks["missing_seed_entries"] = missing_seeds
    checks["alignment_e06_e07"] = not bad_alignment
    checks["alignment_issues"] = bad_alignment
    checks["dt_consistency"] = not bad_dt
    checks["dt_issues"] = bad_dt
    checks["finite_primary_metrics"] = not nonfinite
    checks["nonfinite_entries"] = nonfinite

    # Artifact registry: every entry must exist and match its recorded hash.
    bad_artifacts = []
    for art in report.get("artifacts", []):
        path, sha = art.get("path"), art.get("sha256")
        if not path or not sha or not os.path.exists(path):
            bad_artifacts.append({"path": path, "reason": "missing or unhashed"})
        elif sha256_file(path) != sha:
            bad_artifacts.append({"path": path, "reason": "sha256 mismatch"})
    checks["artifact_registry_verified"] = bool(report.get("artifacts")) and not bad_artifacts
    checks["artifact_issues"] = bad_artifacts

    # E05: at least one verified checkpoint; missing systems are recorded, not hidden.
    e05 = exp.get("E05", {})
    checks["e05_verified_systems"] = e05.get("verified_systems", [])
    checks["e05_missing_systems"] = e05.get("missing_systems", [])
    checks["e05_has_verified_checkpoint"] = bool(e05.get("verified_systems"))
    checks["e05_hashes_ok"] = all(
        s.get("status") != "verified"
        or all(h.get("matches") for h in s.get("dataset_hashes", {}).values())
        for s in e05.get("systems", {}).values()
    )

    checks["e08_present"] = exp.get("E08", {}).get("status") == "verified"
    checks["e09_present"] = exp.get("E09", {}).get("status") == "verified"

    prov = report.get("provenance", {})
    checks["source_identified"] = bool(
        prov.get("source_identified")
        and prov.get("source_snapshot", {}).get("sha256")
        and prov.get("source_snapshot", {}).get("command")
        and prov.get("git_head")
    )
    checks["solver_versions_recorded"] = all(
        prov.get("solver_versions", {}).get(k) for k in ("cvxpy", "osqp")
    )

    required = (
        "experiments_declared", "experiment_seed_coverage", "alignment_e06_e07",
        "dt_consistency", "finite_primary_metrics", "artifact_registry_verified",
        "e05_has_verified_checkpoint", "e05_hashes_ok", "e08_present", "e09_present",
        "source_identified", "solver_versions_recorded",
    )
    passed = all(checks[k] for k in required)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "criteria": checks,
        "required": list(required),
    }


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
        "seeds": [42, 43, 44, 45, 46],
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
    budget["a_values"] = sorted(budget["a_values"])
    os.makedirs(output_dir, exist_ok=True)
    run_id, run_dir, data_dir, ckpt_dir = _run_identity(output_dir, quick_mode)
    _archive_previous_report(output_dir)

    started = time.time()
    source = source_snapshot()
    git = git_provenance()
    env = environment_provenance()
    report = {
        "phase": "P2",
        "description": "Controlled stress tests with the original Koopman baseline",
        "run_id": run_id,
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
            "candidate_pool": "time-varying, axis-wise excitation, clipped to bounds",
            "experiments": list(EXPERIMENTS),
            "gramian_horizon": "T = mpc_horizon * dt for H3; upstream 15*dt recorded in training",
        },
        "acceptance_criteria": dict(ACCEPTANCE_CRITERIA),
        "experiments": {},
        "rotation_sweep": {},
        "benchmark_evaluations": {},
        "mpc_evaluations": {},
        "h3_feasibility": {},
        "failure_mode_separation": {},
        "artifacts": [],
        "provenance": {
            "git_head": git["git_head"],
            "git": git,
            "environment": env,
            "source_snapshot": source,
            "source_identified": bool(source["sha256"] and git["git_head"]),
            "source_clean": bool(git["working_tree_clean"]),
            "artifacts_root": os.path.abspath(run_dir),
            "solver_versions": {
                "cvxpy": _module_version("cvxpy"),
                "osqp": _module_version("osqp"),
            },
        },
    }

    def register(kind, experiment, path, sha256=None):
        art = _artifact(kind, experiment, path, sha256)
        report["artifacts"].append(art)
        return art

    trained_models = {}
    eval_cache = {}
    eval_data_cache = {}
    probe = None

    report["experiments"]["E05"] = run_e05_diagnostics()
    for system in report["experiments"]["E05"]["systems"].values():
        if system.get("status") == "verified":
            report["artifacts"].append(system["checkpoint"])

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
        eval_data_cache[a] = eval_data
        register("dataset", "E06", eval_path, eval_hash)
        if a == budget["a_values"][0]:
            probe = build_response_probe(
                eval_data, sys_inst, num_samples=64, seed=EVAL_SEED
            )

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
            register("dataset", "E06", train_path, train_hash)
            entry["datasets"]["train"][str(seed)] = {
                "path": train_path, "sha256": train_hash, "seed": seed,
                "num_trajectories": budget["num_trajectories"],
                "T": budget["T_train"], "dt": train_meta["dt"],
            }
            train_data = torch.load(train_path)
            stats = _state_stats(train_data)
            ckpt_path = os.path.join(
                ckpt_dir, f"rotation_a{a}_latent{LATENT_DIMS['rotation_family']}_seed{seed}.pt"
            )
            module, prov = train_official_baseline(
                train_data, sys_inst,
                latent_dim=LATENT_DIMS["rotation_family"],
                epochs=budget["epochs"], seed=seed,
                window_len=P2_BASELINE_CONFIG["window_len"],
                max_windows_per_traj=budget["max_windows_per_traj"],
                batch_size=budget["batch_size"],
                checkpoint_path=ckpt_path,
                dataset_sha256=train_hash,
            )
            register("checkpoint", "E06", ckpt_path, prov["checkpoint"]["sha256"])
            trained_models[(a, seed)] = module
            metrics = evaluate_baseline(
                module, sys_inst, eval_path, horizons=HORIZONS, seed=seed,
                sigma_bounds=tuple(ACCEPTANCE_CRITERIA["h3_sigma_bounds"]),
                probe=probe, train_stats=stats,
                gramian_T=float(budget["mpc_horizon"]) * dt,
            )
            metrics["training"] = prov
            metrics["coverage"] = _coverage_diagnostics(
                train_data, eval_data, sys_inst,
                query_x=(probe["x"] if probe is not None else None),
            )
            entry["seeds"][str(seed)] = metrics

        entry["aggregate"] = _aggregate(
            entry["seeds"],
            [
                "one_step_mse", "one_step_rmse", "E_phys", "E_lift", "E_drift",
                "d_orient", "d_orient_p95", "d_orient_max",
            ],
        )
        entry["aggregate"]["E_response"] = _mean_std(
            [e["response"]["raw"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["E_response_normalized"] = _mean_std(
            [e["response"]["normalized"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["E_response_ci95"] = _ci(
            [e["response"]["raw"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["fixed_b_reference"] = _mean_std(
            [e["fixed_b_reference"]["learned_CB_frobenius_sq_to_best_constant"]
             for e in entry["seeds"].values() if e.get("fixed_b_reference")]
        )
        entry["aggregate"]["candidate_regret"] = _mean_std(
            [e["candidate_regret"]["mean"] for e in entry["seeds"].values()]
        )
        entry["aggregate"]["candidate_regret_reencode"] = _mean_std(
            [e["candidate_regret"]["reencode"]["mean"] for e in entry["seeds"].values()]
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
    report["experiments"]["E06"] = {
        "experiment": "E06",
        "status": "verified",
        "a_values": budget["a_values"],
        "seeds": budget["seeds"],
        "shared_probe_across_a": True,
        "fixed_b_reference": "learned CB vs best constant mean(Gx) on the same probe",
    }

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
                    np.eye(2, A.shape[0]),
                    T=float(budget["mpc_horizon"]) * float(sys_inst.dt),
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
        eval_data_cache[sys_name] = eval_data
        register("dataset", "E07", eval_path, eval_hash)
        bench_probe = build_response_probe(
            eval_data, sys_inst, num_samples=64, seed=EVAL_SEED
        )
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
            stats = _state_stats(train_data)
            ckpt_path = os.path.join(
                ckpt_dir, f"{sys_name}_latent{latent_dim}_seed{seed}.pt"
            )
            module, prov = train_official_baseline(
                train_data, sys_inst, latent_dim=latent_dim,
                epochs=budget["epochs"], seed=seed,
                window_len=P2_BASELINE_CONFIG["window_len"],
                max_windows_per_traj=budget["max_windows_per_traj"],
                batch_size=budget["batch_size"],
                checkpoint_path=ckpt_path,
                dataset_sha256=train_hash,
            )
            register("checkpoint", "E07", ckpt_path, prov["checkpoint"]["sha256"])
            trained_models[(sys_name, seed)] = module
            metrics = evaluate_baseline(
                module, sys_inst, eval_path, horizons=HORIZONS, seed=seed,
                sigma_bounds=tuple(ACCEPTANCE_CRITERIA["h3_sigma_bounds"]),
                probe=bench_probe, train_stats=stats,
                gramian_T=float(budget["mpc_horizon"]) * dt,
            )
            metrics["training"] = prov
            metrics["coverage"] = _coverage_diagnostics(
                train_data, eval_data, sys_inst, query_x=bench_probe["x"]
            )
            entry["seeds"][str(seed)] = metrics
        entry["aggregate"] = _aggregate(
            entry["seeds"],
            ["one_step_mse", "one_step_rmse", "E_phys", "E_lift", "E_drift",
             "d_orient", "d_orient_p95", "d_orient_max"],
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
    report["experiments"]["E07"] = {
        "experiment": "E07",
        "status": "verified",
        "systems": ["unicycle", "planar_quadrotor"],
        "seeds": budget["seeds"],
        "closed_loop": "fixed initial-state sweep vs target linearization",
    }

    # ------------------------------------------------------------------
    # E08: closed-loop QP MPC, learned model vs LTI-oracle and zero control
    # ------------------------------------------------------------------
    mpc_states = []
    for a_val in budget["a_values"]:
        sys_dict = load_benchmark("rotation_family", a=a_val)
        sys_inst = sys_dict["instance"]
        for seed in budget["seeds"]:
            module = trained_models[(a_val, seed)]
            A, B = module.model.operator.get_AB()
            C = np.eye(2, A.shape[0])
            x0 = np.array([1.0, 0.5])
            target = np.zeros(2)
            learned = _mpc_eval(
                sys_inst, A.detach().numpy(), B.detach().numpy(),
                C, module.model.encoder,
                x0, target, budget["mpc_steps"], budget["mpc_horizon"], seed,
                discretization="zoh",
            )
            learned_euler = _mpc_eval(
                sys_inst, A.detach().numpy(), B.detach().numpy(),
                C, module.model.encoder,
                x0, target, budget["mpc_steps"], budget["mpc_horizon"], seed,
                discretization="euler",
            )
            A_lin, B_lin = _linearization(sys_inst, target)
            oracle = _mpc_eval(
                sys_inst, A_lin, B_lin, np.eye(2), None,
                x0, target, budget["mpc_steps"], budget["mpc_horizon"], seed,
                discretization="zoh",
            )
            Q, R = _cost_weights(sys_inst)
            zero = _zero_control_cost(
                sys_inst, x0, target, budget["mpc_steps"], Q, R,
            )
            consistency = _lifted_step_consistency(
                module, sys_inst, learned["x_history"], learned["u_history"],
                float(sys_inst.dt),
            )
            mpc_states.extend(learned["x_history"])
            comparable = (
                learned["solver"]["success_rate"] >= ACCEPTANCE_CRITERIA["mpc_success_rate_min"]
                and oracle["solver"]["success_rate"] >= ACCEPTANCE_CRITERIA["mpc_success_rate_min"]
            )
            report["mpc_evaluations"][f"rotation_a_{a_val}_seed{seed}"] = {
                "learned": learned,
                "learned_euler_discretization": learned_euler,
                "oracle_linearization": oracle,
                "zero_control": zero,
                "lifted_step_consistency": consistency,
                "comparison": {
                    "same_plant": True,
                    "same_target": True,
                    "same_action_bounds": True,
                    "same_cost_weights": True,
                    "same_dt": True,
                    "plant": "RK4 at sys.dt",
                    "oracle_type": "exact linearization of the LTI rotation family at the target",
                    "learned_vs_oracle_cost_ratio": (
                        learned["total_cost"] / oracle["total_cost"]
                        if oracle["total_cost"] > 0 else None
                    ),
                    "learned_vs_zero_cost_ratio": (
                        learned["total_cost"] / zero["total_cost"]
                        if zero["total_cost"] > 0 else None
                    ),
                    "euler_vs_zoh_cost_ratio": (
                        learned_euler["total_cost"] / learned["total_cost"]
                        if learned["total_cost"] > 0 else None
                    ),
                    "attribution": (
                        "comparable: both controllers solved all steps"
                        if comparable
                        else "NOT_ATTRIBUTABLE: solver success differs or failed"
                    ),
                },
            }
    report["experiments"]["E08"] = {
        "experiment": "E08",
        "status": "verified",
        "a_values": budget["a_values"],
        "seeds": budget["seeds"],
        "solver": "OSQP through cvxpy",
        "fallback_and_residual_logging": True,
        "discretizations": ["zoh", "euler"],
    }

    # ------------------------------------------------------------------
    # E07 closed-loop: fixed initial-state sweep, same plant/horizon/bounds
    # ------------------------------------------------------------------
    for sys_name in ("unicycle", "planar_quadrotor"):
        sys_dict = load_benchmark(sys_name)
        sys_inst = sys_dict["instance"]
        eval_data = eval_data_cache[sys_name]
        x0s = [np.asarray(t["x0"].reshape(-1)) for t in eval_data[:2]]
        target = np.zeros(sys_inst.Nstates)
        for seed in budget["seeds"]:
            module = trained_models[(sys_name, seed)]
            A, B = module.model.operator.get_AB()
            C = np.eye(sys_inst.Nstates, A.shape[0])
            A_lin, B_lin = _linearization(sys_inst, target)
            Q, R = _cost_weights(sys_inst)
            rows = []
            for x0 in x0s:
                learned = _mpc_eval(
                    sys_inst, A.detach().numpy(), B.detach().numpy(), C,
                    module.model.encoder, x0, target, budget["mpc_steps"],
                    budget["mpc_horizon"], seed, discretization="zoh",
                )
                oracle = _mpc_eval(
                    sys_inst, A_lin, B_lin, np.eye(sys_inst.Nstates), None,
                    x0, target, budget["mpc_steps"], budget["mpc_horizon"],
                    seed, discretization="zoh",
                )
                zero = _zero_control_cost(
                    sys_inst, x0, target, budget["mpc_steps"], Q, R,
                )
                rows.append({
                    "x0": x0.tolist(),
                    "learned": learned,
                    "oracle_linearization": oracle,
                    "zero_control": zero,
                    "comparison": {
                        "plant": "RK4 at sys.dt",
                        "oracle_type": "target linearization of the nonlinear plant",
                        "same_horizon": True,
                        "same_bounds": True,
                        "same_cost_weights": True,
                    },
                })
            report["mpc_evaluations"][f"{sys_name}_seed{seed}"] = {
                "system": sys_name,
                "initial_states_fixed": [x.tolist() for x in x0s],
                "runs": rows,
            }

    # ------------------------------------------------------------------
    # E09: local identifiability, MPC-state coverage, excitation control
    # ------------------------------------------------------------------
    a_e09 = 1.0 if 1.0 in budget["a_values"] else budget["a_values"][-1]
    sys_inst = load_benchmark("rotation_family", a=a_e09)["instance"]
    train_path = report["rotation_sweep"][f"a_{a_e09}"]["datasets"]["train"][str(budget["seeds"][0])]["path"]
    train_data = torch.load(train_path)
    eval_data = eval_data_cache[a_e09]
    train_x = _state_array(train_data)
    local_x, train_u = _state_action_rows(train_data)
    eval_x, _ = _state_action_rows(eval_data)
    mpc_q = torch.as_tensor(np.asarray(mpc_states, dtype=np.float64), dtype=train_x.dtype)
    global_u = train_u.double().numpy()
    if global_u.shape[1] == 1:
        gsv = np.asarray([float(np.std(global_u))])
        gcond = 1.0 if gsv[0] > 0 else None
    else:
        gsv = np.linalg.svd(np.cov(global_u, rowvar=False), compute_uv=False)
        gcond = float(gsv.max() / gsv.min()) if gsv.min() > 1e-12 else None
    report["experiments"]["E09"] = {
        "experiment": "E09",
        "status": "verified",
        "global_input": {
            "covariance_singular_values": gsv.tolist(),
            "condition": gcond,
            "num_samples": int(train_u.shape[0]),
        },
        "mpc_state_coverage": _local_input_stats(
            local_x, train_u, mpc_q, sys_inst=sys_inst
        ),
        "eval_state_coverage": _local_input_stats(
            local_x, train_u, eval_x, sys_inst=sys_inst
        ),
        "excitation_control": {"status": "not_run_quick" if quick_mode else "pending"},
    }
    if not quick_mode:
        reduced = _reduced_excitation_data(train_data, sys_inst, seed=42)
        red_u = torch.cat([t["u_seq"] for t in reduced], dim=0).double().numpy()
        if red_u.shape[1] == 1:
            red_sv = np.asarray([float(np.std(red_u))])
            red_cond = 1.0 if red_sv[0] > 0 else None
        else:
            red_sv = np.linalg.svd(np.cov(red_u, rowvar=False), compute_uv=False)
            red_cond = float(red_sv.max() / red_sv.min()) if red_sv.min() > 1e-12 else None
        red_path = os.path.join(data_dir, f"reduced_excitation_a{a_e09}_seed42.pt")
        torch.save(reduced, red_path)
        red_hash = sha256_file(red_path)
        register("dataset", "E09", red_path, red_hash)
        red_ckpt = os.path.join(ckpt_dir, f"reduced_excitation_a{a_e09}_seed42.pt")
        red_module, red_prov = train_official_baseline(
            reduced, sys_inst, latent_dim=LATENT_DIMS["rotation_family"],
            epochs=budget["epochs"], seed=budget["seeds"][0],
            window_len=P2_BASELINE_CONFIG["window_len"],
            max_windows_per_traj=budget["max_windows_per_traj"],
            batch_size=budget["batch_size"], checkpoint_path=red_ckpt,
            dataset_sha256=red_hash,
        )
        register("checkpoint", "E09", red_ckpt, red_prov["checkpoint"]["sha256"])
        probe = build_response_probe(eval_data, sys_inst, 64, EVAL_SEED)
        red_resp = compute_action_response_error(
            red_module, sys_inst, probe["x"], probe["u"], probe["u_ref"]
        )
        report["experiments"]["E09"]["excitation_control"] = {
            "status": "verified",
            "reduced_input_singular_values": red_sv.tolist(),
            "reduced_input_condition": red_cond,
            "reduced_E_response": red_resp["raw"],
            "nominal_E_response": report["rotation_sweep"][f"a_{a_e09}"]["seeds"][str(budget["seeds"][0])]["response"]["raw"],
            "same_validation_probe": True,
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
            if "learned" in ev
        },
        "local_identifiability": report["experiments"]["E09"]["mpc_state_coverage"],
    }

    report["gates"] = _derive_gates(report)
    report["gates"]["p2_verification"] = _derive_verification(report, budget, quick_mode)
    report["summary"] = {
        "wall_time_s": float(time.time() - started),
        "verdicts": {k: v["verdict"] for k, v in report["gates"].items()},
        "interpretation": (
            "P2 verdicts are derived from measured metrics and the pre-registered "
            "criteria in acceptance_criteria. H2 and H3 remain independent tracks. "
            "p2_verification is the full-evidence gate; quick runs are smoke only."
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

    manifest = {
        "run_id": run_id,
        "generated_at": report["generated_at"],
        "quick_mode": bool(quick_mode),
        "report": os.path.abspath(report_path),
        "artifacts_root": os.path.abspath(run_dir),
        "command": source["command"],
        "git": git,
        "environment": env,
        "source_snapshot": source,
        "budget": budget,
        "experiments": {
            key: {
                "status": value.get("status"),
                "missing_systems": value.get("missing_systems"),
            }
            for key, value in report["experiments"].items()
        },
        "artifacts": report["artifacts"],
        "gates": report["gates"],
    }
    with open(os.path.join(run_dir, "manifest.yaml"), "w") as f:
        yaml.safe_dump(
            __import__("json").loads(__import__("json").dumps(manifest, default=str)),
            f, sort_keys=False,
        )
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
