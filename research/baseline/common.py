"""Shared, minimal config/provenance contract for the research baseline runner.

One YAML file is the single source of truth. The same dict is passed to
KoopmanLitModule, so the file cannot drift from the constructed module.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

MODEL_KEYS = (
    "koopman_mode",
    "control_loss_flag",
    "encoder_loss_flag",
    "use_modulated_u",
    "latent_dim",
    "num_layers",
    "hidden_size",
    "u_encoder_layers",
    "u_encoder_hidden",
    "lambda_encoder_pred",
    "lambda_control",
)
SAMPLER_VERSION = "sampled-window-full-range-v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def git_provenance() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    porcelain = run("status", "--porcelain", "--untracked-files=normal")
    lines = [line for line in porcelain.splitlines() if line.strip()]
    tracked = [line for line in lines if not line.startswith("??")]
    return {
        "git_head": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "working_tree_clean": not lines,
        "source_clean": not tracked,
        "tracked_changes": tracked,
        "untracked_files": [line[3:] for line in lines if line.startswith("??")],
    }


def build_system_config(cfg: dict) -> dict:
    from dynamics import SYSTEM_REGISTRY

    system = SYSTEM_REGISTRY[cfg["system"]]()
    return {
        "dynamics": system.dynamics,
        "sample_initial_state": system.sample_initial_state,
        "umin": system.umin,
        "umax": system.umax,
        "dt": system.dt,
        "state_dim": system.Nstates,
    }


def build_module(cfg: dict):
    """Construct the module from the exact same dict that is saved on disk."""
    from training.train import KoopmanLitModule

    return KoopmanLitModule(build_system_config(cfg), cfg, lr=cfg["lr"])


def _optimizer_and_scheduler(module):
    out = module.configure_optimizers()
    return (
        out["optimizer"],
        out["lr_scheduler"]["scheduler"],
        out["lr_scheduler"].get("monitor"),
    )


def effective_hyperparameters(module, cfg: dict) -> dict:
    from training.loss import prediction_loss

    optimizer, scheduler, monitor = _optimizer_and_scheduler(module)
    state = module.state_dict()
    encoder_linears = [m for m in module.model.encoder.extra_net if isinstance(m, torch.nn.Linear)]
    model_config = {
        "koopman_mode": module.koopman_mode,
        "lambda_encoder_pred": module.lambda_encoder_pred,
        "lambda_control": module.lambda_control,
        "encoder_loss_flag": module.encoder_loss_flag,
        "control_loss_flag": module.control_loss_flag,
        "use_modulated_u": module.model.use_modulated_u,
        "latent_dim": module.model.latent_dim,
        "num_layers": len(encoder_linears) - 1,
        "hidden_size": encoder_linears[0].out_features,
        "u_encoder_layers": cfg.get("u_encoder_layers"),
        "u_encoder_hidden": cfg.get("u_encoder_hidden"),
    }
    if module.model.use_modulated_u:
        u_linears = [m for m in module.model.u_encoder.bilinear_net if isinstance(m, torch.nn.Linear)]
        model_config["u_encoder_layers"] = len(u_linears) - 1
        model_config["u_encoder_hidden"] = u_linears[0].out_features
    return {
        "model_class": type(module).__name__,
        "model_config": model_config,
        "state_dim": module.state_dim,
        "latent_dim": module.model.latent_dim,
        "dt": module.model.dt,
        "optimizer": type(optimizer).__name__,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "scheduler": type(scheduler).__name__,
        "scheduler_factor": getattr(scheduler, "factor", None),
        "scheduler_patience": getattr(scheduler, "patience", None),
        "scheduler_monitor": monitor,
        "prediction_gamma": prediction_loss.__defaults__[1],
        # controllability_loss overwrites its T argument with 15*dt.
        "gramian_horizon_steps": 15,
        "gramian_horizon_seconds": 15 * module.dt,
        "state_dict": {k: list(v.shape) for k, v in state.items()},
        "parameter_count": sum(p.numel() for p in module.parameters()),
    }


REQUIRED_CONFIG_KEYS = (
    "model_class", "koopman_mode", "control_loss_flag", "encoder_loss_flag",
    "use_modulated_u", "latent_dim", "num_layers", "hidden_size",
    "u_encoder_layers", "u_encoder_hidden", "lr", "weight_decay", "optimizer",
    "scheduler", "scheduler_factor", "scheduler_patience", "prediction_gamma",
    "gramian_horizon_steps", "lambda_encoder_pred", "lambda_control",
)


def _actual_flat(eff: dict) -> dict:
    return {
        "model_class": eff["model_class"],
        **{key: eff["model_config"][key] for key in (
            "koopman_mode", "lambda_control", "lambda_encoder_pred",
            "encoder_loss_flag", "control_loss_flag", "use_modulated_u",
            "latent_dim", "num_layers", "hidden_size",
            "u_encoder_layers", "u_encoder_hidden",
        )},
        "state_dim": eff["state_dim"],
        "lr": eff["lr"],
        "weight_decay": eff["weight_decay"],
        "optimizer": eff["optimizer"],
        "scheduler": eff["scheduler"],
        "scheduler_factor": eff["scheduler_factor"],
        "scheduler_patience": eff["scheduler_patience"],
        "prediction_gamma": eff["prediction_gamma"],
        "gramian_horizon_steps": eff["gramian_horizon_steps"],
    }


def compare_effective(module, cfg: dict) -> tuple[dict, dict]:
    """Compare only keys present in cfg, so legacy configs can be audited."""
    eff = effective_hyperparameters(module, cfg)
    actual = _actual_flat(eff)
    expected = {
        "model_class": cfg.get("model_class"),
        "koopman_mode": cfg.get("koopman_mode"),
        "control_loss_flag": cfg.get("control_loss_flag"),
        "encoder_loss_flag": cfg.get("encoder_loss_flag"),
        "use_modulated_u": cfg.get("use_modulated_u"),
        "latent_dim": cfg.get("latent_dim"),
        "num_layers": cfg.get("num_layers"),
        "hidden_size": cfg.get("hidden_size"),
        "u_encoder_layers": cfg.get("u_encoder_layers"),
        "u_encoder_hidden": cfg.get("u_encoder_hidden"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "optimizer": cfg.get("optimizer"),
        "scheduler": cfg.get("scheduler"),
        "scheduler_factor": cfg.get("scheduler_factor"),
        "scheduler_patience": cfg.get("scheduler_patience"),
        "prediction_gamma": cfg.get("prediction_gamma"),
        "gramian_horizon_steps": cfg.get("gramian_horizon_steps"),
        "lambda_encoder_pred": cfg.get("lambda_encoder_pred"),
        "lambda_control": cfg.get("lambda_control"),
        "state_dim": build_system_config(cfg)["state_dim"],
    }
    mismatches = {
        key: {"config": value, "effective": actual[key]}
        for key, value in expected.items()
        if value is not None and value != actual[key]
    }
    return eff, mismatches


def check_effective_hyperparameters(module, cfg: dict) -> dict:
    """Fail loudly if the YAML claims anything the module does not use."""
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    eff, mismatches = compare_effective(module, cfg)
    if mismatches:
        raise ValueError(f"config/effective hyperparameter mismatch: {mismatches}")
    return eff



def dataset_fingerprint(cfg: dict) -> dict:
    out = {}
    for name in ("train", "test"):
        path = cfg[f"data_{name}"]
        actual = sha256_file(ROOT / path)
        expected = cfg[f"data_{name}_sha256"]
        if actual != expected:
            raise ValueError(
                f"{name} dataset hash mismatch: config={expected} actual={actual} ({path})"
            )
        out[name] = {"path": path, "sha256": actual}
    return out


def select_training_trajectories(data, cfg: dict):
    fraction = cfg["train_fraction"]
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    n = max(1, int(len(data) * fraction))
    if n != cfg["num_train_trajectories"]:
        raise ValueError(
            f"num_train_trajectories={cfg['num_train_trajectories']} but "
            f"fraction={fraction} of {len(data)} gives {n}"
        )
    g = torch.Generator().manual_seed(cfg["seed"])
    idx = torch.randperm(len(data), generator=g)[:n]
    return [data[i] for i in idx.tolist()], idx.tolist()


def check_sampler(dataset, cfg: dict, n_trajectories: int) -> dict:
    expected = n_trajectories * min(
        cfg["max_windows_per_traj"], cfg["window_start_max"] + 1
    )
    if len(dataset) != expected:
        raise ValueError(f"sampler produced {len(dataset)} windows, expected {expected}")
    starts = [start for _, start in dataset.samples]
    if min(starts) < 0 or max(starts) > cfg["window_start_max"]:
        raise ValueError("sampler start outside [0, window_start_max]")
    return {
        "version": SAMPLER_VERSION,
        "source": "my_utils/dataset.py:SampledMultiTrajectoryDataset",
        "source_sha256": sha256_file(ROOT / "my_utils/dataset.py"),
        "window_len": cfg["window_len"],
        "max_windows_per_traj": cfg["max_windows_per_traj"],
        "window_seed": cfg["seed"],
        "window_start_max": cfg["window_start_max"],
        "windows_total": len(dataset),
        "n_trajectories": n_trajectories,
    }


def environment_provenance() -> dict:
    import pytorch_lightning as pl

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(manifest, default=str)), f, sort_keys=False)
