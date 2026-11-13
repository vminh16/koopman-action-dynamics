import hashlib
import os
import torch
import numpy as np
from research.benchmarks import load_benchmark


def rk4_step(dynamics_fn, t: float, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Standard Runge-Kutta 4th order step for continuous dynamics dot(x) = f(t, x, u).
    """
    k1 = dynamics_fn(t, x, u)
    k2 = dynamics_fn(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = dynamics_fn(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = dynamics_fn(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_benchmark_dataset(
    sys_name: str,
    num_trajectories: int,
    T: int,
    dt: float = 0.02,
    seed: int = 42,
    save_dir: str = "data",
    sys_kwargs: dict = None,
) -> tuple[str, str, dict]:
    """
    Generate deterministic simulation trajectories for a benchmark dynamical system.

    Returns:
        (save_path, sha256_hash, metadata_dict)
    """
    sys_kwargs = sys_kwargs or {}
    bench = load_benchmark(sys_name, **sys_kwargs)
    dynamics_fn = bench["dynamics"]
    sample_x0 = bench["sample_initial_state"]
    umin = bench["umin"]
    umax = bench["umax"]
    Nstates = bench["Nstates"]
    Ncontrols = bench["Ncontrols"]

    # Seed generators deterministically
    torch.manual_seed(seed)
    np.random.seed(seed)

    trajectories = []
    all_x = []
    all_u = []

    for _ in range(num_trajectories):
        x = sample_x0(1).squeeze(0)
        x0 = x.clone()

        # Generate uniform random controls in [umin, umax]
        u_rand = torch.rand(T, Ncontrols)
        u_seq = umin + u_rand * (umax - umin)

        x_seq = []
        cur_t = 0.0

        for t_idx in range(T):
            u_t = u_seq[t_idx : t_idx + 1]  # shape [1, Ncontrols]
            x_in = x.unsqueeze(0)           # shape [1, Nstates]
            x_next = rk4_step(dynamics_fn, cur_t, x_in, u_t, dt).squeeze(0)
            x_seq.append(x_next)
            x = x_next
            cur_t += dt

        x_seq_tensor = torch.stack(x_seq)
        trajectories.append({
            "x0": x0,
            "u_seq": u_seq,
            "x_seq": x_seq_tensor,
        })
        all_x.append(x_seq_tensor)
        all_u.append(u_seq)

    all_x_cat = torch.cat(all_x, dim=0)  # [num_trajectories * T, Nstates]
    all_u_cat = torch.cat(all_u, dim=0)  # [num_trajectories * T, Ncontrols]

    metadata = {
        "system": sys_name,
        "sys_kwargs": sys_kwargs,
        "num_trajectories": num_trajectories,
        "T": T,
        "dt": dt,
        "seed": seed,
        "action_mean": all_u_cat.mean(dim=0).tolist(),
        "action_std": all_u_cat.std(dim=0).tolist(),
        "action_min": all_u_cat.min(dim=0).values.tolist(),
        "action_max": all_u_cat.max(dim=0).values.tolist(),
        "state_bounds": {
            "min": all_x_cat.min(dim=0).values.tolist(),
            "max": all_x_cat.max(dim=0).values.tolist(),
            "mean": all_x_cat.mean(dim=0).tolist(),
            "std": all_x_cat.std(dim=0).tolist(),
        },
    }

    os.makedirs(save_dir, exist_ok=True)
    suffix = ""
    if "a" in sys_kwargs:
        suffix = f"_a{sys_kwargs['a']}"
    filename = f"stress_{sys_name}{suffix}_N{num_trajectories}_T{T}_s{seed}.pt"
    save_path = os.path.join(save_dir, filename)

    torch.save(trajectories, save_path)

    # Compute SHA-256 fingerprint
    h = hashlib.sha256()
    with open(save_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha256_hash = h.hexdigest()

    metadata["sha256"] = sha256_hash
    metadata["filepath"] = save_path

    return save_path, sha256_hash, metadata

