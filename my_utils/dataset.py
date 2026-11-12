import os
import torch
from tqdm import tqdm
import random
from my_utils.simulate import simulate_trajectory

def collect_dataset(system, num_trajectories, T, dt=None, device='cpu'):
    dataset = []
    dyn_fn = system["dynamics"]
    sample_fn = system["sample_initial_state"]
    umin, umax = system["umin"].to(device), system["umax"].to(device)
    dt = dt if dt is not None else system["dt"]

    for _ in tqdm(range(num_trajectories), desc="Sampling trajectories"):
        x0 = sample_fn().to(device)
        u_list = [
            torch.rand_like(umin) * (umax - umin) + umin
            for _ in range(T)
        ]
        # X, U, _ = simulate_trajectory(dyn_fn, x0, u_list, dt=dt, device=device)
        traj = simulate_trajectory(dyn_fn, x0, u_list, dt=dt, device=device)
        # dataset.append((X, U))
        dataset.append(traj)
    return dataset


def save_dataset(dataset, system_name, split="train"):
    os.makedirs("data", exist_ok=True)
    path = f"data/{system_name.lower()}_{split}.pt"
    torch.save(dataset, path)
    print(f"✅ Saved {split} dataset to {path}")


def load_dataset(system_name, split="train"):
    path = f"data/{system_name.lower()}_{split}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return torch.load(path)


class SampledMultiTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_len, max_windows_per_traj=20, seed=42):
        self.samples = []
        self.window_len = window_len

        for traj_idx, traj in enumerate(data):
            T = traj["u_seq"].shape[0]
            max_start = T - window_len + 1
            if max_start <= 0:
                continue

            rng = random.Random(seed + traj_idx)
            starts = rng.sample(
                range(max_start), k=min(max_windows_per_traj, max_start)
            )
            for start in starts:
                self.samples.append((traj_idx, start))

        self.data = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, start = self.samples[idx]
        traj = self.data[traj_idx]

        if start == 0:
            x0 = traj["x0"].squeeze(0)
        else:
            x0 = traj["x_seq"][start - 1].squeeze(0)
        u_seq = traj["u_seq"][start : start + self.window_len]
        x_seq = traj["x_seq"][start : start + self.window_len].squeeze(1)

        return x0, u_seq, x_seq


if __name__ == "__main__":
    import argparse
    from dynamics import SYSTEM_REGISTRY

    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="pendulum", choices=SYSTEM_REGISTRY.keys())
    parser.add_argument("--num", type=int, default=1000, help="Number of trajectories")
    parser.add_argument("--T", type=int, default=50, help="Steps per trajectory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    system_class = SYSTEM_REGISTRY[args.system]
    system = system_class()
    system_config = {
        "dynamics": system.dynamics,
        "sample_initial_state": system.sample_initial_state,
        "umin": system.umin,
        "umax": system.umax,
        "dt": system.dt
    }

    print(f"📦 Sampling {args.num} trajectories from '{args.system}'...")
    data = collect_dataset(system_config, num_trajectories=args.num, T=args.T, dt=system.dt)
    save_dataset(data, system_name=args.system, split=args.split)
    print("✅ Done.")