import torch
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import DataLoader
import numpy as np

from my_utils.dataset import load_dataset
from dynamics import SYSTEM_REGISTRY


def reencode_rollout(model, x0, u_seq, step=50):
    """
    Perform segmented rollout with re-encoding every `step` steps.
    Each segment starts from previous predicted x and calls model.forward().

    Args:
        model: KoopmanModel or KoopmanLitModule
        x0: Tensor [1, dx]
        u_seq: Tensor [1, T, du]
        step: int

    Returns:
        x_preds: [1, T, dx]
    """
    B, T, _ = u_seq.shape
    dx = x0.shape[-1]
    x_preds = []

    curr_x = x0.clone()

    for start in range(0, T, step):
        end = min(start + step, T)
        u_chunk = u_seq[:, start:end, :]  # [1, chunk, du]

        z_chunk = model(curr_x, u_chunk)  # [1, chunk, dz]
        x_chunk = z_chunk[:, :, :dx]      # [1, chunk, dx]

        x_preds.append(x_chunk)
        curr_x = x_chunk[:, -1, :].detach()

    return torch.cat(x_preds, dim=1)  # [1, T, dx]


@torch.no_grad()
def evaluate_model(model, config, window_len=200, max_traj=100, plot_examples=3, save_dir="figures", seed=42, use_reencode=False, reencode_step=50):
    system_name = config["system"]
    print(f"🔍 Evaluating on test split [{system_name}] with window_len={window_len}, reencode={use_reencode}")

    system = SYSTEM_REGISTRY[system_name]()
    dt = system.dt
    state_dim = system.sample_initial_state(1).shape[-1]

    koopman_mode = config.get("koopman_mode", "unknown")
    latent_dim = config.get("latent_dim", "unknown")
    use_modulated_u = config.get("use_modulated_u", False)
    train_window_len = config.get("window_len", "unknown")

    data = load_dataset(system_name, split="test")
    # print(data)
    random.seed(seed)
    data = random.sample(data, k=min(max_traj, len(data)))

    model.eval()
    mse_list = []

    for i, traj in enumerate(data):
        if traj["u_seq"].shape[0] < window_len + 1:
            continue
        x0 = traj["x0"].unsqueeze(0)
        # x0 = traj["x_seq"][0]
        x_seq = traj["x_seq"][:window_len]   # [T, D]
        u_seq = traj["u_seq"][:+window_len]       # [T, u_dim]
        # x0 = x_seq[0].squeeze(0).unsqueeze(0)     # [1, D]
        u_seq = u_seq.unsqueeze(0)                # [1, T, u_dim]
        x_true = x_seq.squeeze(1).unsqueeze(0) # [1, T, D]

        if use_reencode:
            x_pred = reencode_rollout(model, x0, u_seq, step=reencode_step)  # [1, T, D]
        else:
            z_seq = model(x0, u_seq)                              # [1, T, dz]
            x_pred = z_seq[:, :, :state_dim]                     # [1, T, D]

        # print(x_pred)
        # print(x_seq)
        mse = torch.mean((x_pred - x_true) ** 2).item()
        mse_list.append(mse)

        # import pdb;
        # pdb.set_trace()
        if i < plot_examples:
            t = torch.arange(window_len) * dt
            np.save(os.path.join(save_dir, f"{system_name}_traj{i}_true.npy"), x_true[0].cpu().numpy())
            np.save(os.path.join(save_dir, f"{system_name}_traj{i}_pred.npy"), x_pred[0].cpu().numpy())
            for d in range(state_dim):
                plt.figure(figsize=(6, 3))
                plt.plot(t, x_true[0, :, d].cpu(), label=f"True dim {d}")
                plt.plot(t, x_pred[0, :, d].cpu(), '--', label=f"Pred dim {d}")
                plt.xlabel("Time [s]")
                plt.ylabel(f"State {d}")
                # if d == 0:
                #     plt.ylim(-3.14, 3.14)
                plt.title(f"Traj {i} - dim {d} ({'Reenc' if use_reencode else 'Normal'})")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                # plt.show()

                os.makedirs(save_dir, exist_ok=True)
                fname = f"{save_dir}/{system_name}_mode-{koopman_mode}_dz{latent_dim}" \
                        f"{'_uenc' if use_modulated_u else ''}_trainwin{train_window_len}_evalwin{window_len}" \
                        f"_{'reenc' if use_reencode else 'normal'}_step{reencode_step}_traj{i}_dim{d}.png"
                plt.savefig(fname)
                plt.close()

    avg_mse = sum(mse_list) / len(mse_list)
    print(f"\n📈 Average MSE on {len(mse_list)} test windows: {avg_mse:.6f}")

import os, random, torch
import matplotlib.pyplot as plt


def evaluate_mlp_simple(model,
                        system_name,
                        window_len=15,
                        max_traj=100,
                        plot_examples=3,
                        save_dir="figures_mlp",
                        seed=42):
    """

    """
    system = SYSTEM_REGISTRY[system_name]()
    dt = system.dt
    data = load_dataset(system_name, split="test")

    random.seed(seed)
    data = random.sample(data, k=min(max_traj, len(data)))

    model.eval()
    device = next(model.parameters()).device


    fwd_model = model.model if hasattr(model, "model") else model

    mse_list = []

    with torch.no_grad():
        for i, traj in enumerate(data):
            if traj["u_seq"].shape[0] < window_len or traj["x_seq"].shape[0] < window_len:
                continue

            # x0: [1, D] 或 [1,1,D]
            x0 = traj["x0"].unsqueeze(0).to(device)
            if x0.ndim == 3 and x0.shape[1] == 1:
                x0 = x0.squeeze(1)                  # [1, D]

            # u_seq: [1, T, U] 或 [1,1,T,U]
            u_seq = traj["u_seq"][:window_len].unsqueeze(0).to(device)
            if u_seq.ndim == 4 and u_seq.shape[1] == 1:
                u_seq = u_seq.squeeze(1)            # [1, T, U]

            # x_true: 目标真值 [1, T, D]（兼容 [T,1,D]）
            x_true = traj["x_seq"][:window_len].to(device)   # [T,D] 或 [T,1,D]
            if x_true.ndim == 3 and x_true.shape[1] == 1:
                x_true = x_true.squeeze(1)          # [T, D]
            x_true = x_true.unsqueeze(0)            # [1, T, D]

            # 预测（模型输出 [B,T,*]；若多出维度，裁到 D）
            out = fwd_model(x0, u_seq)              # [1, T, *]
            x_pred = out[..., :x_true.shape[-1]]    # [1, T, D]

            mse = torch.mean((x_pred - x_true) ** 2).item()
            mse_list.append(mse)

            # 可选画图
            if plot_examples and save_dir and i < plot_examples:
                os.makedirs(save_dir, exist_ok=True)
                t = (torch.arange(window_len, device=device) * dt).cpu().numpy()
                x_true_np = x_true[0].detach().cpu().numpy()  # [T,D]
                x_pred_np = x_pred[0].detach().cpu().numpy()  # [T,D]

                np.save(os.path.join(save_dir, f"{system_name}_traj{i}_true.npy"), x_true_np)
                np.save(os.path.join(save_dir, f"{system_name}_traj{i}_pred.npy"), x_pred_np)


                D = x_true_np.shape[-1]
                for d in range(D):
                    plt.figure(figsize=(6, 3))
                    plt.plot(t, x_true_np[:, d], label=f"True dim {d}")
                    plt.plot(t, x_pred_np[:, d], '--', label=f"Pred dim {d}")
                    plt.xlabel("Time [s]"); plt.ylabel(f"State {d}")
                    plt.title(f"{system_name} | traj {i} | dim {d}")
                    plt.legend(); plt.grid(); plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"{system_name}_traj{i}_dim{d}.png"))
                    plt.close()

    avg_mse = sum(mse_list) / max(1, len(mse_list))
    print(f"📈 Average MSE over {len(mse_list)} trajectories (T={window_len}): {avg_mse:.6e}")
    return avg_mse, mse_list