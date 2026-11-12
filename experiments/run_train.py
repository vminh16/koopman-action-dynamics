import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from dynamics import SYSTEM_REGISTRY
from my_utils.dataset import load_dataset
from my_utils.dataset import SampledMultiTrajectoryDataset as KoopmanTrajectoryDataset
from training.train import KoopmanLitModule, NeuralODELitModule, MLPLitModule

import argparse
import yaml, os

set_seed = 42

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="pendulum")
    parser.add_argument("--koopman_mode", type=str, default="free", choices=["free", "controllable"])
    parser.add_argument("--control_loss_flag", action="store_true", help="use controllability loss")
    parser.add_argument("--use_modulated_u", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--u_encoder_layers", type=int, default=2)
    parser.add_argument("--u_encoder_hidden", type=int, default=64)
    parser.add_argument("--window_len", type=int, default=50)
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.05,
        help="Fraction of training trajectories to use (paper settings: 0.01, 0.05, 0.10, 0.30)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    pl.seed_everything(set_seed, workers=True)
    # System dynamics setup
    system = SYSTEM_REGISTRY[args.system]()
    system_config = {
        "dynamics": system.dynamics,
        "sample_initial_state": system.sample_initial_state,
        "umin": system.umin,
        "umax": system.umax,
        "dt": system.dt,
        "state_dim": system.Nstates,
    }

    model_config = {
        "koopman_mode": args.koopman_mode,
        "control_loss_flag": args.control_loss_flag,
        "use_modulated_u": args.use_modulated_u,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "u_encoder_layers": args.u_encoder_layers,
        "u_encoder_hidden": args.u_encoder_hidden,
    }

    # load data
    data = load_dataset(args.system, split="train")
    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train_fraction must be in the interval (0, 1].")
    num_train_trajectories = max(1, int(len(data) * args.train_fraction))
    g = torch.Generator().manual_seed(set_seed)
    trajectory_idx = torch.randperm(len(data), generator=g)[:num_train_trajectories]
    data = [data[i] for i in trajectory_idx.tolist()]
    dataset = KoopmanTrajectoryDataset(
        data,
        window_len=args.window_len,
        max_windows_per_traj=20,
        seed=set_seed,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # construct model
    # model = NeuralODELitModule(system_config, model_config, lr=args.lr)
    # model = MLPLitModule(system_config, model_config)
    model = KoopmanLitModule(system_config, model_config, lr=args.lr)

    # log
    tag = f"{args.system}_mode-{args.koopman_mode}"
    if args.use_modulated_u:
        tag += "_uenc"
    tag += f"_win{args.window_len}"
    tag += f"_frac{args.train_fraction:g}"
    tag += f"_dz{args.latent_dim}_hs{args.hidden_size}_nl{args.num_layers}"
    tag += f"_ul{args.u_encoder_layers}_uh{args.u_encoder_hidden}"

    logger = TensorBoardLogger("lightning_logs", name=tag)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(model, dataloader)

    # save
    ckpt_path = f"checkpoints/{tag}_MLPmodel.pt"
    torch.save(model.state_dict(), ckpt_path)


    config = {
        "system": args.system,
        "koopman_mode": args.koopman_mode,
        "control_loss_flag": args.control_loss_flag,
        "use_modulated_u": args.use_modulated_u,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "u_encoder_layers": args.u_encoder_layers,
        "u_encoder_hidden": args.u_encoder_hidden,
        "window_len": args.window_len,
        "train_fraction": args.train_fraction,
        "num_train_trajectories": num_train_trajectories,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_encoder_pred": 0.5,
        "lambda_control": 0.01,
        "encoder_loss_flag": True,
        "checkpoint": ckpt_path
    }

    os.makedirs("configs", exist_ok=True)
    yaml_path = f"configs/{tag}_MLPmodel.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    print(f"✅ Saved config to {yaml_path}")