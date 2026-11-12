# experiments/run_eval.py

import argparse
import yaml
import torch
import os

from training.train import KoopmanLitModule, NeuralODELitModule, MLPLitModule
from my_utils.dataset import load_dataset
from training.evaluate import evaluate_model, evaluate_mlp_simple
from dynamics import SYSTEM_REGISTRY


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Init system
    system_class = SYSTEM_REGISTRY[config["system"]]
    system = system_class()
    system_config = {
        "dynamics": system.dynamics,
        "sample_initial_state": system.sample_initial_state,
        "umin": system.umin,
        "umax": system.umax,
        "dt": system.dt,
        "state_dim": system.Nstates
    }

    # Build model
    model = KoopmanLitModule(system_config, config)
    # model = NeuralODELitModule(system_config, config)
    # model = MLPLitModule(system_config, config)
    ckpt = torch.load(config["checkpoint"])
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded model from {config['checkpoint']}")

    # avg_mse, per_traj = evaluate_mlp_simple(
    #     model=model,  #
    #     system_name="pendulum",
    #     window_len=15,
    #     max_traj=100,
    #     plot_examples=3,
    #     save_dir="figures_mlp",
    #     seed=42
    # )


    # Evaluate
    evaluate_model(
        model=model,
        config = config,
        window_len=args.window_len,
        max_traj=args.max_traj,
        plot_examples=args.num_plot,
        save_dir="figures",
        use_reencode = args.use_reencode,
        # default fallback inside the script
        reencode_step=args.reencode_step if args.reencode_step is not None else 50
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--window_len", type=int, default=250, help="Length of window")
    parser.add_argument("--max_traj", type=int, default=100, help="How many trajectories in total for evaluation")
    parser.add_argument("--num_plot", type=int, default=5, help="How many trajectories to visualize")
    parser.add_argument("--use_reencode", action="store_true")
    parser.add_argument("--reencode_step", type=int, help="Number of steps between re-encodes (only if --use_reencode)")
    args = parser.parse_args()

    main(args)