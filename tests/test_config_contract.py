"""Assert-based contract check: configs/research/*.yaml match what trains."""
from pathlib import Path

from research.baseline.common import (
    ROOT,
    build_module,
    check_effective_hyperparameters,
    dataset_fingerprint,
    load_config,
    select_training_trajectories,
)

CONFIGS = sorted((ROOT / "configs" / "research").glob("*.yaml"))


def main():
    assert CONFIGS, "no research configs found"
    for path in CONFIGS:
        cfg = load_config(path)
        model = build_module(cfg)
        check_effective_hyperparameters(model, cfg)
        dataset_fingerprint(cfg)

        from my_utils.dataset import load_dataset

        train = load_dataset(cfg["system"], split="train")
        select_training_trajectories(train, cfg)
        print(f"PASS {path.name}")

    # Guard: reproduce the original bug (model gets a subset dict) and require a failure.
    from training.train import KoopmanLitModule
    from research.baseline.common import build_system_config

    cfg = load_config(CONFIGS[0])
    cfg["lambda_control"] = 0.01  # value written in the original config files
    legacy_model_config = {
        key: cfg[key] for key in (
            "koopman_mode", "control_loss_flag", "use_modulated_u", "latent_dim",
            "num_layers", "hidden_size", "u_encoder_layers", "u_encoder_hidden",
        )
    }
    subset_model = KoopmanLitModule(build_system_config(cfg), legacy_model_config, lr=cfg["lr"])
    try:
        check_effective_hyperparameters(subset_model, cfg)
    except ValueError as exc:
        assert "lambda_control" in str(exc), exc
        print("PASS drift guard rejects the legacy subset-config bug")
    else:
        raise AssertionError("drift guard did not reject the legacy subset-config bug")


if __name__ == "__main__":
    main()
