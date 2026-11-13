"""Train the author's Koopman module from one locked YAML config.

The same dict loaded from disk is passed to KoopmanLitModule and saved in the
manifest. Any config/effective-hyperparameter drift aborts before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from research.baseline.common import (
    ROOT,
    build_module,
    check_effective_hyperparameters,
    check_sampler,
    dataset_fingerprint,
    environment_provenance,
    git_provenance,
    load_config,
    select_training_trajectories,
    sha256_file,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Single locked YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data/model only")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    cfg = load_config(args.config)

    from my_utils.dataset import SampledMultiTrajectoryDataset, load_dataset
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    data_fingerprint = dataset_fingerprint(cfg)
    all_train_data = load_dataset(cfg["system"], split="train")
    train_data, selected_idx = select_training_trajectories(all_train_data, cfg)
    dataset = SampledMultiTrajectoryDataset(
        train_data,
        window_len=cfg["window_len"],
        max_windows_per_traj=cfg["max_windows_per_traj"],
        seed=cfg["seed"],
    )
    sampler = check_sampler(dataset, cfg, len(train_data))

    pl.seed_everything(cfg["seed"], workers=True)
    model = build_module(cfg)
    effective = check_effective_hyperparameters(model, cfg)

    idx_hash = hashlib.sha256(json.dumps(selected_idx).encode()).hexdigest()
    manifest = {
        "experiment_id": cfg["run_id"],
        "status": "running",
        "config_path": args.config,
        "config_sha256": sha256_file(args.config),
        "config": cfg,
        "effective": effective,
        "provenance": {**git_provenance(), **environment_provenance()},
        "dataset": {
            "files": data_fingerprint,
            "train_trajectories_available": len(all_train_data),
            "train_trajectories_used": len(train_data),
            "train_fraction": cfg["train_fraction"],
            "selection_seed": cfg["seed"],
            "selected_indices_sha256": idx_hash,
        },
        "sampler": sampler,
    }
    manifest_path = ROOT / "outputs" / "reproduction" / cfg["run_id"] / "manifest.yaml"
    print(f"effective: {json.dumps(effective['model_config'], default=str)}")

    if args.dry_run:
        print("dry run: config, dataset and model contract are valid (manifest not written)")
        return

    write_manifest(manifest_path, manifest)
    print(f"manifest (pre-train): {manifest_path}")

    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    logger = TensorBoardLogger("outputs/lightning_logs", name=cfg["run_id"])
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=cfg["deterministic"],
    )
    try:
        trainer.fit(model, dataloader)
        checkpoint = Path(cfg["checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        manifest["status"] = "complete"
        manifest["training"] = {
            "epochs_completed": cfg["epochs"],
            "global_step": trainer.global_step,
            "final_metrics": {
                k: float(v) for k, v in trainer.callback_metrics.items()
                if isinstance(v, torch.Tensor) and v.numel() == 1
            },
        }
        manifest["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        write_manifest(manifest_path, manifest)
        raise
    write_manifest(manifest_path, manifest)
    print(f"checkpoint: {cfg['checkpoint']}")
    print(f"manifest (post-train): {manifest_path}")


if __name__ == "__main__":
    main()
