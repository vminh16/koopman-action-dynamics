"""Audit an existing checkpoint against a config: class, keys, shapes, data hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from research.baseline.common import (
    ROOT,
    build_module,
    build_system_config,
    compare_effective,
    git_provenance,
    load_config,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Defaults to the config's checkpoint field")
    parser.add_argument("--hparams", help="Lightning hparams.yaml recorded during training")
    parser.add_argument("--model-class", help="Override when a legacy config omits it")
    parser.add_argument("--train-sha256", help="Override when a legacy config omits it")
    parser.add_argument("--test-sha256", help="Override when a legacy config omits it")
    parser.add_argument("--output", help="JSON report path")
    return parser.parse_args()


def read_lightning_hparams(path):
    """Read only the plain-scalar model_config/lr blocks; skip system_config tags."""
    text = Path(path).read_text()
    model_block = text[text.index("model_config:"):text.index("system_config:")]
    lr_line = next(line for line in text.splitlines() if line.startswith("lr:"))
    model_config = yaml.safe_load(model_block)["model_config"]
    return model_config, yaml.safe_load(lr_line)["lr"]


def audit_training_hparams(cfg, path):
    from training.train import KoopmanLitModule

    model_config, lr = read_lightning_hparams(path)
    recorded = KoopmanLitModule(build_system_config(cfg), model_config, lr=lr)
    effective, mismatches = compare_effective(recorded, cfg)
    return {
        "hparams": str(path),
        "recorded_model_config": model_config,
        "recorded_lr": lr,
        "effective": effective,
        "config_vs_training_mismatches": mismatches,
        "config_vs_training_match": not mismatches,
    }


def audit_dataset(cfg, args):
    out = {}
    ok = True
    any_expected = False
    for name in ("train", "test"):
        path = cfg.get(f"data_{name}", f"data/{cfg['system']}_{name}.pt")
        actual = sha256_file(ROOT / path)
        expected = getattr(args, f"{name}_sha256") or cfg.get(f"data_{name}_sha256")
        out[name] = {
            "path": path,
            "sha256": actual,
            "expected_sha256": expected,
            "matches": expected is None or actual == expected,
        }
        if expected is not None:
            any_expected = True
            ok = ok and actual == expected
    return out, (ok if any_expected else None)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint or cfg["checkpoint"])
    state_dict = torch.load(checkpoint, map_location="cpu")
    model = build_module(cfg)
    model_state = model.state_dict()

    keys = list(state_dict)
    looks_koopman = any(k.startswith("model.operator.") for k in keys)
    looks_mlp = any(k.startswith("model.net.") for k in keys)
    if looks_koopman and looks_mlp:
        detected = "mixed"
    elif looks_koopman:
        detected = "KoopmanLitModule"
    elif looks_mlp:
        detected = "MLPLitModule"
    else:
        detected = "unknown"

    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    shape_mismatch = {
        key: {"checkpoint": list(state_dict[key].shape), "model": list(model_state[key].shape)}
        for key in sorted(set(state_dict) & set(model_state))
        if state_dict[key].shape != model_state[key].shape
    }
    try:
        model.load_state_dict(state_dict, strict=True)
        load_ok = True
        load_error = None
    except Exception as exc:
        load_ok = False
        load_error = repr(exc)

    try:
        dataset, dataset_ok = audit_dataset(cfg, args)
    except Exception as exc:
        dataset, dataset_ok = {"error": repr(exc)}, False

    try:
        effective, mismatches = compare_effective(model, cfg)
        config_effective_ok = not mismatches
    except Exception as exc:
        effective, mismatches = {"error": repr(exc)}, {}
        config_effective_ok = False

    training_hparams_audit = (
        audit_training_hparams(cfg, args.hparams) if args.hparams else None
    )

    config_model_class = cfg.get("model_class") or args.model_class
    class_ok = config_model_class is None or detected == config_model_class
    architecture_ok = (
        class_ok and not missing and not unexpected
        and not shape_mismatch and load_ok
    )
    report = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "detected_model_class": detected,
        "config_model_class": config_model_class,
        "architecture_ok": architecture_ok,
        "strict_load_ok": load_ok,
        "strict_load_error": load_error,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "dataset_hashes_ok": dataset_ok,
        "dataset": dataset,
        "config_effective_match": config_effective_ok,
        "effective_mismatches": mismatches,
        "effective": effective,
        "training_hparams_audit": training_hparams_audit,
        "provenance": git_provenance(),
        "verdict": "PASS" if architecture_ok and dataset_ok is not False else "FAIL",
    }
    output = Path(
        args.output
        or ROOT / "outputs" / "reproduction" / "checkpoint_audit" / f"{checkpoint.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in (
        "checkpoint", "detected_model_class", "config_model_class",
        "architecture_ok", "dataset_hashes_ok", "config_effective_match", "verdict",
    )}, indent=2))
    if training_hparams_audit:
        print(json.dumps({
            "config_vs_training_match": training_hparams_audit["config_vs_training_match"],
            "config_vs_training_mismatches": training_hparams_audit["config_vs_training_mismatches"],
        }, indent=2))
    print(f"report: {output}")
    if not config_effective_ok:
        print("WARNING: config does not match the module's effective hyperparameters "
              "(see effective.error)")
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
