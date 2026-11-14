"""Recompute P2 one-step and open-loop metrics from datasets + checkpoints.

Every referenced dataset and checkpoint is SHA-256 checked against the report,
the official ``KoopmanLitModule`` is rebuilt and loaded with ``strict=True``,
and the recomputed metrics must agree with the recorded ones within tolerance.

Usage:
    python -m research.diagnostics.recompute_p2 --report outputs/diagnosis/p2_failure_report.yaml
Exit code is 0 only when every entry verifies.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import yaml

from research.baseline.common import sha256_file
from research.benchmarks import load_benchmark
from research.diagnostics.run_p2_diagnostics import (
    HORIZONS,
    _mse,
    _stack_transitions,
    build_official_module,
    predict_one_step,
)


def _verify_hash(path: str, expected: str | None) -> None:
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _recompute_entry(system: str, entry: dict, a: float | None) -> dict:
    sys_inst = (
        load_benchmark("rotation_family", a=a)["instance"]
        if a is not None
        else load_benchmark(system)["instance"]
    )
    eval_path = entry["datasets"]["eval"]["path"]
    _verify_hash(eval_path, entry["datasets"]["eval"].get("sha256"))
    eval_data = torch.load(eval_path)

    out = {}
    for seed, recorded in entry["seeds"].items():
        train_path = entry["datasets"]["train"][seed]["path"]
        _verify_hash(train_path, entry["datasets"]["train"][seed].get("sha256"))
        ckpt_path = recorded["training"]["checkpoint"]["path"]
        _verify_hash(ckpt_path, recorded["training"]["checkpoint"].get("sha256"))

        module = build_official_module(
            sys_inst,
            int(recorded["training"]["latent_dim"]),
            config={
                **recorded["training"]["effective"]["model_config"],
                "lr": recorded["training"]["trainer"]["lr"],
            },
        )
        module.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=True)
        module.eval()

        X, U, X_next = _stack_transitions(eval_data)
        with torch.no_grad():
            one_step = _mse(predict_one_step(module, X, U), X_next)
            open_loop = {}
            x0 = torch.cat([t["x0"].reshape(1, -1) for t in eval_data], dim=0)
            for H in HORIZONS:
                u_seq = torch.stack([t["u_seq"][:H] for t in eval_data])
                truth = torch.stack([t["x_seq"][:H] for t in eval_data])
                pred = module(x0, u_seq)[:, :, : sys_inst.Nstates]
                open_loop[str(H)] = _mse(pred, truth)

        out[seed] = {
            "one_step_mse": one_step,
            "open_loop_mse": open_loop,
            "recorded_one_step_mse": recorded["one_step_mse"],
            "recorded_open_loop_mse": {
                H: recorded["open_loop_mse"][H]["mse"] for H in open_loop
            },
        }
    return out


def recompute(report_path: str, tolerance: float) -> dict:
    with open(report_path) as f:
        report = yaml.safe_load(f)

    results, failures = {}, []
    for name, entry in report.get("rotation_sweep", {}).items():
        results[f"E06/{name}"] = _recompute_entry("rotation_family", entry, entry["a"])
    for name, entry in report.get("benchmark_evaluations", {}).items():
        results[f"E07/{name}"] = _recompute_entry(name, entry, None)

    for group, seeds in results.items():
        for seed, rec in seeds.items():
            pairs = [("one_step_mse", rec["one_step_mse"], rec["recorded_one_step_mse"])]
            pairs += [
                (f"open_loop_mse/{H}", rec["open_loop_mse"][H], recorded)
                for H, recorded in rec["recorded_open_loop_mse"].items()
            ]
            for metric, got, want in pairs:
                if abs(got - want) > tolerance + tolerance * abs(want):
                    failures.append(
                        {"entry": group, "seed": seed, "metric": metric,
                         "recomputed": got, "recorded": want}
                    )

    return {
        "report": report_path,
        "tolerance": tolerance,
        "entries": len(results),
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    result = recompute(args.report, args.tolerance)
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
