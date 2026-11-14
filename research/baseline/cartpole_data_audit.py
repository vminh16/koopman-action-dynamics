"""Fail-closed CartPole dataset audit; all numerical checks run on CPU float64.

Transitions are checked independently against the recorded predecessor, rather
than rolling out accumulated errors. Distribution moments are descriptive only:
finite random samples are not required to have exactly uniform moments.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch

from dynamics.cartpole import CartPoleDynamics
from research.baseline.common import ROOT, sha256_file, write_manifest
from research.baseline.data_generator import rk4_step


class AuditFailure(ValueError):
    """Fatal audit failure carrying serializable evidence for the CLI report."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _stack(data, key):
    if not data:
        raise ValueError('empty trajectory collection')
    values = torch.stack([row[key].detach().cpu().double() for row in data])
    if not torch.isfinite(values).all():
        raise ValueError(f'{key}: non-finite values')
    return values


def audit_tensor_structure(data, expected_count, expected_T):
    if len(data) != expected_count or not data:
        raise ValueError(f'expected {expected_count} trajectories, got {len(data)}')
    shapes = dict(x0=(1, 4), u_seq=(expected_T, 1), x_seq=(expected_T, 1, 4))
    dtype = None
    for index, row in enumerate(data):
        if not isinstance(row, dict) or set(row) != set(shapes):
            raise ValueError(f'trajectory {index}: expected keys {list(shapes)}')
        for key, shape in shapes.items():
            value = row[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f'trajectory {index}: {key} must have shape {shape}')
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError(f'trajectory {index}: {key} must be float32/float64')
            if dtype is None:
                dtype = value.dtype
            if value.dtype != dtype:
                raise ValueError(f'trajectory {index}: inconsistent dtype for {key}')
            if not torch.isfinite(value).all():
                raise ValueError(f'trajectory {index}: {key} contains NaN/Inf')
    return dict(count=len(data), nan_count=0, inf_count=0, dtype=str(dtype),
                **{f'{key}_shape': list(shape) for key, shape in shapes.items()})


def audit_actuator_bounds(data, u_min=-10.0, u_max=10.0):
    if not math.isfinite(u_min) or not math.isfinite(u_max) or u_min >= u_max:
        raise ValueError('invalid actuator bounds')
    controls = _stack(data, 'u_seq')
    low, high = controls.min().item(), controls.max().item()
    if low < u_min or high > u_max:
        raise ValueError(f'actuator bounds violated: [{low}, {high}]')
    return dict(bounds_satisfied=True, min_u=low, max_u=high,
                mean_u=controls.mean().item(), variance_u=controls.var(unbiased=False).item(),
                uniform_expected_mean=(u_min + u_max) / 2,
                uniform_expected_variance=(u_max - u_min)**2 / 12)


def audit_initial_state_bounds(data):
    initial = _stack(data, 'x0').reshape(-1, 4)
    if (initial[:, 0].abs() > 4).any() or (initial[:, 2].abs() > .1).any():
        raise ValueError('initial position/angle bounds violated')
    if (initial[:, [1, 3]] != 0).any():
        raise ValueError('initial velocities must be zero')
    return dict(bounds_satisfied=True, x_min=initial[:, 0].min().item(),
                x_max=initial[:, 0].max().item(),
                theta_min=initial[:, 2].min().item(), theta_max=initial[:, 2].max().item(),
                x_dot_max=initial[:, 1].abs().max().item(),
                theta_dot_max=initial[:, 3].abs().max().item())


def audit_dynamics_mutation():
    x = torch.tensor([[.3, .2, 3.5, -.1]], dtype=torch.float64, requires_grad=True)
    before = x.detach().clone()
    try:
        result = CartPoleDynamics().dynamics(0., x, torch.tensor([[.7]], dtype=torch.float64))
        if not torch.equal(x.detach(), before):
            return False, 'dynamics changed its input'
        result.square().sum().backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            return False, 'missing or non-finite gradients'
    except RuntimeError as exc:
        return False, str(exc)
    return True, ''


def audit_transition_consistency(trajectories, tol=1e-4):
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be finite and positive')
    if not trajectories:
        raise ValueError('empty trajectory collection')
    system = CartPoleDynamics()
    maximum, squared, count = 0., 0., 0
    # Batch trajectories to bound memory, while checking every recorded step.
    for start in range(0, len(trajectories), 128):
        batch = trajectories[start:start + 128]
        recorded = _stack(batch, 'x_seq')
        previous = torch.cat((_stack(batch, 'x0').unsqueeze(1), recorded[:, :-1]), dim=1)
        controls = _stack(batch, 'u_seq')
        predicted = rk4_step(system.dynamics, 0., previous.reshape(-1, 4),
                             controls.reshape(-1, 1), system.dt).reshape_as(recorded)
        error = (predicted - recorded).abs()
        if not torch.isfinite(error).all():
            raise ValueError('non-finite transition error')
        maximum = max(maximum, error.max().item())
        squared += error.square().sum().item()
        count += error.numel()
    stats = dict(consistent=maximum < tol, max_error=maximum, rmse=math.sqrt(squared / count),
                 tol=tol, dt=system.dt, transitions_checked=count // 4)
    if maximum >= tol:
        raise AuditFailure(f'RK4 transition mismatch: max_error={maximum:.9g} >= tol={tol}', stats)
    return stats


def _trajectory_hash(row):
    # Hash states independently of x0/controls, detecting reused state sequences.
    states = row['x_seq'].detach().cpu().double().contiguous()
    # Canonicalize signed zero before hashing numerically identical trajectories.
    states = torch.where(states == 0, torch.zeros_like(states), states)
    return hashlib.sha256(states.numpy().tobytes()).hexdigest()


def audit_train_test_separation(train_data, test_data):
    train = _stack(train_data, 'x0').reshape(-1, 4)
    test = _stack(test_data, 'x0').reshape(-1, 4)
    hashes = {_trajectory_hash(row) for row in train_data}
    collisions = sum(_trajectory_hash(row) in hashes for row in test_data)
    if collisions:
        raise ValueError(f'train/test trajectory overlap: {collisions}')
    minimum = float('inf')
    for start in range(0, len(train), 256):
        # Direct differences avoid cancellation near the 1e-6 leakage threshold.
        distances = (train[start:start + 256, None] - test[None]).square().sum(-1).sqrt()
        minimum = min(minimum, distances.min().item())
    if minimum < 1e-6:
        raise ValueError(f'train/test initial condition leakage: distance={minimum}')
    return dict(disjoint=True, min_x0_distance=minimum, trajectory_hash_collisions=collisions)


def audit_cartpole_dataset(train_path: str | Path, test_path: str | Path, tol=1e-4) -> dict:
    report = dict(system='cartpole', status='passed', splits={}, errors=[])
    datasets = []
    for name, path, expected in (('train', train_path, 10000), ('test', test_path, 1000)):
        data = torch.load(path, map_location='cpu', weights_only=True)
        datasets.append(data)
        # Structural/actuator violations stop immediately, before numerical work.
        stats = dict(path=str(path), sha256=sha256_file(path),
                     structure=audit_tensor_structure(data, expected, 250),
                     actuators=audit_actuator_bounds(data),
                     initial_states=audit_initial_state_bounds(data))
        report['splits'][name] = stats
        try:
            stats['transitions'] = audit_transition_consistency(data, tol)
        except AuditFailure as exc:
            stats['transitions'] = exc.report
            report['errors'].append(f'{name}: {exc}')
    safe, message = audit_dynamics_mutation()
    report['dynamics'] = dict(mutation_free=safe, backward_safe=safe)
    if not safe:
        report['errors'].append(f'dynamics mutation/autograd audit failed: {message}')
    report['separation'] = audit_train_test_separation(*datasets)
    if report['errors']:
        report['status'] = 'failed'
        raise AuditFailure('; '.join(report['errors']), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/cartpole_train.pt')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/cartpole_test.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction/cartpole_data_audit.yaml')
    parser.add_argument('--tol', type=float, default=1e-4)
    args = parser.parse_args()
    try:
        report = audit_cartpole_dataset(args.train, args.test, args.tol)
    except AuditFailure as exc:
        write_manifest(args.output, exc.report)
        parser.exit(1, f'AUDIT FAILED: {exc}\nreport: {args.output}\n')
    write_manifest(args.output, report)
    print('ALL MATHEMATICAL INTEGRITY CHECKS PASSED')
    for name, stats in report['splits'].items():
        print(f"{name}: {stats['structure']['count']} trajectories, SHA-256 {stats['sha256']}, "
              f"RK4 max error {stats['transitions']['max_error']:.6g}")
    print(f'report: {args.output}')


if __name__ == '__main__':
    main()
