"""Paired diagnostic controls on frozen P2 data; never overwrites historical gates.

Run: python -m research.diagnostics.p2_followup --stage e05|verify|train|recompute
General heads require this loader, not the original SPD model's inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.integrate import quad_vec
from scipy.linalg import expm

from research.baseline.common import ROOT, git_provenance, load_config, sha256_file
from research.benchmarks import load_benchmark
from research.diagnostics import run_p2_diagnostics as p2
from research.diagnostics.metrics import (
    compute_action_response_error, compute_continuous_gramian_and_margin,
    compute_h3_feasibility, compute_vector_field_mismatches, _van_loan_gramian,
)
from training.loss import (
    prediction_loss, encoder_loss_from_true, encoder_loss_from_pred,
    controllability_loss,
)

OUT = ROOT / 'outputs/diagnosis/p2_followup_2026-11-16'
OLD = ROOT / 'outputs/diagnosis/p2_failure_report.yaml'
ARMS = ('baseline', 'epochmean', 'noenc', 'general', 'general_noenc_epochmean')
SEEDS = tuple(range(42, 47))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def old_report():
    return yaml.safe_load(OLD.read_text())


def checked_data(entry):
    path = ROOT / entry['path']
    if sha256_file(path) != entry['sha256']:
        raise ValueError(f'Dataset hash mismatch: {path}')
    return torch.load(path, weights_only=True)


def make_model(seed, general=False):
    torch.manual_seed(seed)
    sys = load_benchmark('rotation_family', a=0)['instance']
    model = p2.build_official_module(sys, 6)
    if general:
        op = model.model.operator.core
        model.register_buffer('diagnostic_head_base', op.build_P()[:2, :2].detach().clone())
        with torch.no_grad():
            op.H_head.zero_()
    return model


def matrices(model):
    op = model.model.operator.core
    if not hasattr(model, 'diagnostic_head_base'):
        return op.get_AB()
    head = model.diagnostic_head_base @ torch.matrix_exp(op.H_head)
    P = torch.block_diag(head, op.build_P()[2:, 2:])
    return P @ op.build_Ac() @ torch.linalg.inv(P), P @ op.build_Bc()


def rollout(model, x, u):
    A, B = matrices(model)
    z = model.model.encoder(x)
    values = []
    for action in u.unbind(1):
        z = z + model.dt * (z @ A.T + action @ B.T)
        values.append(z)
    return torch.stack(values, 1)


def loss(model, batch, noenc=False):
    x, u, y = batch
    z = rollout(model, x, u)
    pred = prediction_loss(z[:, :, :2], y)
    total = pred
    if not noenc:
        total = total + encoder_loss_from_true(y, z[:, :, 2:], model.model.encoder)
        total = total + .5 * encoder_loss_from_pred(z[:, :, :2], z[:, :, 2:], model.model.encoder)
    A, B = matrices(model)
    return total + .005 * controllability_loss(A, B, state_dim=2, T=.3, dt=.02, method='coupled')


def verify():
    result = {}
    for seed in SEEDS:
        model = make_model(seed)
        general = make_model(seed, True)
        batch = (torch.randn(4, 2), torch.randn(4, 15, 1), torch.randn(4, 15, 2))
        official = model(*batch[:2])
        cached = rollout(model, *batch[:2])
        torch.testing.assert_close(official, cached, atol=1e-6, rtol=1e-5)
        for a, b in zip(matrices(model), matrices(general)):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        model.log = lambda *args, **kwargs: None
        upstream_loss = model.training_step(batch, 0)
        cached_loss = loss(model, batch)
        torch.testing.assert_close(upstream_loss, cached_loss)
        params = list(model.parameters())
        gradients = torch.autograd.grad(upstream_loss, params)
        cached_gradients = torch.autograd.grad(cached_loss, params)
        max_error = max(float((a-b).abs().max()) for a, b in zip(gradients, cached_gradients))
        for a, b in zip(gradients, cached_gradients):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        result[str(seed)] = {'forward_max_error': float((official-cached).abs().max().detach()),
                             'loss_gradient_max_error': max_error, 'general_initial_AB_match': True}
    # Independent finite-difference check of the general head's matrix exponential.
    M = torch.randn(2, 2, dtype=torch.float64, requires_grad=True) * .01
    assert torch.autograd.gradcheck(torch.matrix_exp, (M,))
    save(OUT / 'harness_verification.json', result)
    print('Harness verification PASS', flush=True)


def refresh_e05():
    entries = {}
    for system, seeds in (('pendulum', (42,)), ('cartpole', (42, 43, 44))):
        for seed in seeds:
            suffix = 'yaml' if system == 'pendulum' else 'yml'
            cfg_path = f'configs/research/{system}_controllable_win15_frac0.05_seed{seed}.{suffix}'
            cfg = load_config(ROOT / cfg_path)
            first = p2.evaluate_e05_system(system, cfg_path, cfg['checkpoint'])
            second = p2.evaluate_e05_system(system, cfg_path, cfg['checkpoint'])
            assert first['status'] == 'verified', first
            assert first == second, 'E05 repeat differed'
            first['config_sha256'] = sha256_file(ROOT / cfg_path)
            entries[f'{system}-s{seed}'] = first
    save(OUT / 'e05_refresh.json', {'historical_report_sha256': sha256_file(OLD),
        'entries': entries, 'repeat_verdict': 'PASS',
        'meaning': 'checkpoint/config/data identity and repeatability; not paper or dataset-transition certification'})
    print('E05 refresh PASS: Pendulum and all three C02 checkpoints', flush=True)


def evaluate(model, sys, data, probe):
    X, U, Y = p2._stack_transitions(data)
    with torch.no_grad():
        A, B = matrices(model)
        adapter = SimpleNamespace(A=A, B=B, state_dim=2, encode=model.model.encoder)
        z = model.model.encoder(X)
        one = z + .02 * (z @ A.T + U @ B.T)
        mse = float((one[:, :2].double()-Y.double()).square().mean())
        x0 = torch.cat([row['x0'].reshape(1, -1) for row in data])
        horizons = {}
        for H in (15, 30, 50):
            actions = torch.stack([row['u_seq'][:H] for row in data])
            truth = torch.stack([row['x_seq'][:H] for row in data])
            predicted = rollout(model, x0, actions)[:, :, :2]
            horizons[str(H)] = float((predicted.double()-truth.double()).square().mean())
        response = compute_action_response_error(adapter, sys, probe['x'], probe['u'], probe['u_ref'])
        G = sys.analytic_Gx(probe['x']).double()
        delta = probe['u'].double()-probe['u_ref'].double()
        # Action-weighted least-squares best constant map on the fixed probe.
        weights = delta[:, 0].square()
        best = (weights[:, None, None] * G).sum(0) / weights.sum()
        floor = float((weights * (G-best).square().sum((1, 2))).sum() / weights.sum())
        learned_gap = float((B[:2].double()-best).square().sum())
    vf = compute_vector_field_mismatches(adapter, sys, probe['x'][:8])
    An, Bn = A.detach().double().numpy(), B.detach().double().numpy()
    C = np.eye(2, 6)
    gramian = compute_continuous_gramian_and_margin(An, Bn, C, T=.24, method='expm')
    Wq, _ = quad_vec(lambda t: C @ expm(An*t) @ Bn @ Bn.T @ expm(An.T*t) @ C.T, 0, .24)
    W = C @ _van_loan_gramian(An, Bn@Bn.T, .24) @ C.T
    np.testing.assert_allclose(W, Wq, atol=1e-11, rtol=1e-8)
    h3 = compute_h3_feasibility(adapter, sys, probe['x'][:8], gramian['rho_T'])
    return {'one_step_mse': mse, 'open_loop_mse': horizons,
        'response_normalized': response['normalized'], 'response_raw': response['raw'],
        'best_constant_normalized_floor': floor, 'learned_CB_sq_distance_to_best': learned_gap,
        'CB': Bn[:2].tolist(), 'd_orient_p95': vf['d_orient_p95'],
        'gramian': {k: gramian[k] for k in ('rho_T', 'lambda_min_Wy', 'cond_Wy', 'rank_deficient')},
        'gramian_quadrature_max_error': float(np.max(np.abs(W-Wq))),
        'h3': {k: h3[k] for k in ('d_star_p95', 'd_orient_p95_below_rho', 'd_star_p95_below_rho')}}


def train():
    report = old_report()
    base_data = checked_data(report['rotation_sweep']['a_0.0']['datasets']['eval'])
    probe = p2.build_response_probe(base_data, load_benchmark('rotation_family', a=0)['instance'])
    for epochs, arms in ((25, ARMS), (100, ('baseline', 'general_noenc_epochmean'))):
        for name, entry in report['rotation_sweep'].items():
            sys = load_benchmark('rotation_family', a=entry['a'])['instance']
            eval_data = checked_data(entry['datasets']['eval'])
            for seed in SEEDS:
                train_entry = entry['datasets']['train'][str(seed)]
                data = checked_data(train_entry)
                ds = p2.build_training_dataset(data, seed=seed)
                tensors = tuple(torch.stack([ds[i][k] for i in range(len(ds))]) for k in range(3))
                for arm in arms:
                    directory = OUT / f'{name}-{arm}-e{epochs}-s{seed}'
                    if (directory / 'result.json').exists():
                        continue
                    model = make_model(seed, 'general' in arm)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=10)
                    rng = torch.Generator().manual_seed(700000+seed)
                    history = []
                    for epoch in range(epochs):
                        totals = []
                        for ids in torch.randperm(len(ds), generator=rng).split(64):
                            optimizer.zero_grad(set_to_none=True)
                            total = loss(model, tuple(t[ids] for t in tensors), 'noenc' in arm)
                            if not torch.isfinite(total):
                                raise ValueError('Nonfinite loss')
                            total.backward()
                            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                                raise ValueError('Nonfinite gradient')
                            optimizer.step()
                            totals.append((float(total.detach()), len(ids)))
                        mean = sum(t*n for t, n in totals)/len(ds)
                        scheduler.step(mean if 'epochmean' in arm else totals[-1][0])
                        history.append({'epoch': epoch+1, 'mean_loss': mean, 'last_loss': totals[-1][0], 'lr': optimizer.param_groups[0]['lr']})
                    directory.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), directory / 'checkpoint.pt')
                    metrics = evaluate(model, sys, eval_data, probe)
                    record = {'status': 'complete', 'a': entry['a'], 'seed': seed, 'arm': arm,
                        'epochs': epochs, 'updates': epochs*((len(ds)+63)//64), 'windows': len(ds),
                        'window_samples': ds.samples, 'shuffle_seed': 700000+seed,
                        'train': train_entry, 'eval': entry['datasets']['eval'],
                        'history': history, 'metrics': metrics,
                        'checkpoint_sha256': sha256_file(directory/'checkpoint.pt'),
                        'source_sha256': sha256_file(__file__), 'git': git_provenance()}
                    save(directory/'result.json', record)
                    print(directory.name, metrics['response_normalized'], flush=True)


def recompute():
    report = old_report()
    probe = p2.build_response_probe(checked_data(report['rotation_sweep']['a_0.0']['datasets']['eval']),
                                   load_benchmark('rotation_family', a=0)['instance'])
    entries = []
    for path in sorted(OUT.glob('*/result.json')):
        rec = json.loads(path.read_text())
        checked_data(rec['train'])
        data = checked_data(rec['eval'])
        assert sha256_file(path.parent/'checkpoint.pt') == rec['checkpoint_sha256']
        assert rec['source_sha256'] == sha256_file(__file__)
        model = make_model(rec['seed'], 'general' in rec['arm'])
        model.load_state_dict(torch.load(path.parent/'checkpoint.pt', weights_only=True), strict=True)
        sys = load_benchmark('rotation_family', a=rec['a'])['instance']
        actual = evaluate(model, sys, data, probe)
        def compare(a, b):
            if isinstance(a, dict):
                for k in a: compare(a[k], b[k])
            elif isinstance(a, list):
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-7)
            elif a is None or isinstance(a, bool):
                assert a == b
            else:
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-7)
        compare(actual, rec['metrics'])
        entries.append(path.parent.name)
    expected = {f'a_{a}-{arm}-e{epochs}-s{seed}' for epochs, arms in
                ((25, ARMS), (100, ('baseline', 'general_noenc_epochmean')))
                for a in (0.0, .5, 1.0, 2.0) for arm in arms for seed in SEEDS}
    assert set(entries) == expected, (len(entries), len(expected))
    save(OUT/'recompute.json', {'verdict': 'PASS', 'runs': len(entries), 'entries': entries,
                              'historical_report_sha256': sha256_file(OLD)})
    print(f'Recompute PASS: {len(entries)} checkpoints', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('e05', 'verify', 'train', 'recompute'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {'e05': refresh_e05, 'verify': verify, 'train': train, 'recompute': recompute}[args.stage]()


if __name__ == '__main__':
    main()
