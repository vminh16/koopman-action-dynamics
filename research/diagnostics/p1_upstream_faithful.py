"""P1 upstream-code-faithful reproduction (diagnostic; preregistered in TRACKING.md).

Question: does the authors' code protocol (sampler max_start=5) and/or the
pre-2025-10-19 dense-P operator (upstream 91a47d0) reproduce Table 1?
Reuses the verified P1 harness (Euler rollout, original losses, upstream-style
last-batch ReduceLROnPlateau). Does not modify baseline code or old runs.
"""
import argparse
import json
import random
import statistics
import time
import types

import torch

from research.baseline.common import ROOT, build_module, sha256_file
from research.diagnostics.p1_audit import experiment as e
from my_utils.dataset import SampledMultiTrajectoryDataset

OUT = ROOT / 'outputs/reproduction/p1_upstream_faithful_2026-11-16'
SEEDS = (42, 43, 44)
FRACTIONS = (0.01, 0.05, 0.1, 0.3)
# arm -> (operator, sampler). winfrac arms: 5%-only check, kept for replay.
ARMS = {
    'spd_full': ('spd', 'traj_full'),         # repo C02/P1 baseline
    'spd_s5': ('spd', 'traj_s5'),
    'spd_s5_winfrac': ('spd', 'win_s5'),      # exact public-code protocol (Koopman ctor)
    'dense_s5': ('dense', 'traj_s5'),         # Table-1-faithful candidate
    'dense_s5_winfrac': ('dense', 'win_s5'),
    'dense_full': ('dense', 'traj_full'),
}
GRID_ARMS = ('spd_full', 'spd_s5', 'dense_s5', 'dense_full')
PAPER = {'pendulum': {0.01: 0.3747, 0.05: 0.0318, 0.1: 0.0114, 0.3: 0.0061},
         'cartpole': {0.01: 0.0095, 0.05: 0.0024, 0.1: 0.001153, 0.3: 0.000571}}


def config(system, seed, fraction):
    """Locked 5% config with only the trajectory fraction changed."""
    cfg = dict(e.config(system, seed))
    cfg['train_fraction'] = fraction
    cfg['num_train_trajectories'] = max(1, int({'pendulum': 5000, 'cartpole': 10000}[system] * fraction))
    return cfg


def run_dir(system, fraction, arm, seed):
    return OUT / 'runs' / f'{system}-f{fraction:g}-{arm}-s{seed}'


def use_dense_p(model, seed):
    """Operator of upstream commit 91a47d0: P = H H^T + eps I, dense N x N, eps=1e-2, H*0.01."""
    op = model.model.operator.core
    g = torch.Generator().manual_seed(900000 + seed)
    op.H_dense = torch.nn.Parameter(torch.randn(op.N, op.N, generator=g) * 0.01)
    op.eps_dense = 1e-2

    def build_P(self):
        return self.H_dense @ self.H_dense.T + self.eps_dense * torch.eye(self.N, dtype=self.H_dense.dtype, device=self.H_dense.device)
    op.build_P = types.MethodType(build_P, op)
    op.H_head.requires_grad_(False); op.H_tail.requires_grad_(False)  # unused block parameters


def make_model(system, seed, operator, fraction=0.05):
    torch.manual_seed(seed)
    model = build_module(config(system, seed, fraction))
    if operator == 'dense':
        use_dense_p(model, seed)
    return model


def load_split(system, seed, fraction):
    """Train subset + 200 validation trajectories outside all three seeds' subsets (as P1)."""
    raw = torch.load(ROOT / e.config(system, seed)['data_train'], map_location='cpu', weights_only=True)
    _, train_idx = e.select_training_trajectories(raw, config(system, seed, fraction))
    occupied = set()
    for s in SEEDS:
        occupied.update(e.select_training_trajectories(raw, config(system, s, fraction))[1])
    val_idx = random.Random(20261115).sample([i for i in range(len(raw)) if i not in occupied], 200)
    val = [raw[i] for i in val_idx]
    validation = (torch.stack([r['x0'].reshape(-1) for r in val]), torch.stack([r['u_seq'][:15] for r in val]),
                  torch.stack([r['x_seq'][:15].reshape(15, -1) for r in val]))
    return raw, train_idx, validation


def windows(raw, train_idx, seed, sampler, fraction):
    if sampler == 'win_s5':  # upstream run_train.py: Subset of int(frac*N) windows over ALL trajectories
        ds = SampledMultiTrajectoryDataset(raw, 15, 20, seed)
        ds.samples = [(i, s) for i in range(len(raw)) for s in range(5)]
        keep = torch.randperm(len(ds.samples), generator=torch.Generator().manual_seed(seed))
        ds.samples = [ds.samples[i] for i in keep[:max(1, int(fraction * len(ds.samples)))].tolist()]
    else:
        ds = SampledMultiTrajectoryDataset([raw[i] for i in train_idx], 15, 20, seed)  # traj_full = C02/P1
        if sampler == 'traj_s5':  # upstream max_start=5: all starts 0..4
            ds.samples = [(i, s) for i in range(len(train_idx)) for s in range(5)]
    items = [ds[i] for i in range(len(ds))]
    info = dict(windows=len(items), trajectories=len({t for t, _ in ds.samples}), max_start=max(s for _, s in ds.samples))
    return [torch.stack([it[k] for it in items]) for k in range(3)], info


def train(system, fraction, arm, seed, epochs, device='cpu'):
    torch.set_num_threads(1)
    operator, sampler = ARMS[arm]
    d = run_dir(system, fraction, arm, seed)
    d.mkdir(parents=True, exist_ok=True)
    if (d / 'result.json').exists():
        print('SKIP', d.name, flush=True); return
    raw, train_idx, validation = load_split(system, seed, fraction)
    tensors, info = windows(raw, train_idx, seed, sampler, fraction)
    tensors = [t.to(device) for t in tensors]; validation = [t.to(device) for t in validation]
    model = make_model(system, seed, operator, fraction).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.1, patience=10)
    order_rng = torch.Generator().manual_seed(700000 + seed)
    rec = dict(run_id=d.name, system=system, fraction=fraction, arm=arm, seed=seed, operator=operator,
               sampler=sampler, device=device, **info, script_sha256=sha256_file(__file__), history=[])
    t0 = time.monotonic(); steps = 0
    for epoch in range(epochs):
        model.train(); last = None
        for ids in torch.randperm(len(tensors[0]), generator=order_rng).split(64):
            opt.zero_grad(set_to_none=True)
            total, _, _ = e.losses(model, *[t[ids] for t in tensors])
            if not torch.isfinite(total):
                rec.update(status='failed_nonfinite', epoch=epoch)
                (d / 'result.json').write_text(json.dumps(rec, indent=2)); return
            total.backward(); opt.step(); steps += 1; last = float(total.detach())
        sched.step(last)  # upstream semantics: Lightning step-level train loss as monitor
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                A, B = e.matrices(model)
                yp = e.rollout(model, *validation[:2], A, B)[:, :, :model.state_dim]
                val = (yp.double() - validation[2].double()).square().mean().item()
            rec['history'].append(dict(epoch=epoch + 1, steps=steps, lr=opt.param_groups[0]['lr'], val_first15=val))
            print(d.name, rec['history'][-1], flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, d / 'checkpoint.pt')
    rec.update(status='complete', steps=steps, elapsed_seconds=time.monotonic() - t0,
               final_lr=opt.param_groups[0]['lr'], checkpoint_sha256=sha256_file(d / 'checkpoint.pt'))
    (d / 'result.json').write_text(json.dumps(rec, indent=2))


@torch.no_grad()
def evaluate():
    torch.set_num_threads(4)
    out = {}
    for system in ('pendulum', 'cartpole'):
        raw = torch.load(ROOT / f'data/{system}_test.pt', map_location='cpu', weights_only=True)
        n = 2 if system == 'pendulum' else 4
        x = torch.stack([r['x0'].reshape(n) for r in raw]); u = torch.stack([r['u_seq'][:15] for r in raw])
        y = torch.stack([r['x_seq'][:15].reshape(15, n) for r in raw]).double()
        random.seed(42); sub = random.sample(range(len(raw)), 100)  # upstream evaluate_model subset
        for fraction in FRACTIONS:
            for arm, (operator, _) in ARMS.items():
                rows = []
                for seed in SEEDS:
                    d = run_dir(system, fraction, arm, seed)
                    if not (d / 'result.json').exists():
                        continue
                    res = json.loads((d / 'result.json').read_text())
                    if res['status'] != 'complete':
                        rows.append(dict(seed=seed, status=res['status'])); continue
                    assert sha256_file(d / 'checkpoint.pt') == res['checkpoint_sha256']
                    model = make_model(system, seed, operator, fraction)
                    model.load_state_dict(torch.load(d / 'checkpoint.pt', map_location='cpu', weights_only=True), strict=True)
                    model.eval()
                    A, B = e.matrices(model)
                    se = (e.rollout(model, x, u, A, B)[:, :, :n].double() - y).square().mean((1, 2))
                    C = torch.eye(n, A.shape[0], dtype=A.dtype)
                    reach = torch.cat([C @ torch.linalg.matrix_power(A, k) @ B for k in range(n)], 1)
                    rows.append(dict(seed=seed, status='complete', test_H15=se.mean().item(),
                                     upstream_eval_100traj_H15=se[sub].mean().item(), CB=B[:n, 0].tolist(),
                                     output_rank=int(torch.linalg.matrix_rank(reach.double())),
                                     steps=res['steps'], final_lr=res['final_lr'], windows=res['windows'],
                                     trajectories=res['trajectories']))
                if not rows:
                    continue
                vals = [r['test_H15'] for r in rows if r['status'] == 'complete']
                summ = dict(runs=rows, paper=PAPER[system][fraction])
                if len(vals) == len(SEEDS):
                    mean = statistics.mean(vals)
                    summ.update(mean=mean, std=statistics.stdev(vals), ratio_to_paper=mean / PAPER[system][fraction],
                                mean_100traj=statistics.mean(r['upstream_eval_100traj_H15'] for r in rows),
                                within_2x=mean <= 2 * PAPER[system][fraction])
                out[f'{system}/f{fraction:g}/{arm}'] = summ
                print(system, fraction, arm, {k: v for k, v in summ.items() if k != 'runs'}, flush=True)
    # summary.json holds the CPU 5% check (old run-dir naming); keep it as the device control.
    (OUT / 'summary_grid.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('train', 'eval', 'jobs'), required=True)
    p.add_argument('--system', choices=('pendulum', 'cartpole'))
    p.add_argument('--fraction', type=float, default=0.05, choices=FRACTIONS)
    p.add_argument('--arm', choices=tuple(ARMS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.stage == 'train':
        train(a.system, a.fraction, a.arm, a.seed, a.epochs, a.device)
    elif a.stage == 'jobs':  # one "system fraction arm seed" line per pending grid run, for xargs
        for s in ('pendulum', 'cartpole'):
            for f in FRACTIONS:
                for arm in GRID_ARMS:
                    for seed in SEEDS:
                        if not (run_dir(s, f, arm, seed) / 'result.json').exists():
                            print(s, f, arm, seed)
    else:
        evaluate()
