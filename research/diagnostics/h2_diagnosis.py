"""H2 readiness diagnosis: is the frozen fixed-B baseline's failure a model-class limit?

Preregistered in TRACKING.md (2026-11-17). Diagnostic only; trains the author's model
(frozen general-head baseline, docs/BASELINE.md) and its own Appendix G.5 variant.
D1 rotation sweep a in {0,.5,1,2}; D2 controls at a=2 (data N128, latent 12/24, G.5);
D3 CartPole stratified evaluation of existing P1 grid checkpoints (no training).
"""
import argparse
import json
import math
import statistics

import numpy as np
import torch
from scipy.linalg import expm

from research.baseline.common import ROOT, sha256_file
from research.benchmarks import load_benchmark
from research.diagnostics import p2_config_check as c
from research.diagnostics import p2_followup as old
from research.diagnostics import run_p2_diagnostics as p2

OUT = ROOT/'outputs/diagnosis/h2_diagnosis_2026-11-17'
SEEDS = (92, 93, 94, 95, 96)
A_VALUES = (0., .5, 1., 2.)
# arm -> (a values, ntrain, latent N, use_modulated_u); every run = 3000 updates, LR .01.
ARMS = {'base': (A_VALUES, 32, 6, False), 'data128': ((2.,), 128, 6, False),
        'latent12': ((2.,), 32, 12, False), 'latent24': ((2.,), 32, 24, False),
        'g5': ((0., 2.), 32, 6, True)}
UPDATES = 3000


def make_model(seed, general=True, latent=6, modulated=False):
    """Frozen general-head baseline (p2_followup.make_model) with optional N / G.5."""
    torch.manual_seed(seed)
    sys = load_benchmark('rotation_family', a=0)['instance']
    model = p2.build_official_module(sys, latent, {**p2.P2_BASELINE_CONFIG, 'use_modulated_u': modulated})
    op = model.model.operator.core
    model.register_buffer('diagnostic_head_base', op.build_P()[:2, :2].detach().clone())
    with torch.no_grad():
        op.H_head.zero_()
    return model


def effective_input(model, z, action):
    m = model.model
    return m.u_encoder(z[:, :2], action) if m.use_modulated_u else action


def rollout(model, x, u, AB=None):
    """Euler latent rollout identical to p2_config_check.rollout, plus the G.5 input map."""
    A, B = old.matrices(model) if AB is None else AB
    z = model.model.encoder(x)
    values = []
    for action in u.unbind(1):
        z = z+model.dt*(z@A.T+effective_input(model, z, action)@B.T)
        values.append(z)
    return torch.stack(values, 1)


def response(model, X):
    """Physical instantaneous input response r(x)=d xdot/du at states X (symmetric +-1)."""
    A, B = old.matrices(model)
    z = model.model.encoder(X)
    one = torch.ones(len(X), 1)
    return ((effective_input(model, z, one)-effective_input(model, z, -one))@B.T/2)[:, :2]


def run(arm, a, seed):
    alist, ntrain, latent, modulated = ARMS[arm]
    assert a in alist
    windows = 20*ntrain
    cfg = {'lr': .01, 'scheduler': 'constant', 'epochs': UPDATES*64//windows, 'noenc': False}
    assert cfg['epochs']*math.ceil(windows/64) == UPDATES
    gen = c.generate
    patched = dict(generate=lambda s, split, *ar, **kw: gen(s, split, *ar, **({**kw, 'count': ntrain} if split == 'train' else kw)),
                   make_model=lambda s, general=False: make_model(s, True, latent, modulated), rollout=rollout)
    saved = {k: getattr(c, k) for k in patched}
    try:
        for k, v in patched.items():
            setattr(c, k, v)
        record = c.train_run(f'{arm}-a{a:g}-s{seed}', cfg, seed, compatible=False, a=a, general=True)
    finally:
        for k, v in saved.items():
            setattr(c, k, v)
    assert record['status'] != 'complete' or record['updates'] == UPDATES
    return record


def jobs():
    return [(arm, a, s) for arm, (alist, *_) in ARMS.items() for a in alist for s in SEEDS]


def make_data():
    """Generate every dataset once, sequentially, so parallel trainers only read."""
    for arm, a, s in jobs():
        ntrain = ARMS[arm][1]
        c.generate(s, 'train', False, a, count=ntrain)
        c.generate(s, 'validation', False, a)
        c.generate(s, 'test', False, a)


def rotation_metrics(model, a, seed, ntrain):
    test, test_id = c.generate(seed, 'test', False, a)
    train, _ = c.generate(seed, 'train', False, a, count=ntrain)
    X = torch.cat([torch.cat([r['x0'], r['x_seq'][:-1]]) for r in test])
    U = torch.cat([r['u_seq'] for r in test]).double()
    T = torch.cat([torch.cat([r['x0'], r['x_seq'][:-1]]) for r in train]).double()
    with torch.no_grad():
        G = torch.stack([torch.cos(a*X[:, 0]), torch.sin(a*X[:, 0])], 1).double()
        R = response(model, X).double()
        resp = float((G-R).square().sum(1).mean())
        floor = float((G-G.mean(0)).square().sum(1).mean())  # best constant map (unit-norm G)
        A, B = old.matrices(model)
        z = model.model.encoder(X)
        drift = (z@A.T)[:, :2].double()-X.double()@torch.tensor(c.F.T)
        inputs = U*(R-G)
        x0 = torch.cat([r['x0'] for r in test]); u = torch.stack([r['u_seq'][:15] for r in test])
        y = torch.stack([r['x_seq'][:15] for r in test]).double()
        h15 = float((rollout(model, x0, u)[:, :, :2].double()-y).square().mean())
        state = x0.double().numpy(); E = expm(.02*c.F); ign = []
        for _ in range(15):
            state = state@E.T; ign.append(state.copy())
        ignore = float(np.mean((np.stack(ign, 1)-y.numpy())**2))
        scale = T.std(0).clamp_min(1e-8)
        cover = float(torch.quantile(torch.cdist(X.double()/scale, T/scale).min(1).values, .95))
    return dict(test_data=test_id, response=resp, fixed_B_floor=floor, excess=resp-floor,
                drift_mse=float(drift.square().mean()), input_mse=float(inputs.square().mean()),
                H15=h15, H15_ratio_to_ignore_input=h15/ignore, coverage_p95=cover,
                CB=old.matrices(model)[1][:2, 0].tolist())


def evaluate_rotation():
    rows = {}
    for arm, a, s in jobs():
        d = c.OUT/'runs'/f'{arm}-a{a:g}-s{s}'
        rec = json.loads((d/'result.json').read_text())
        assert (rec['name'], rec['a'], rec['seed'], rec['general'], rec['compatible']) == (d.name, a, s, True, False), d
        row = dict(arm=arm, a=a, seed=s, status=rec['status'])
        if rec['status'] == 'complete':
            ck = rec['checkpoints'][str(rec['epochs_completed'])]
            assert sha256_file(ROOT/ck['path']) == ck['sha256']
            _, ntrain, latent, modulated = ARMS[arm]
            model = make_model(s, True, latent, modulated)
            model.load_state_dict(torch.load(ROOT/ck['path'], weights_only=True), strict=True)
            model.eval()
            row.update(rotation_metrics(model, a, s, ntrain), checkpoint=ck)
        rows[f'{arm}/a{a:g}/s{s}'] = row
    return rows


def cell(rows, arm, a, key='response'):
    v = [r[key] for r in rows.values() if r['arm'] == arm and r['a'] == a and r['status'] == 'complete']
    return statistics.mean(v) if len(v) == len(SEEDS) else None


def verdict(rows):
    """Preregistered rules R1-R6 (TRACKING.md, 2026-11-17)."""
    base = [cell(rows, 'base', a) for a in A_VALUES]
    a2 = [r for r in rows.values() if r['arm'] == 'base' and r['a'] == 2. and r['status'] == 'complete']
    allc = [r for r in rows.values() if r['status'] == 'complete']
    R = {}
    R['R1_symptom'] = (sum(r['response'] > .05 for r in a2) >= 4 and base[3] >= 1.5*base[0]
                       and all(x <= y for x, y in zip(base, base[1:])))
    R['R2_at_fixed_B_floor'] = sum(r['excess'] <= .25*r['fixed_B_floor'] for r in a2) >= 4
    R['R3_data_does_not_fix'] = cell(rows, 'data128', 2.) >= .8*base[3]
    R['R4_latent_does_not_fix'] = min(cell(rows, 'latent12', 2.), cell(rows, 'latent24', 2.)) >= .8*base[3]
    R['R5_G5_fixes'] = cell(rows, 'g5', 2.) <= .5*base[3]
    R['R6_coverage'] = all(r['coverage_p95'] <= 3. for r in allc) and len(allc) == len(rows)
    ok = R['R1_symptom'] and R['R2_at_fixed_B_floor'] and R['R3_data_does_not_fix'] and R['R4_latent_does_not_fix'] and R['R6_coverage']
    R['diagnosis'] = ('NO_GO_CHEAP_FIX_G5' if R['R5_G5_fixes'] else
                      'FIXED_B_CLASS_LIMIT_CONFIRMED' if ok else 'NOT_CONFIRMED')
    R['base_response_by_a'] = dict(zip(map(str, A_VALUES), base))
    return R


# ---- D3: CartPole stratified (existing P1 grid checkpoints, no training) ----
CART_ARMS = ('dense_s5', 'dense_full')
BINS = ((0., math.pi/6), (math.pi/6, math.pi/2), (math.pi/2, math.pi+1e-9))


def cartpole_true_response(dyn, X):
    u = torch.zeros(len(X), 1, dtype=X.dtype)
    return (dyn.dynamics(0., X, u+1)-dyn.dynamics(0., X, u-1))/2  # exact: affine in force


def evaluate_cartpole(per_traj=10, frac=.05):
    from dynamics import CartPoleDynamics
    from research.diagnostics import p1_upstream_faithful as f
    from research.diagnostics.p1_audit import experiment as e
    dyn = CartPoleDynamics()
    raw = torch.load(ROOT/'data/cartpole_test.pt', map_location='cpu', weights_only=True)
    X = torch.stack([torch.cat([r['x0'].reshape(1, 4), r['x_seq'].reshape(250, 4)]) for r in raw])
    U = torch.stack([r['u_seq'] for r in raw])
    g = torch.Generator().manual_seed(20261117)
    starts = torch.randint(0, 250-15, (len(raw), per_traj), generator=g)
    ti = torch.arange(len(raw))[:, None].expand_as(starts).reshape(-1); st = starts.reshape(-1)
    x0 = X[ti, st]; u = torch.stack([U[ti, st+k] for k in range(15)], 1)
    y = torch.stack([X[ti, st+1+k] for k in range(15)], 1).double()
    wrap = lambda t: (t+math.pi) % (2*math.pi)-math.pi
    band = wrap(x0[:, 2].double()).abs()
    G = cartpole_true_response(dyn, x0.double())
    out = {'windows': len(x0), 'window_seed': 20261117, 'per_trajectory': per_traj, 'fraction': frac}
    for arm in CART_ARMS:
        for seed in (42, 43, 44):
            d = f.run_dir('cartpole', frac, arm, seed)
            res = json.loads((d/'result.json').read_text())
            # Guard against loading the wrong grid cell (arm/operator/fraction/seed).
            assert (res['status'], res['arm'], res['operator'], res['fraction'], res['seed'], res['system']) == \
                ('complete', arm, 'dense', frac, seed, 'cartpole'), d
            assert sha256_file(d/'checkpoint.pt') == res['checkpoint_sha256']
            model = f.make_model('cartpole', seed, 'dense', frac)
            model.load_state_dict(torch.load(d/'checkpoint.pt', map_location='cpu', weights_only=True), strict=True)
            model.eval()
            with torch.no_grad():
                A, B = e.matrices(model)
                pred = e.rollout(model, x0, u, A, B)[:, :, :4].double()
                err = pred-y; err[..., 2] = wrap(err[..., 2])
                se = err.square().mean((1, 2))
                CB = B[:4, 0].double()
                rerr = (G-CB).square().sum(1)/G.square().sum(1)
            for lo, hi in BINS:
                m = (band >= lo) & (band < hi)
                out[f'{arm}/s{seed}/theta[{lo:.2f},{hi:.2f})'] = dict(
                    n=int(m.sum()), H15_wrapped=float(se[m].mean()), normalized_response_error=float(rerr[m].mean()))
    return out


def cartpole_rule(out):
    """Supporting rule: disease visible if hanging-band response error >= 1.5x upright band (all seeds)."""
    res = {}
    for arm in CART_ARMS:
        ok = []
        for seed in (42, 43, 44):
            up = out[f'{arm}/s{seed}/theta[0.00,0.52)']; down = out[f'{arm}/s{seed}/theta[1.57,3.14)']
            ok.append(down['normalized_response_error'] >= 1.5*up['normalized_response_error'])
        res[arm] = all(ok)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('jobs', 'data', 'train', 'eval'), required=True)
    p.add_argument('--arm', choices=tuple(ARMS)); p.add_argument('--a', type=float); p.add_argument('--seed', type=int)
    args = p.parse_args()
    torch.set_num_threads(1); c.OUT = OUT; OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'jobs':
        for arm, a, s in jobs():
            if not (OUT/'runs'/f'{arm}-a{a:g}-s{s}'/'result.json').exists():
                print(arm, a, s)
    elif args.stage == 'data':
        make_data()
    elif args.stage == 'train':
        run(args.arm, args.a, args.seed)
    else:
        rows = evaluate_rotation(); cart = evaluate_cartpole()
        c.save(OUT/'summary.json', {'rotation': rows, 'verdict': verdict(rows), 'cartpole': cart,
                                   'cartpole_rule': cartpole_rule(cart), 'script_sha256': sha256_file(__file__)})
        print(json.dumps({'verdict': verdict(rows), 'cartpole_rule': cartpole_rule(cart)}, indent=1))
