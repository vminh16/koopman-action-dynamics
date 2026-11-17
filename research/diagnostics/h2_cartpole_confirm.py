"""C4: confirmatory CartPole test on fresh seeds (preregistered in TRACKING.md, 2026-11-17).

Fixed-B (`dense_full`, P1 harness) and cos-G.5 (h2_gate variant) trained on CartPole 5% with
new locked configs seeds 45-47; hanging-band response error compared with each class's
normalized floor (fixed-B: best constant map; G.5: best fixed direction x per-state gain).
"""
import argparse
import json
import math
import statistics

import torch

from research.baseline.common import ROOT, sha256_file
from research.diagnostics import h2_diagnosis as h
from research.diagnostics import h2_gate as g
from research.diagnostics import h2_g5_tests as t
from research.diagnostics import p1_upstream_faithful as f
from research.diagnostics import p2_config_check as c

OUT = ROOT/'outputs/diagnosis/h2_cartpole_confirm_2026-11-17'
SEEDS = (45, 46, 47)


def fixed_dir(system, fraction, arm, seed):
    return OUT/'fixedB'/f'{system}-f{fraction:g}-{arm}-s{seed}'


def train(arm, seed):
    saved = (f.SEEDS, f.run_dir, g.OUT)
    try:
        f.SEEDS = SEEDS  # validation = 200 trajectories outside the new seeds' subsets
        if arm == 'fixedB':
            f.run_dir = fixed_dir
            f.train('cartpole', .05, 'dense_full', seed, 100)
        else:
            g.OUT = OUT
            g.with_cos(t.cart_train, seed)
    finally:
        f.SEEDS, f.run_dir, g.OUT = saved


def load(arm, seed):
    saved = g.OUT
    try:
        g.OUT = OUT
        d = fixed_dir('cartpole', .05, 'dense_full', seed) if arm == 'fixedB' else g.cart_dir('cartpole', .05, 'dense_full', seed)
        res = json.loads((d/'result.json').read_text())
        assert (res['status'], res['arm'], res['operator'], res['fraction'], res['seed']) == ('complete', 'dense_full', 'dense', .05, seed), d
        assert sha256_file(d/'checkpoint.pt') == res['checkpoint_sha256']
        model = f.make_model('cartpole', seed, 'dense', .05) if arm == 'fixedB' else g.cart_model('cartpole', seed, 'dense')
        model.load_state_dict(torch.load(d/'checkpoint.pt', map_location='cpu', weights_only=True), strict=True)
        return model.eval()
    finally:
        g.OUT = saved


def evaluate():
    from dynamics import CartPoleDynamics
    from research.diagnostics.p1_audit import experiment as e
    raw = torch.load(ROOT/'data/cartpole_test.pt', map_location='cpu', weights_only=True)
    X = torch.stack([torch.cat([r['x0'].reshape(1, 4), r['x_seq'].reshape(250, 4)]) for r in raw])
    gen = torch.Generator().manual_seed(20261117)  # same windows as D3/T2/C3
    st = torch.randint(0, 250-15, (len(raw), 10), generator=gen)
    x0 = X[torch.arange(len(raw))[:, None].expand_as(st).reshape(-1), st.reshape(-1)]
    G = h.cartpole_true_response(CartPoleDynamics(), x0.double())
    w = 1/G.square().sum(1)
    const = (G*w[:, None]).sum(0)/w.sum()
    fixed_floor = (G-const).square().sum(1)*w
    Gn = G*w.sqrt()[:, None]
    d = torch.linalg.eigh(Gn.T@Gn/len(Gn))[1][:, -1]
    scalar_floor = 1-(Gn@d)**2
    band = ((x0[:, 2].double()+math.pi) % (2*math.pi)-math.pi).abs()
    out = {'floors': {}, 'runs': {}}
    for lo, hi in h.BINS:
        m = (band >= lo) & (band < hi)
        out['floors'][f'[{lo:.2f},{hi:.2f})'] = dict(fixed_B=float(fixed_floor[m].mean()), scalar_gain=float(scalar_floor[m].mean()))
    for arm in ('fixedB', 'cosG5'):
        for seed in SEEDS:
            model = load(arm, seed)
            with torch.no_grad():
                A, B = e.matrices(model)
                z = model.model.encoder(x0); one = torch.ones(len(x0), 1)
                R = ((t.eff(model, z, one)-t.eff(model, z, -one))@B.T/2)[:, :4].double()
                rerr = (G-R).square().sum(1)*w
            for lo, hi in h.BINS:
                m = (band >= lo) & (band < hi)
                out['runs'][f'{arm}/s{seed}/[{lo:.2f},{hi:.2f})'] = float(rerr[m].mean())
    return out


def rules(out):
    hb = '[1.57,3.14)'; fl = out['floors'][hb]
    fb = [out['runs'][f'fixedB/s{s}/{hb}'] for s in SEEDS]; g5 = [out['runs'][f'cosG5/s{s}/{hb}'] for s in SEEDS]
    R = dict(fixed_B_at_floor=all(.05 < v <= 1.25*fl['fixed_B'] for v in fb),
             G5_at_floor_not_fixing=all(.05 < v <= 1.25*fl['scalar_gain'] for v in g5) and
             statistics.mean(g5) >= .5*statistics.mean(fb))
    R['C4'] = 'PASS' if R['fixed_B_at_floor'] and R['G5_at_floor_not_fixing'] else 'FAIL'
    t1 = json.loads((ROOT/'outputs/diagnosis/h2_g5_tests_2026-11-17/summary.json').read_text())['rules']['T1_rotation_residual']
    R['H2_readiness'] = 'GO' if t1 and R['C4'] == 'PASS' else 'NOT_YET'
    return R


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('jobs', 'train', 'eval'), required=True)
    p.add_argument('--arm'); p.add_argument('--seed', type=int)
    a = p.parse_args(); torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'jobs':
        for arm in ('fixedB', 'cosG5'):
            for s in SEEDS:
                print(arm, s)
    elif a.stage == 'train':
        train(a.arm, a.seed)
    else:
        out = evaluate(); R = rules(out)
        c.save(OUT/'summary.json', {**out, 'rules': R, 'script_sha256': sha256_file(__file__)})
        print(json.dumps({'floors': out['floors'], 'runs': out['runs'], 'rules': R}, indent=1))
