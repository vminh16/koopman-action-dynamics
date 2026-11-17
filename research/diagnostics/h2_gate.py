"""H2 gate follow-up (preregistered in TRACKING.md, 2026-11-17).

C1 unicycle v2 and C2 planar quadrotor v2 with equal tuning budgets for fixed-B and G.5
(LR x updates chosen on dev seeds by validation H15 only), confirmed on new seeds; C2 scores
only the thrust-direction rows (y_ddot, z_ddot). C3 CartPole G.5 whose gain network sees
cos(theta) instead of the unwrapped angle (labeled diagnostic variant).
Reuses h2_g5_tests (data v2, general head, losses, metrics) without editing it.
"""
import argparse
import json
import math
import statistics

import torch

from research.baseline.common import ROOT, sha256_file
from research.diagnostics import h2_g5_tests as t
from research.diagnostics import p2_config_check as c

OUT = ROOT/'outputs/diagnosis/h2_gate_2026-11-17'
DEV, CONFIRM = (42, 43, 44), (112, 113, 114, 115, 116)
LRS, BUDGETS = (.003, .01), (3000, 12000)
ROWS = {'unicycle': None, 'planar_quadrotor': (3, 4)}


def use(budget):
    """Point h2_g5_tests at a per-budget run directory (data regenerate deterministically)."""
    t.OUT, t.UPDATES = OUT/f'u{budget}', budget


def train(system, arm, lr, budget, seed):
    use(budget)
    return t.train(system, arm, lr, seed)


def select():
    out = {}
    for system in ROWS:
        for arm in t.ARMS:
            ranks = []
            for budget in BUDGETS:
                for lr in LRS:
                    use(budget)
                    rs = [json.loads((t.OUT/'runs'/f'{system}-{arm}-lr{lr:g}-s{s}'/'result.json').read_text()) for s in DEV]
                    ok = [r['val_H15'] for r in rs if r['status'] == 'complete']
                    ranks.append(((3-len(ok), statistics.mean(ok) if ok else math.inf, budget, lr), dict(lr=lr, budget=budget)))
            out[f'{system}/{arm}'] = min(ranks, key=lambda r: r[0])[1]
    c.save(OUT/'selected.json', out)
    return out


def metrics(system, model, seed, rows):
    """h2_g5_tests.scaled_metrics restricted to the given state rows (None = all)."""
    test, test_id = t.generate(system, seed, 'test'); train_data, _ = t.generate(system, seed, 'train')
    b = t.bench(system); X, Tr = t.states(test), t.states(train_data)
    Ut = torch.cat([r['u_seq'] for r in train_data])
    s = b['dynamics'](0., Tr, Ut).double().std(0).clamp_min(1e-8)
    idx = list(rows) if rows else list(range(X.shape[1]))
    with torch.no_grad():
        G = (b['analytic_Gx'](X).double()/s[None, :, None])[:, idx]
        R = (t.response(model, X).double()/s[None, :, None])[:, idx]
        norm = float(G.square().sum((1, 2)).mean())
        sfloor = sum(float(G[:, :, j].square().sum(1).mean()-torch.linalg.eigvalsh(G[:, :, j].T@G[:, :, j]/len(G))[-1])
                     for j in range(G.shape[2]))/norm
        sc = Tr.double().std(0).clamp_min(1e-8)
        return dict(test_data=test_id, rows=idx, response=float((G-R).square().sum((1, 2)).mean())/norm,
                    fixed_B_floor=float((G-G.mean(0)).square().sum((1, 2)).mean())/norm, scalar_gain_floor=sfloor,
                    H15=t.h15(model, test),
                    coverage_p95=float(torch.quantile(torch.cdist(X.double()/sc, Tr.double()/sc).min(1).values, .95)))


# ---- C3: CartPole G.5 with cos(theta) features for the gain network ----
class CosFeature(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, u):
        return self.inner(torch.cat([x[:, :2], torch.cos(x[:, 2:3]), x[:, 3:]], 1), u)


def cart_dir(system, fraction, arm, seed):
    return OUT/'cartpole'/f'{system}-f{fraction:g}-g5cos-{arm}-s{seed}'


def cart_model(system, seed, operator, fraction=.05):
    model = ORIGINAL_CART_MODEL(system, seed, operator, fraction)
    model.model.u_encoder = CosFeature(model.model.u_encoder)
    return model


ORIGINAL_CART_MODEL = t.cart_model


def with_cos(fn, *args):
    saved = (t.cart_dir, t.cart_model)
    try:
        t.cart_dir, t.cart_model = cart_dir, cart_model
        return fn(*args)
    finally:
        t.cart_dir, t.cart_model = saved


def rules(conf, cart):
    R = {}
    for key, system in (('C1_unicycle', 'unicycle'), ('C2_quadrotor_thrust_rows', 'planar_quadrotor')):
        base = [conf[f'{system}/base/s{s}'] for s in CONFIRM]; g = [conf[f'{system}/g5/s{s}'] for s in CONFIRM]
        R[key] = dict(
            fixed_B_disease=sum(r['response'] > .05 and r['response']-r['fixed_B_floor'] <= .25*r['fixed_B_floor'] for r in base) >= 4,
            G5_does_not_fix=sum(r['response'] > .05 for r in g) >= 4 and
            statistics.mean(r['response'] for r in g) >= .5*statistics.mean(r['response'] for r in base))
    ref = json.loads((ROOT/'outputs/diagnosis/h2_diagnosis_2026-11-17/summary.json').read_text())['cartpole']
    R['C3_cos_G5_fixes_cartpole'] = all(
        cart[f'g5_dense_full/s{s}/theta{b}']['normalized_response_error'] <= .5*ref[f'dense_full/s{s}/theta{b}']['normalized_response_error']
        for s in (42, 43, 44) for b in ('[0.00,0.52)', '[1.57,3.14)'))
    R['coverage_ok'] = max(r['coverage_p95'] for r in conf.values()) <= 3.
    second = any(R[k]['fixed_B_disease'] and R[k]['G5_does_not_fix'] for k in ('C1_unicycle', 'C2_quadrotor_thrust_rows'))
    # T1 (rotation residual beyond G.5) passed in h2_g5_tests; required here as a recorded fact.
    t1 = json.loads((ROOT/'outputs/diagnosis/h2_g5_tests_2026-11-17/summary.json').read_text())['rules']['T1_rotation_residual']
    R['H2_readiness'] = 'GO' if t1 and second and R['coverage_ok'] else 'NOT_YET'
    return R


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('data', 'jobs-dev', 'select', 'jobs-confirm', 'train', 'eval'), required=True)
    p.add_argument('--job', nargs=5)
    a = p.parse_args()
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'data':  # sequential, so parallel trainers only read
        for budget in BUDGETS:
            use(budget)
            for system in ROWS:
                for s in DEV+CONFIRM:
                    for split in ('train', 'validation', 'test'):
                        t.generate(system, s, split)
    elif a.stage == 'jobs-dev':
        for s in (42, 43, 44):
            print('cartpole g5cos 0 0', s)
        for system in ROWS:
            for arm in t.ARMS:
                for budget in BUDGETS:
                    for lr in LRS:
                        for s in DEV:
                            print(system, arm, lr, budget, s)
    elif a.stage == 'select':
        print(select())
    elif a.stage == 'jobs-confirm':
        sel = json.loads((OUT/'selected.json').read_text())
        for system in ROWS:
            for arm in t.ARMS:
                for s in CONFIRM:
                    print(system, arm, sel[f'{system}/{arm}']['lr'], sel[f'{system}/{arm}']['budget'], s)
    elif a.stage == 'train':
        system, arm, lr, budget, s = a.job
        if system == 'cartpole':
            with_cos(t.cart_train, int(s))
        else:
            train(system, arm, float(lr), int(budget), int(s))
    else:
        sel = json.loads((OUT/'selected.json').read_text()); conf = {}
        for system, rows in ROWS.items():
            for arm in t.ARMS:
                for s in CONFIRM:
                    use(sel[f'{system}/{arm}']['budget'])
                    model = t.load_run(system, arm, sel[f'{system}/{arm}']['lr'], s)
                    conf[f'{system}/{arm}/s{s}'] = metrics(system, model, s, rows)
        cart = with_cos(t.cart_eval)
        R = rules(conf, cart)
        c.save(OUT/'summary.json', dict(confirm=conf, cartpole=cart, selected=sel, rules=R, script_sha256=sha256_file(__file__)))
        print(json.dumps(R, indent=1))
