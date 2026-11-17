"""H2 tests against the author's G.5 input encoder (preregistered in TRACKING.md, 2026-11-17).

T1 rotation residual (a=2), T2 CartPole sign flip (P1 harness + G.5), T3 unicycle,
T4 planar quadrotor. Fixed-B general head vs G.5 (use_modulated_u); oracle G(x) is used
for evaluation only. Model selection (T3/T4 LR) uses validation H15 MSE only.
"""
import argparse
import json
import math
import statistics

import torch

from research.baseline.common import ROOT, build_module, sha256_file
from research.baseline.data_generator import rk4_step
from research.benchmarks import load_benchmark
from research.diagnostics import h2_diagnosis as h
from research.diagnostics import p2_config_check as c
from research.diagnostics import run_p2_diagnostics as p2
from training.loss import prediction_loss, encoder_loss_from_true, encoder_loss_from_pred

OUT = ROOT/'outputs/diagnosis/h2_g5_tests_2026-11-17'
DEV, CONFIRM = (42, 43, 44), (102, 103, 104, 105, 106)
LRS = (.001, .01)
ARMS = ('base', 'g5')
# system -> benchmark kwargs, latent N (= n+4 as rotation), input sampler, hold steps
SYSTEMS = {
    'unicycle': dict(kw=dict(omega_bounds=(-3., 3.)), latent=7, hold=10),
    'planar_quadrotor': dict(kw={}, latent=10, hold=5),
}
T, WIN, UPDATES = 50, 15, 3000


# ---------------- data ----------------
def bench(system):
    return load_benchmark(system, **SYSTEMS[system]['kw'])


def sample_u(system, sys, g):
    hold = SYSTEMS[system]['hold']
    k = T//hold
    if system == 'unicycle':  # piecewise-constant (v, omega) so heading changes inside a window
        u = sys.umin+torch.rand(k, 2, generator=g)*(sys.umax-sys.umin)
    else:  # hover-centred thrusts: mg/2 +- .5 N per rotor, so the vehicle tilts without tumbling
        u = sys.m*sys.g/2+(torch.rand(k, 2, generator=g)-.5)
    return u.repeat_interleave(hold, 0)


def generate(system, seed, split):
    path = OUT/'data'/f'{system}-{split}-s{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=True), {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}
    count = {'train': 32, 'validation': 8, 'test': 16}[split]
    b = bench(system); sys = b['instance']
    torch.manual_seed({'train': 10000, 'validation': 20000, 'test': 30000}[split]+seed)
    g = torch.Generator().manual_seed(40000+{'train': 0, 'validation': 1, 'test': 2}[split]*1000+seed)
    data = []
    for _ in range(count):
        x = sys.sample_initial_state(1)[0]
        x0, u, xs = x.clone(), sample_u(system, sys, g), []
        for t in range(T):
            x = rk4_step(b['dynamics'], 0., x[None], u[t:t+1], sys.dt)[0]
            xs.append(x)
        data.append({'x0': x0[None], 'u_seq': u, 'x_seq': torch.stack(xs)})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data, {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}


# ---------------- model (general head for any n, optional G.5) ----------------
def make_model(system, seed, g5):
    torch.manual_seed(seed)
    sys = bench(system)['instance']
    model = p2.build_official_module(sys, SYSTEMS[system]['latent'], {**p2.P2_BASELINE_CONFIG, 'use_modulated_u': g5})
    op, n = model.model.operator.core, sys.Nstates
    model.register_buffer('diagnostic_head_base', op.build_P()[:n, :n].detach().clone())
    with torch.no_grad():
        op.H_head.zero_()
    return model


def matrices(model):
    """Same geometry as p2_followup.matrices, for state dimension n."""
    op, n = model.model.operator.core, model.state_dim
    head = model.diagnostic_head_base@torch.matrix_exp(op.H_head)
    P = torch.block_diag(head, op.build_P()[n:, n:])
    return P@op.build_Ac()@torch.linalg.inv(P), P@op.build_Bc()


def eff(model, z, u):
    m = model.model
    return m.u_encoder(z[:, :model.state_dim], u) if m.use_modulated_u else u


def rollout(model, x, u, AB=None):
    A, B = matrices(model) if AB is None else AB
    z = model.model.encoder(x); out = []
    for a in u.unbind(1):
        z = z+model.dt*(z@A.T+eff(model, z, a)@B.T); out.append(z)
    return torch.stack(out, 1)


def loss(model, batch):
    """p2_config_check.components generalised from n=2 (same terms, weights, coupled Euler W)."""
    x, u, y = batch; n = model.state_dim
    A, B = matrices(model)
    z = rollout(model, x, u, (A, B))
    total = prediction_loss(z[:, :, :n], y)+encoder_loss_from_true(y, z[:, :, n:], model.model.encoder) \
        + .5*encoder_loss_from_pred(z[:, :, :n], z[:, :, n:], model.model.encoder)
    Phi = torch.eye(A.shape[0]); W = torch.zeros(n, n)
    for _ in range(15):
        term = (Phi@B)[:n]; W = W+model.dt*(term@term.T); Phi = Phi+model.dt*(A@Phi)
    return total+.005*torch.nn.functional.softplus(.03-torch.linalg.eigvalsh(W).min())


def response(model, X):
    """Physical response r(x) = d xdot / du, columns per input (symmetric +-e_j)."""
    A, B = matrices(model); z = model.model.encoder(X); cols = []
    for j in range(B.shape[1]):
        e = torch.zeros(len(X), B.shape[1]); e[:, j] = 1
        cols.append(((eff(model, z, e)-eff(model, z, -e))@B.T/2)[:, :model.state_dim])
    return torch.stack(cols, -1)


def h15(model, data):
    x = torch.cat([r['x0'] for r in data]); u = torch.stack([r['u_seq'][:WIN] for r in data])
    y = torch.stack([r['x_seq'][:WIN] for r in data]).double()
    return float((rollout(model, x, u)[:, :, :model.state_dim].double()-y).square().mean())


def train(system, arm, lr, seed):
    d = OUT/'runs'/f'{system}-{arm}-lr{lr:g}-s{seed}'
    if (d/'result.json').exists():
        return json.loads((d/'result.json').read_text())
    torch.set_num_threads(1)
    data, tid = generate(system, seed, 'train'); val, vid = generate(system, seed, 'validation')
    batch, samples = c.tensors(data, seed)
    assert len(batch[0]) == 640
    model = make_model(system, seed, arm == 'g5')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = torch.Generator().manual_seed(700000+seed); hist = []; status = 'complete'; steps = 0
    for epoch in range(1, UPDATES*64//640+1):
        for ids in torch.randperm(640, generator=rng).split(64):
            opt.zero_grad(set_to_none=True)
            total = loss(model, tuple(t[ids] for t in batch))
            if not torch.isfinite(total):
                status = 'failed_nonfinite'; break
            total.backward(); opt.step(); steps += 1
        if status != 'complete':
            break
        if epoch % 100 == 0:
            with torch.no_grad():
                hist.append(dict(epoch=epoch, val_H15=h15(model, val)))
            print(d.name, hist[-1], flush=True)
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(name=d.name, system=system, arm=arm, lr=lr, seed=seed, status=status, updates=steps,
               train=tid, validation=vid, window_samples=samples, history=hist, script_sha256=sha256_file(__file__))
    if status == 'complete':
        torch.save(model.state_dict(), d/'checkpoint.pt'); rec['checkpoint_sha256'] = sha256_file(d/'checkpoint.pt')
        rec['val_H15'] = hist[-1]['val_H15']
    (d/'result.json').write_text(json.dumps(rec, indent=2))
    return rec


def select():
    """Per system x arm: fewest dev failures, then lowest mean validation H15; then lower LR."""
    out = {}
    for system in SYSTEMS:
        for arm in ARMS:
            ranks = []
            for lr in LRS:
                rs = [json.loads((OUT/'runs'/f'{system}-{arm}-lr{lr:g}-s{s}'/'result.json').read_text()) for s in DEV]
                ok = [r['val_H15'] for r in rs if r['status'] == 'complete']
                ranks.append(((3-len(ok), statistics.mean(ok) if ok else math.inf, lr), lr))
            out[f'{system}/{arm}'] = min(ranks)[1]
    c.save(OUT/'selected_lr.json', out)
    return out


def load_run(system, arm, lr, seed):
    d = OUT/'runs'/f'{system}-{arm}-lr{lr:g}-s{seed}'
    rec = json.loads((d/'result.json').read_text())
    assert (rec['system'], rec['arm'], rec['lr'], rec['seed'], rec['status']) == (system, arm, lr, seed, 'complete'), d
    assert sha256_file(d/'checkpoint.pt') == rec['checkpoint_sha256']
    model = make_model(system, seed, arm == 'g5')
    model.load_state_dict(torch.load(d/'checkpoint.pt', weights_only=True), strict=True)
    return model.eval()


def states(data):
    return torch.cat([torch.cat([r['x0'], r['x_seq'][:-1]]) for r in data])


def scaled_metrics(system, model, seed):
    """Response errors in rows scaled by train std of each state derivative (quadrotor rows
    differ by ~25x in physical units); normalised by E||G_s||^2."""
    test, test_id = generate(system, seed, 'test'); train_data, _ = generate(system, seed, 'train')
    b = bench(system)
    X, Tr = states(test), states(train_data)
    Ut = torch.cat([r['u_seq'] for r in train_data])
    s = b['dynamics'](0., Tr, Ut).double().std(0).clamp_min(1e-8)
    with torch.no_grad():
        G = b['analytic_Gx'](X).double()/s[None, :, None]
        R = response(model, X).double()/s[None, :, None]
        norm = float(G.square().sum((1, 2)).mean())
        resp = float((G-R).square().sum((1, 2)).mean())/norm
        floor = float((G-G.mean(0)).square().sum((1, 2)).mean())/norm
        sfloor = sum(float(G[:, :, j].square().sum(1).mean()-torch.linalg.eigvalsh(G[:, :, j].T@G[:, :, j]/len(G))[-1])
                     for j in range(G.shape[2]))/norm
        sc = Tr.double().std(0).clamp_min(1e-8)
        cover = float(torch.quantile(torch.cdist(X.double()/sc, Tr.double()/sc).min(1).values, .95))
        return dict(test_data=test_id, response=resp, fixed_B_floor=floor, scalar_gain_floor=sfloor,
                    H15=h15(model, test), coverage_p95=cover)


# ---------------- T1 rotation residual ----------------
def rotation_jobs():
    return [('rotation', arm, 2., s) for arm in ARMS for s in CONFIRM]


def rotation_eval():
    rows = {}
    for _, arm, a, s in rotation_jobs():
        d = c.OUT/'runs'/f'{arm}-a{a:g}-s{s}'
        rec = json.loads((d/'result.json').read_text())
        assert (rec['name'], rec['a'], rec['seed'], rec['status']) == (d.name, a, s, 'complete'), d
        ck = rec['checkpoints'][str(rec['epochs_completed'])]
        assert sha256_file(ROOT/ck['path']) == ck['sha256']
        model = h.make_model(s, True, 6, arm == 'g5')
        model.load_state_dict(torch.load(ROOT/ck['path'], weights_only=True), strict=True)
        m = h.rotation_metrics(model.eval(), a, s, 32)
        test, _ = c.generate(s, 'test', False, a)
        X = torch.cat([torch.cat([r['x0'], r['x_seq'][:-1]]) for r in test]).double()
        G = torch.stack([torch.cos(a*X[:, 0]), torch.sin(a*X[:, 0])], 1)
        m['scalar_gain_floor'] = 1-float(torch.linalg.eigvalsh(G.T@G/len(G))[-1])
        rows[f'{arm}/s{s}'] = m
    return rows


# ---------------- T2 CartPole + G.5 (P1 harness) ----------------
def cart_dir(system, fraction, arm, seed):
    return OUT/'cartpole'/f'{system}-f{fraction:g}-g5-{arm}-s{seed}'


def cart_model(system, seed, operator, fraction=.05):
    from research.diagnostics import p1_upstream_faithful as f
    torch.manual_seed(seed)
    model = build_module({**f.config(system, seed, fraction), 'use_modulated_u': True})
    f.use_dense_p(model, seed)
    return model


def cart_rollout(model, x0, u, A, B):
    z = model.model.encoder(x0); out = []
    for t in range(u.shape[1]):
        z = z+model.dt*(z@A.T+eff(model, z, u[:, t])@B.T); out.append(z)
    return torch.stack(out, 1)


def cart_train(seed):
    from research.diagnostics import p1_upstream_faithful as f
    from research.diagnostics.p1_audit import experiment as e
    saved = (f.run_dir, f.make_model, e.rollout)
    try:
        f.run_dir, f.make_model, e.rollout = cart_dir, cart_model, cart_rollout
        f.train('cartpole', .05, 'dense_full', seed, 100)
    finally:
        f.run_dir, f.make_model, e.rollout = saved


def cart_eval():
    from dynamics import CartPoleDynamics
    from research.diagnostics.p1_audit import experiment as e
    dyn = CartPoleDynamics()
    raw = torch.load(ROOT/'data/cartpole_test.pt', map_location='cpu', weights_only=True)
    X = torch.stack([torch.cat([r['x0'].reshape(1, 4), r['x_seq'].reshape(250, 4)]) for r in raw])
    U = torch.stack([r['u_seq'] for r in raw])
    g = torch.Generator().manual_seed(20261117)  # same windows as h2_diagnosis D3
    starts = torch.randint(0, 250-15, (len(raw), 10), generator=g)
    ti = torch.arange(len(raw))[:, None].expand_as(starts).reshape(-1); st = starts.reshape(-1)
    x0 = X[ti, st]; u = torch.stack([U[ti, st+k] for k in range(15)], 1)
    y = torch.stack([X[ti, st+1+k] for k in range(15)], 1).double()
    wrap = lambda t: (t+math.pi) % (2*math.pi)-math.pi
    band = wrap(x0[:, 2].double()).abs()
    G = h.cartpole_true_response(dyn, x0.double())
    out = {}
    for seed in (42, 43, 44):
        d = cart_dir('cartpole', .05, 'dense_full', seed)
        res = json.loads((d/'result.json').read_text())
        assert (res['status'], res['arm'], res['operator'], res['fraction'], res['seed']) == ('complete', 'dense_full', 'dense', .05, seed), d
        assert sha256_file(d/'checkpoint.pt') == res['checkpoint_sha256']
        model = cart_model('cartpole', seed, 'dense')
        model.load_state_dict(torch.load(d/'checkpoint.pt', map_location='cpu', weights_only=True), strict=True)
        model.eval()
        with torch.no_grad():
            A, B = e.matrices(model)
            err = cart_rollout(model, x0, u, A, B)[:, :, :4].double()-y; err[..., 2] = wrap(err[..., 2])
            se = err.square().mean((1, 2))
            z = model.model.encoder(x0); one = torch.ones(len(x0), 1)
            R = ((eff(model, z, one)-eff(model, z, -one))@B.T/2)[:, :4].double()
            rerr = (G-R).square().sum(1)/G.square().sum(1)
        for lo, hi in h.BINS:
            m = (band >= lo) & (band < hi)
            out[f'g5_dense_full/s{seed}/theta[{lo:.2f},{hi:.2f})'] = dict(
                n=int(m.sum()), H15_wrapped=float(se[m].mean()), normalized_response_error=float(rerr[m].mean()))
    return out


# ---------------- rules (preregistered) ----------------
def rules(rot, cart, conf):
    R = {}
    g5 = [r for k, r in rot.items() if k.startswith('g5/')]
    R['T1_rotation_residual'] = (sum(r['response'] > .05 for r in g5) >= 4 and
                                 sum(r['response']-r['scalar_gain_floor'] <= .25*r['scalar_gain_floor'] for r in g5) >= 4)
    ref = json.loads((h.OUT/'summary.json').read_text())['cartpole']
    up, down = '[0.00,0.52)', '[1.57,3.14)'
    R['T2_cartpole_G5_fixes_sign_flip'] = all(
        cart[f'g5_dense_full/s{s}/theta{b}']['normalized_response_error'] <= .5*ref[f'dense_full/s{s}/theta{b}']['normalized_response_error']
        for s in (42, 43, 44) for b in (up, down))
    for key, system in (('T3_unicycle', 'unicycle'), ('T4_quadrotor', 'planar_quadrotor')):
        base = [conf[f'{system}/base/s{s}'] for s in CONFIRM]; g = [conf[f'{system}/g5/s{s}'] for s in CONFIRM]
        disease = sum(r['response'] > .05 and r['response']-r['fixed_B_floor'] <= .25*r['fixed_B_floor'] for r in base) >= 4
        not_fixed = sum(r['response'] > .05 for r in g) >= 4 and \
            statistics.mean(r['response'] for r in g) >= .5*statistics.mean(r['response'] for r in base)
        R[key] = dict(fixed_B_disease=disease, G5_does_not_fix=not_fixed)
    allcov = [r['coverage_p95'] for r in list(rot.values())+list(conf.values())]
    R['coverage_ok'] = max(allcov) <= 3.
    survives = R['T1_rotation_residual'] and any(R[k]['fixed_B_disease'] and R[k]['G5_does_not_fix'] for k in ('T3_unicycle', 'T4_quadrotor'))
    R['H2_diagnosis'] = 'DISEASE_BEYOND_G5_CONFIRMED' if survives and R['coverage_ok'] else 'NOT_CONFIRMED_BEYOND_G5'
    return R


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('data', 'jobs-dev', 'select', 'jobs-confirm', 'train', 'eval'), required=True)
    p.add_argument('--job', nargs=4)
    a = p.parse_args()
    torch.set_num_threads(1); c.OUT = OUT/'rotation'; OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'data':
        for system in SYSTEMS:
            for s in DEV+CONFIRM:
                for split in ('train', 'validation', 'test'):
                    generate(system, s, split)
        for _, arm, al, s in rotation_jobs():
            for split in ('train', 'validation', 'test'):
                c.generate(s, split, False, al, **({'count': 32} if split == 'train' else {}))
    elif a.stage == 'jobs-dev':  # rotation T1 + CartPole T2 + T3/T4 dev tuning
        for _, arm, al, s in rotation_jobs():
            print('rotation', arm, al, s)
        for s in (42, 43, 44):
            print('cartpole', 'g5', 0, s)
        for system in SYSTEMS:
            for arm in ARMS:
                for lr in LRS:
                    for s in DEV:
                        print(system, arm, lr, s)
    elif a.stage == 'select':
        print(select())
    elif a.stage == 'jobs-confirm':
        sel = json.loads((OUT/'selected_lr.json').read_text())
        for system in SYSTEMS:
            for arm in ARMS:
                for s in CONFIRM:
                    print(system, arm, sel[f'{system}/{arm}'], s)
    elif a.stage == 'train':
        system, arm, x, s = a.job; s = int(s)
        if system == 'rotation':
            h.run(arm, float(x), s)
        elif system == 'cartpole':
            cart_train(s)
        else:
            train(system, arm, float(x), s)
    else:
        sel = json.loads((OUT/'selected_lr.json').read_text())
        conf = {f'{system}/{arm}/s{s}': scaled_metrics(system, load_run(system, arm, sel[f'{system}/{arm}'], s), s)
                for system in SYSTEMS for arm in ARMS for s in CONFIRM}
        rot, cart = rotation_eval(), cart_eval()
        R = rules(rot, cart, conf)
        c.save(OUT/'summary.json', dict(rotation=rot, cartpole=cart, confirm=conf, selected_lr=sel,
                                       rules=R, script_sha256=sha256_file(__file__)))
        print(json.dumps(R, indent=1))
