"""Build a report and scientific figure from the verified, complete P2 follow-up."""
import json
import statistics
from collections import defaultdict
import numpy as np
from research.diagnostics.p2_followup import OUT, ARMS, SEEDS, save


def summarize():
    verified = json.loads((OUT/'verification.json').read_text())
    assert verified['verdict'] == 'PASS'
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/result.json'))]
    assert len(rows) == 140
    groups = defaultdict(list)
    for r in rows:
        groups[(r['epochs'], r['arm'], r['a'])].append(r)
    summary = []
    for (epochs, arm, a), runs in sorted(groups.items()):
        assert sorted(r['seed'] for r in runs) == list(SEEDS)
        item = {'epochs': epochs, 'arm': arm, 'a': a}
        for key in ('response_normalized', 'best_constant_normalized_floor', 'learned_CB_sq_distance_to_best', 'one_step_mse', 'd_orient_p95'):
            values = [r['metrics'][key] for r in runs]
            item[key] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
        for key in ('15', '30', '50'):
            values = [r['metrics']['open_loop_mse'][key] for r in runs]
            item[f'open_loop_H{key}'] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
        item['sampled_h3_window_count'] = sum(r['metrics']['h3']['d_star_p95_below_rho'] for r in runs)
        item['final_lrs'] = [r['history'][-1]['lr'] for r in runs]
        summary.append(item)
    index = {(r['epochs'], r['arm'], r['a'], r['seed']): r for r in rows}
    ratios = {}
    for epochs, arms in ((25, ARMS), (100, ('baseline', 'general_noenc_epochmean'))):
        for arm in arms:
            rs = []
            monotone = []
            for seed in SEEDS:
                values = [index[epochs, arm, a, seed]['metrics']['response_normalized'] for a in (0., .5, 1., 2.)]
                rs.append(values[-1]/values[0])
                monotone.append(all(x<=y*(1+1e-6) for x,y in zip(values, values[1:])))
            ratios[f'{arm}-e{epochs}'] = {'per_seed': rs, 'minimum': min(rs), 'mean': statistics.mean(rs),
                'all_monotone': all(monotone), 'passes_response_growth_only': min(rs)>=1.5 and all(monotone)}
    e05 = json.loads((OUT/'e05_refresh.json').read_text())
    math = json.loads((OUT/'mathematical_controls.json').read_text())
    result = {'verification': verified, 'groups': summary, 'response_growth': ratios,
        'h2_readiness': 'NO-GO; input-map fitting remains unresolved',
        'h3_readiness': 'NO-GO; no new MPC/certificate gate is opened',
        'historical_gates_changed': False}
    save(OUT/'summary.json', result)
    lines = ['# P2 follow-up — 2026-11-16', '',
        'The planned sequential follow-up is complete: E05 refresh, 140 paired rotation runs, independent mathematical controls, checkpoint recomputation and source/pairing checks. Historical P2 evidence is preserved.', '',
        '## Protocol and scope', '',
        'Frozen historical P2 rotation datasets: 30 training trajectories, 10 evaluation trajectories, 600 fixed windows of length 15, batch 64, dt .02, latent dimension 6. Five seeds 42–46; strengths 0/.5/1/2. AdamW LR .001, weight decay 1e-5. Explicit shuffle RNG 700000+seed. CPU float32 training; float64 MSE reductions and output Gramians. The held-out response probe is shared across all strengths and seeds.', '',
        'At 25 epochs: baseline, epoch-mean scheduler, removal of BOTH encoder consistency terms, general invertible physical head P0 exp(M), and their combined diagnostic variant. At 100 epochs: baseline and combined, chosen before observing results. General heads preserve initial A/B and require the diagnostic loader. These are labeled deviations from upstream; no oracle gradients, H2/H3 method, new excitation distribution or sampler change is used. Evaluation uses final epochs; there is no test-based checkpoint/arm selection.', '',
        f"Total: {verified['completed_runs']} completed runs, {verified['updates']:,} optimizer updates; 40 long runs reproduce their first 25 epochs exactly. New ablations target the rotation family; Unicycle/Quadrotor outcomes remain historical and are not generalized from these runs.", '',
        '## E05 refresh', '',
        'Fixed explicit CartPole registration: previous discovery missed .yml configs and C02 checkpoint names. Pendulum seed 42 and CartPole C02 seeds 42/43/44 strict-load, match dataset hashes, repeat identically, and match a later recomputation. This verifies artifact identity and computation, not exact paper reproduction or the failed dataset-transition tolerance.', '',
        '| Checkpoint | rho_T | Numerically rank deficient | d_orient p95 |',
        '|---|---:|---|---:|']
    for name, r in e05['entries'].items():
        rho = 'undefined' if r['rho_T'] is None else f"{r['rho_T']:.7g}"
        lines.append(f"| {name} | {rho} | {r['rank_deficient']} | {r['d_orient_p95']:.6g} |")
    lines += ['', 'CartPole remains algebraically output-controllable according to the P1 independent checks. Numerical short-horizon Gramian rank deficiency does not contradict that algebraic property. No positive usable margin is invented.', '',
        '## Measured paired results', '',
        'Values are mean ± sample standard deviation across five seeds. Response is normalized continuous-time action-response error; H15 is unweighted physical-state rollout MSE from x0, without re-encoding.', '',
        '| Epochs | Arm | a | Response | H15 MSE |', '|---:|---|---:|---:|---:|']
    for r in summary:
        e, h = r['response_normalized'], r['open_loop_H15']
        lines.append(f"| {r['epochs']} | {r['arm']} | {r['a']} | {e['mean']:.6g} ± {e['sample_std']:.3g} | {h['mean']:.6g} ± {h['sample_std']:.3g} |")
    lines += ['', '## Why the original stress ratio was weak', '',
        'For the single-input rotation family, ||G(x)||=1. On the shared probe, with weights w=(u-u_ref)^2, normalized response error decomposes exactly as:', '',
        '`E_response = weighted Var(G) + ||CB - weighted mean(G)||²`', '',
        'The first term is the best-constant-map floor on this probe. The second is fitting error beyond that floor. This is a physical vector-field/action-response identity, not a bound on rollout MSE.', '',
        '| a | Best-constant floor | Ignore-input one-step MSE |', '|---:|---:|---:|']
    for name, r in math['rotation_controls'].items():
        lines.append(f"| {name[2:]} | {r['fixed_B_normalized_floor']:.6g} | {r['exact_drift_ignore_input_one_step_mse']:.8g} |")
    lines += ['',
        'At a=0 the floor is zero, yet the learned response error is near the ignore-input reference of 1. As stress increases the irreducible floor rises, while distance to the best constant map can fall; total error can therefore change little. A small growth ratio cannot isolate model-class limitation when the easy control has not been fitted.', '',
        'The old optimization gate used one-step MSE <= .05. An exact-drift predictor that ignores inputs scores about .000269 here, so that threshold does not establish action-response fidelity. The historical gate was evaluated as written; its diagnostic power is limited.', '',
        'There is also a verified head restriction: at a=0 the true map is [1,0]^T, but original SPD parameterization forces B_physical[1]>=.1. Its normalized response error is therefore at least .01. This proves an exact-representation mismatch, but does not explain an observed error near 1 by itself.', '',
        f"An analytic nonsymmetric head [[.2,1],[-1,0]] (det=1) reconstructs the a=0 A/B exactly, has a real exponential realization, and gives exact-ZOH held-out one-step MSE {math['general_head_oracle']['heldout_one_step_exact_ZOH_mse']:.3g}. This is a feasibility control, not a trained model or evidence of H2 superiority.", '',
        '## Gates and interpretation', '',
        '| Diagnostic arm | Minimum per-seed growth ratio | All seeds monotone | Response-growth criterion only |', '|---|---:|---|---|']
    for name, r in ratios.items():
        lines.append(f"| {name} | {r['minimum']:.6g} | {r['all_monotone']} | {r['passes_response_growth_only']} |")
    lines += ['',
        'Historical P2 and verification PASS remain preserved. H2 readiness stays NO-GO: these controls do not establish a well-fitted easy-system baseline or compare matched learning mechanisms. The analytical fixed-B floor is real, but it alone does not validate factorized learning.', '',
        'H3 stays NO-GO. CartPole E05 has no usable numerical short-horizon margin; rotation H3 diagnostics are sampled checks, not global certificates. This follow-up does not rerun MPC for diagnostic variants, so historical solver latency/success cannot be transferred to them.', '',
        'Next justified prerequisite: establish action-response fidelity on the constant-input-map positive control, and then preregister a stress/excitation protocol whose excess fitting error is small relative to the theoretical floor. A response-based optimization criterion would be a NEW protocol, not a retroactive change to the old gate. Do not start H2/H3 training yet.', '',
        '## Reproduce and verify', '', 'Run from repository root:', '', '```bash',
        'export MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1',
        '.venv/bin/python -m research.diagnostics.p2_followup --stage verify',
        '.venv/bin/python -m research.diagnostics.p2_followup --stage e05',
        '.venv/bin/python -m research.diagnostics.p2_followup --stage train',
        '.venv/bin/python -m research.diagnostics.p2_followup --stage recompute',
        '.venv/bin/python -m research.diagnostics.p2_followup_math',
        '.venv/bin/python -m research.diagnostics.verify_p2_followup',
        '.venv/bin/python -m research.diagnostics.summarize_p2_followup', '```', '',
        'Completed runs are reused by the training command; use a separate archived output path/code copy for a fresh training repetition. Recompute always reloads all 140 checkpoints and verifies data hashes. Imported source hashes are checked by verify_p2_followup. Gramians use float64 block exponentials and independent adaptive quadrature.', '',
        'Artifacts: summary.json, mathematical_controls.json, e05_refresh.json, source_snapshot.json, recompute.json, verification.json, historical_recompute.json, and one checkpoint/result per run. See results.png for the plotted comparison.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    plot(summary)
    print('Report and figure generated', flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for epochs, arm, label in ((25, 'baseline', 'baseline 25'), (100, 'baseline', 'baseline 100'),
                              (25, 'general_noenc_epochmean', 'combined 25'), (100, 'general_noenc_epochmean', 'combined 100')):
        group = sorted([r for r in rows if r['epochs']==epochs and r['arm']==arm], key=lambda r:r['a'])
        x = [r['a'] for r in group]
        axes[0, 0].errorbar(x, [r['response_normalized']['mean'] for r in group],
                          yerr=[r['response_normalized']['sample_std'] for r in group], label=label, marker='o', capsize=3)
        axes[1, 0].plot(x, [r['open_loop_H15']['mean'] for r in group], marker='o', label=label)
    baseline = sorted([r for r in rows if r['epochs']==100 and r['arm']=='baseline'], key=lambda r:r['a'])
    x = [r['a'] for r in baseline]
    floor = [r['best_constant_normalized_floor']['mean'] for r in baseline]
    gap = [r['learned_CB_sq_distance_to_best']['mean'] for r in baseline]
    axes[0, 0].plot(x, floor, 'k--', label='best constant floor')
    axes[0, 0].axhline(1, color='gray', linestyle=':', label='ignore input')
    axes[0, 0].set(ylabel='Normalized response error', title='Response fitting remains weak')
    axes[0, 1].bar(np.arange(4), floor, label='fixed-map floor')
    axes[0, 1].bar(np.arange(4), gap, bottom=floor, label='excess map-fitting error')
    axes[0, 1].set(xticks=np.arange(4), xticklabels=x, ylabel='Normalized response error', title='Baseline 100: opposing error components')
    axes[1, 0].set(ylabel='Physical state MSE', title='Open-loop H15; no re-encoding')
    for a in (0., .5, 1., 2.):
        group = [r for r in rows if r['epochs']==25 and r['a']==a]
        axes[1, 1].plot(np.arange(len(group)), [r['response_normalized']['mean'] for r in group], marker='o', label=f'a={a}')
    labels = [r['arm'] for r in rows if r['epochs']==25 and r['a']==0]
    axes[1, 1].set(xticks=np.arange(len(labels)), xticklabels=[{'baseline':'baseline', 'epochmean':'epoch mean', 'general':'general head', 'general_noenc_epochmean':'combined', 'noenc':'no encoder'}[x] for x in labels], ylabel='Normalized response error', title='Isolated ablations at 25 epochs')
    axes[1, 1].tick_params(axis='x', labelrotation=20)
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]): ax.set_xlabel('Rotation strength a')
    fig.suptitle('P2 follow-up: five paired seeds; diagnostic variants')
    fig.savefig(OUT/'results.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    summarize()
