"""Verify all experiment artifacts and render the causal audit report."""
import json
import math
from pathlib import Path
import statistics
import subprocess

from research.baseline.common import ROOT, sha256_file

out=ROOT/'outputs/reproduction/p1_root_cause_2026-11-15'
records=[]
for path in sorted(out.glob('*/manifest.json')):
    result=path.parent/'result.json'
    assert result.exists(), f'Incomplete run: {path.parent.name}'
    r=json.loads(result.read_text());assert r['status']=='complete', r['run_id']
    assert r['epochs']==100 and r['steps']==100*math.ceil(r['training_windows']/64)
    assert sha256_file(path.parent/'checkpoint.pt')==r['checkpoint_sha256']
    assert not (set(r['train_indices']) & set(r['validation_indices']))
    assert math.isfinite(r['final_validation_mse'])
    records.append(r)
for system in ('pendulum','cartpole'):
    for seed in (42,43,44):
        same=[r for r in records if r['system']==system and r['seed']==seed]
        assert all(r['train_indices']==same[0]['train_indices'] for r in same)
        assert all(r['validation_indices']==same[0]['validation_indices'] for r in same)
pend=json.loads((out/'pendulum_final_summary.json').read_text())
cart=json.loads((out/'cartpole_final_summary.json').read_text())
domain=json.loads((out/'cartpole_start0_diagnostic_summary.json').read_text())
index={r['run_id']:r for r in records}
source={str(p.relative_to(ROOT)):sha256_file(p) for p in out.glob('*.py')}
basefiles=['training/train.py','training/loss.py','models/koopman_operator.py','models/model_builder.py','models/encoder.py','my_utils/dataset.py','dynamics/cartpole.py','dynamics/pendulum.py','docs/paper/20675_Learning_Koopman_Represe.pdf']
source.update({p:sha256_file(ROOT/p) for p in basefiles})
result=dict(status='ROOT_CAUSES_SUPPORTED_BUT_EXACT_PAPER_REPRODUCTION_NOT_CERTIFIED',scope='Two environments at 5% training data. No claim of full Table 1 fraction-grid reproduction.',completed_runs=len(records),completed_epochs=sum(r['epochs'] for r in records),optimizer_updates=sum(r['steps'] for r in records),root_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=source,paired_test={'pendulum':pend['summary'],'cartpole':cart['summary']},domain_diagnostic_mean=domain['negative_head_start0_mean'],domain_diagnostic_std=domain['negative_head_start0_sample_std'],run_table=[{k:r[k] for k in ('run_id','system','seed','arm','epochs','steps','final_validation_mse','elapsed_seconds','checkpoint_sha256')} for r in records])
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# P1 causal and mathematical audit — Pendulum and CartPole (2026-11-15)','',
'## Verdict','',
'**Exact paper reproduction is NOT CERTIFIED on both datasets.** The experiments identify causal failures and produce reference-scale short-horizon scores, but a changed sampler cannot be relabeled as the paper protocol. The public PDF/repo do not pin the exact Table 1 test-window/reduction contract or all optimizer/regularizer settings. No arbitrary match tolerance is invented.',
'',f"Completed **{len(records)} runs × 100 epochs**, totaling **{result['optimizer_updates']:,} optimizer updates**, all on CPU with the same official encoder and cached Euler equations. No production source, old checkpoint or data file was modified. The cached forward and gradients were checked against official forward on both systems. Geometry autograd was checked against finite differences; initial general-head dynamics match the corresponding signed/unsigned SPD control exactly.",
'', '## Preregistered paired test results','',
'5% means 250 Pendulum or 500 CartPole training trajectories. Each paired run uses 20 fixed random windows per trajectory, H=15, batch 64, initial lr .001, 100 epochs. Validation consists of 200 trajectories outside the union of the three train subsets, with a fixed selection seed. All final test values below use the same complete 1,000 held-out trajectories from x0, raw unweighted physical-state MSE. Variants were chosen on validation; final checkpoints were not selected on test. See [PROTOCOL.md](PROTOCOL.md).',
'', '| Dataset | Arm | Test H=15 mean ± sample std (3 seeds) | Paper 5% reference |','|---|---|---:|---:|']
for system,d,paper in [('Pendulum',pend,.0318),('CartPole',cart,.0024)]:
    for arm,metrics in d['summary'].items():
        a=metrics['open_loop_H15'];lines.append(f"| {system} | {arm} | {a['mean']:.8f} ± {a['sample_std']:.8f} | {paper:.4f} |")
lines+=['', 'Paper references are from Table 1 of the local PDF. A lower value does not prove exact reproduction when the metric contract/objective is not completely pinned.',
'', '| Dataset | Seed | Baseline H=15 | Variant H=15 | Relative reduction |','|---|---:|---:|---:|']
for system,d in [('Pendulum',pend),('CartPole',cart)]:
    for r in d['paired']:lines.append(f"| {system} | {r['seed']} | {r['baseline']:.8f} | {r['variant']:.8f} | {100*r['relative_reduction']:.2f}% |")
lines+=['', 'The new causal baseline uses a fixed explicit shuffle generator (700000+seed), so it is compared to its paired variants rather than claimed bitwise identical to old GPU/Lightning runs. Data/init/window seeds are held fixed within each pair.',
'', '## Confirmed causal findings','',
'### 1. Pendulum: additional encoder objectives and scheduler semantics materially affect convergence','',
'At seed 42, removing both encoder penalties changes only the objective terms, while epochmean changes only the scheduler monitor. Both improve validation. The noenc contrast was confirmed on three seeds: test H=15 improves in every pair. These encoder penalties are additional to the paper §3.4 stated prediction-plus-Gramian objective. `encoder_loss_flag=false` alone is insufficient because the predicted-state encoder penalty is added independently.',
'', '| Seed-42 Pendulum arm | Final validation H=15 | Final LR |','|---|---:|---:|']
for arm in ('baseline','epochmean','constant','noenc','noenc_epochmean','general_head','nogram'):
    r=index[f'pendulum-{arm}-s42'];lines.append(f"| {arm} | {r['final_validation_mse']:.8f} | {r['history'][-1]['lr']:.2e} |")
lines+=['', 'The combined noenc+epochmean result is not better than noenc alone: the effects are not additive. The original step-only logging feeds the latest batch metric to ReduceLROnPlateau; epochmean uses a sample-weighted epoch average. Constant LR and epochmean happen to produce identical seed-42 runs because that monitor never reduces LR in this run.',
'', '**Important trade-off:** noenc improves short-horizon prediction but degrades long rollouts. Test H=250 mean MSE changes from '+f"{pend['summary']['baseline']['open_loop_H250']['mean']:.6f} to {pend['summary']['noenc']['open_loop_H250']['mean']:.6f}"+'. Do not apply this as an unconditional production fix or claim long-horizon reproduction.',
'', '### 2. CartPole: positive-definite orientation excludes the true local input sign','',
'With the current state order and unmodulated input, canonical Bc has physical head e4 and P_head=HH^T+eps I. Therefore B[3,0]=sum(H[3,:]^2)+eps>0. The physical angular-force coefficient near the upright state is about -1.4634. Negative_head changes only the sign of the entire physical block, preserving invertibility and output rank, while permitting a negative coefficient. Its three-seed paired test improves in every seed, but remains far from the paper number.',
'', 'A one-coordinate signature was also tried and was worse than negating the entire head. The local CartPole canonical transform has positive determinant and is generally nonsymmetric; the single-coordinate signature moves to the opposite determinant component. This does not prove that every transform in that component is useless over finite rollouts, but shows why correcting one coefficient sign is not a complete capacity argument.',
'', 'The general_head arm permits a locally general invertible block P0 exp(M), with identical initial dynamics to negative_head (CartPole) or baseline (Pendulum). It preserves canonical output rank and passes finite-difference gradient checks, but does not by itself close the CartPole validation gap. No oracle gradients were used for any learned model.',
'', '### 3. CartPole: training-window coverage and first-window evaluation interact strongly','',
'An explicitly diagnostic 2×2 seed-42 control changes the window-start distribution while preserving trajectory count, initialization, minibatch ordering and update budget. start0 repeats each of the 500 first windows 20 times; it does not create 10,000 distinct windows.',
'', '| Physical transform | Random windows over whole trajectory | First window only (start=0) |','|---|---:|---:|']
lines.append(f"| Original SPD head | {index['cartpole-baseline-s42']['final_validation_mse']:.8f} | {index['cartpole-baseline_start0-s42']['final_validation_mse']:.8f} |")
lines.append(f"| Negative-definite head | {index['cartpole-negative_head-s42']['final_validation_mse']:.8f} | {index['cartpole-negative_head_start0-s42']['final_validation_mse']:.8f} |")
lines+=['', 'These are validation MSEs. The start0 intervention changes both state coverage and unique-window diversity (500 unique repeated windows versus 10,000 distinct random windows); the experiment establishes the effect of this complete sampler intervention, not an isolated numerical effect of state coverage alone. Fixing the input sign alone is insufficient; changing the sampled region alone with the wrong sign is insufficient. Their combination attains low local prediction error. The negative_head_start0 diagnostic was then repeated with three seeds and evaluated once on held-out test data:',
'', '| Seed | Diagnostic start0 test H=15 MSE |','|---:|---:|']
for r in domain['runs']:
    if r['arm']=='negative_head_start0':lines.append(f"| {r['seed']} | {r['test_first15_mse']:.8f} |")
lines+=['',f"Mean ± sample std: **{domain['negative_head_start0_mean']:.8f} ± {domain['negative_head_start0_sample_std']:.8f}**. This is **DIAGNOSTIC_ONLY_NOT_FAITHFUL_PAPER_REPRODUCTION**, because it intentionally departs from Appendix G.1 random-window training. The baseline_start0 arm has seed 42 only; do not infer a three-seed matched start0 baseline.",
'', '## Mathematical audit','',
'### Canonical rank and physical fidelity are different','',
'For any block-invertible P, A=P Ac P^-1 and B=P Bc preserve the canonical output-rank construction. Positive definiteness is stronger than invertibility and restricts representable input orientations. An analytic local CartPole linearization has an exact nonsymmetric P with positive determinant; Pendulum admits P approximately 3I. Both local identities were reconstructed to floating-point precision in `local_canonical_audit.json`. All final paired checkpoints pass rank n using the first n reachability blocks and pass similarity identities.',
'', '### Fixed B imposes a genuine global input-map restriction','',
'When z=Psi(x)=[x,phi(x)] and output x=Cz, the model instantaneous physical input map is constant CB. True dynamics are f0(x)+G(x)u. For independent zero-mean control with variance sigma²:',
'', '```text', 'E||f0(x)+G(x)u - C A Psi(x)-CBu||²', '= E||f0(x)-C A Psi(x)||² + sigma² E||G(x)-CB||²', '>= sigma² E||G(x)-E G(x)||².', '```',
'', 'This is a continuous-time vector-field error statement, **not** a lower bound on the reported multi-step state MSE. On the actual sampled seed-42 training windows, the bound is 0 for Pendulum (constant G=[0,3]) and about 32.0146 for CartPole. CartPole mean angular input gain is about -0.3980 over sampled train windows versus -1.4557 over first-15 validation states; its variation is much greater in training. This proves global representational mismatch for exact dynamics, not that a particular finite-horizon paper MSE is mathematically impossible. No H2/H3 readiness gate is changed by this audit.',
'', 'An evaluation-only upright linearization oracle gives CartPole validation H=15 MSE about 7.40e-6; it uses physical knowledge and is not a learned baseline result. It demonstrates that the initial-state evaluation region is much easier than the full trajectory domain.',
'', '### Gramian objective mismatch is real, but the executed term is not the main accuracy driver here','',
'The paper writes reciprocal minimum eigenvalue plus a condition penalty; executed code uses 0.005*softplus(.03-lambda_min). The prior measured gradient discrepancy remains valid. The new 100-epoch nogram ablation removes only that executed penalty:',
'', '| Dataset | Baseline validation H=15 | nogram validation H=15 |','|---|---:|---:|']
for system in ('pendulum','cartpole'):
    b=index[f'{system}-baseline-s42'];n=index[f'{system}-nogram-s42'];lines.append(f"| {system} | {b['final_validation_mse']:.8f} | {n['final_validation_mse']:.8f} |")
lines+=['', 'Removing the implemented penalty changes accuracy little in these seed-42 controls. It is a paper-objective/conditioning discrepancy, not demonstrated to be the dominant cause of the prediction-error gap. No unregularized reciprocal was silently installed at eigenvalues near 1e-12; its normalization, horizon, condition weight and stable differentiation must be pinned before calling such a change a faithful reproduction.',
'', 'Independent float64 block-exponential and adaptive-quadrature calculations agree for all final paired checkpoints. Gramian numerical rank depends on horizon, tolerance, scaling and conditioning; do not confuse it with the exact output-rank criterion or clip tiny eigenvalues to fabricate a margin. Oracle horizon sensitivity is saved separately.',
'', '## What remains unresolved for exact reproduction','',
'- Obtain/pin the actual Table 1 evaluation script: first vs sampled windows, horizon, state normalization, angle convention and reduction/weighting.',
'- Pin the experimental transform/state/input convention. The theorem allows invertible blocks, Appendix F discusses conservative SPD blocks, and the public code retains a sign restriction incompatible with the current CartPole upright input response.',
'- Pin the Gramian horizon, condition penalty weight and numerical stabilization used for the reported experiments; the formula and executable repo objective differ.',
'- Preserve the old data audit failure: the single over-tolerance transition is reproduced by float32 dopri5 and is absent from C02 subsets. Do not relabel it as data corruption or erase it.',
'- These experiments cover the 5% setting on two datasets, not the full fraction grid. New variants are diagnostic and retain separate checkpoints. Three seeds quantify observed variation but do not guarantee universal convergence.',
'', 'The official public README/operator/evaluator were fetched read-only and snapshotted under `upstream/` with blob hashes in `upstream_sources.json`. The README itself uses window 50 / lr .005 examples, differing from Table 4/5 values. The evaluator also skips length-250 trajectories when asked for H=250. Neither supplies a frozen, complete Table 1 protocol. No issue/email was sent to the authors.',
'', '## Complete exploratory validation table','', '| Run | Validation H=15 | Updates | Final LR |','|---|---:|---:|---:|']
for r in records:lines.append(f"| {r['run_id']} | {r['final_validation_mse']:.8f} | {r['steps']} | {r['history'][-1]['lr']:.2e} |")
lines+=['', '## Reproduction and evidence','',
'- `PROTOCOL.md`: prospective arm definitions, validation split and explicitly labeled extensions.',
'- `harness_verification.json`, `geometry_verification.json`: forward, gradient, initial-dynamics and rank checks.',
'- `summary.json`: source hashes, full run inventory, totals and final aggregates.',
'- `<run>/manifest.json`, `progress.json`, `result.json`, `checkpoint.pt`: training data indices, history, hashes and saved weights.',
'- `pendulum_final_summary.json`, `cartpole_final_summary.json`: primary paired test results, per-trajectory errors, independent Gramian/structure checks.',
'- `cartpole_start0_diagnostic_summary.json`: explicitly non-paper sampling control.',
'- All experiment Python files reside here, use the existing .venv, and leave production code unchanged. Geometry checkpoints require the associated geometry wrapper; do not load them through the original positive-SPD inference path.',
'', '```bash', '# Numerical/gradient checks', 'PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python outputs/reproduction/p1_root_cause_2026-11-15/experiment.py --verify', 'PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python outputs/reproduction/p1_root_cause_2026-11-15/verify_geometry.py', '# Example paired run (completed runs are skipped)', 'PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python outputs/reproduction/p1_root_cause_2026-11-15/experiment.py --system pendulum --arm noenc --seed 42', '# Rebuild the verified report after all runs and final evaluations exist', 'PYTHONPATH=. .venv/bin/python outputs/reproduction/p1_root_cause_2026-11-15/summarize.py', '```','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:result[k] for k in ('status','completed_runs','completed_epochs','optimizer_updates')},indent=2))
