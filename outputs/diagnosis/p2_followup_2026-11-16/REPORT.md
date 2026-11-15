# P2 follow-up — 2026-11-16

The planned sequential follow-up is complete: E05 refresh, 140 paired rotation runs, independent mathematical controls, checkpoint recomputation and source/pairing checks. Historical P2 evidence is preserved.

## Protocol and scope

Frozen historical P2 rotation datasets: 30 training trajectories, 10 evaluation trajectories, 600 fixed windows of length 15, batch 64, dt .02, latent dimension 6. Five seeds 42–46; strengths 0/.5/1/2. AdamW LR .001, weight decay 1e-5. Explicit shuffle RNG 700000+seed. CPU float32 training; float64 MSE reductions and output Gramians. The held-out response probe is shared across all strengths and seeds.

At 25 epochs: baseline, epoch-mean scheduler, removal of BOTH encoder consistency terms, general invertible physical head P0 exp(M), and their combined diagnostic variant. At 100 epochs: baseline and combined, chosen before observing results. General heads preserve initial A/B and require the diagnostic loader. These are labeled deviations from upstream; no oracle gradients, H2/H3 method, new excitation distribution or sampler change is used. Evaluation uses final epochs; there is no test-based checkpoint/arm selection.

Total: 140 completed runs, 65,000 optimizer updates; 40 long runs reproduce their first 25 epochs exactly. New ablations target the rotation family; Unicycle/Quadrotor outcomes remain historical and are not generalized from these runs.

## E05 refresh

Fixed explicit CartPole registration: previous discovery missed .yml configs and C02 checkpoint names. Pendulum seed 42 and CartPole C02 seeds 42/43/44 strict-load, match dataset hashes, repeat identically, and match a later recomputation. This verifies artifact identity and computation, not exact paper reproduction or the failed dataset-transition tolerance.

| Checkpoint | rho_T | Numerically rank deficient | d_orient p95 |
|---|---:|---|---:|
| pendulum-s42 | 0.00234354 | False | 0.439668 |
| cartpole-s42 | undefined | True | 0.909166 |
| cartpole-s43 | undefined | True | 0.934735 |
| cartpole-s44 | undefined | True | 0.901136 |

CartPole remains algebraically output-controllable according to the P1 independent checks. Numerical short-horizon Gramian rank deficiency does not contradict that algebraic property. No positive usable margin is invented.

## Measured paired results

Values are mean ± sample standard deviation across five seeds. Response is normalized continuous-time action-response error; H15 is unweighted physical-state rollout MSE from x0, without re-encoding.

| Epochs | Arm | a | Response | H15 MSE |
|---:|---|---:|---:|---:|
| 25 | baseline | 0.0 | 1.01002 ± 0.009 | 0.00393127 ± 0.000358 |
| 25 | baseline | 0.5 | 1.02058 ± 0.00747 | 0.0043062 ± 0.000443 |
| 25 | baseline | 1.0 | 1.03169 ± 0.00381 | 0.00445121 ± 0.000634 |
| 25 | baseline | 2.0 | 1.04994 ± 0.00352 | 0.00386979 ± 0.000457 |
| 25 | epochmean | 0.0 | 1.01002 ± 0.009 | 0.00393127 ± 0.000358 |
| 25 | epochmean | 0.5 | 1.02058 ± 0.00747 | 0.0043062 ± 0.000443 |
| 25 | epochmean | 1.0 | 1.03169 ± 0.00381 | 0.00445121 ± 0.000634 |
| 25 | epochmean | 2.0 | 1.04994 ± 0.00352 | 0.00386979 ± 0.000457 |
| 25 | general | 0.0 | 0.977832 ± 0.00894 | 0.00371984 ± 0.000283 |
| 25 | general | 0.5 | 0.990846 ± 0.00971 | 0.004134 ± 0.000352 |
| 25 | general | 1.0 | 1.00656 ± 0.00921 | 0.00434366 ± 0.000554 |
| 25 | general | 2.0 | 1.03827 ± 0.00582 | 0.00379175 ± 0.00048 |
| 25 | general_noenc_epochmean | 0.0 | 0.992446 ± 0.00598 | 0.00253057 ± 0.000154 |
| 25 | general_noenc_epochmean | 0.5 | 1.00239 ± 0.00555 | 0.00253799 ± 0.000112 |
| 25 | general_noenc_epochmean | 1.0 | 1.01432 ± 0.00461 | 0.00253878 ± 0.000111 |
| 25 | general_noenc_epochmean | 2.0 | 1.03688 ± 0.00219 | 0.00241802 ± 0.000209 |
| 25 | noenc | 0.0 | 1.00935 ± 0.00328 | 0.00257658 ± 0.000158 |
| 25 | noenc | 0.5 | 1.01851 ± 0.00238 | 0.00257199 ± 0.000109 |
| 25 | noenc | 1.0 | 1.02709 ± 0.00102 | 0.00256175 ± 7.52e-05 |
| 25 | noenc | 2.0 | 1.04032 ± 0.000935 | 0.00241363 ± 0.000161 |
| 100 | baseline | 0.0 | 1.01463 ± 0.00987 | 0.00281214 ± 0.000241 |
| 100 | baseline | 0.5 | 1.02298 ± 0.00441 | 0.00296188 ± 0.000301 |
| 100 | baseline | 1.0 | 1.03282 ± 0.00261 | 0.00295688 ± 0.000327 |
| 100 | baseline | 2.0 | 1.04865 ± 0.00752 | 0.00265457 ± 0.000225 |
| 100 | general_noenc_epochmean | 0.0 | 0.954208 ± 0.0126 | 0.0024184 ± 0.000122 |
| 100 | general_noenc_epochmean | 0.5 | 0.971219 ± 0.0128 | 0.00252684 ± 0.000122 |
| 100 | general_noenc_epochmean | 1.0 | 0.994283 ± 0.0156 | 0.00256262 ± 0.000147 |
| 100 | general_noenc_epochmean | 2.0 | 1.03351 ± 0.00472 | 0.00257398 ± 3.02e-05 |

## Why the original stress ratio was weak

For the single-input rotation family, ||G(x)||=1. On the shared probe, with weights w=(u-u_ref)^2, normalized response error decomposes exactly as:

`E_response = weighted Var(G) + ||CB - weighted mean(G)||²`

The first term is the best-constant-map floor on this probe. The second is fitting error beyond that floor. This is a physical vector-field/action-response identity, not a bound on rollout MSE.

| a | Best-constant floor | Ignore-input one-step MSE |
|---:|---:|---:|
| 0.0 | 0 | 0.00026866228 |
| 0.5 | 0.1145 | 0.00026854295 |
| 1.0 | 0.386224 | 0.00026829028 |
| 2.0 | 0.845222 | 0.00026800983 |

At a=0 the floor is zero, yet the learned response error is near the ignore-input reference of 1. As stress increases the irreducible floor rises, while distance to the best constant map can fall; total error can therefore change little. A small growth ratio cannot isolate model-class limitation when the easy control has not been fitted.

The old optimization gate used one-step MSE <= .05. An exact-drift predictor that ignores inputs scores about .000269 here, so that threshold does not establish action-response fidelity. The historical gate was evaluated as written; its diagnostic power is limited.

There is also a verified head restriction: at a=0 the true map is [1,0]^T, but original SPD parameterization forces B_physical[1]>=.1. Its normalized response error is therefore at least .01. This proves an exact-representation mismatch, but does not explain an observed error near 1 by itself.

An analytic nonsymmetric head [[.2,1],[-1,0]] (det=1) reconstructs the a=0 A/B exactly, has a real exponential realization, and gives exact-ZOH held-out one-step MSE 3.65e-16. This is a feasibility control, not a trained model or evidence of H2 superiority.

## Gates and interpretation

| Diagnostic arm | Minimum per-seed growth ratio | All seeds monotone | Response-growth criterion only |
|---|---:|---|---|
| baseline-e25 | 1.02842 | True | False |
| epochmean-e25 | 1.02842 | True | False |
| noenc-e25 | 1.02618 | True | False |
| general-e25 | 1.05424 | True | False |
| general_noenc_epochmean-e25 | 1.03756 | True | False |
| baseline-e100 | 1.02032 | False | False |
| general_noenc_epochmean-e100 | 1.06727 | True | False |

Historical P2 and verification PASS remain preserved. H2 readiness stays NO-GO: these controls do not establish a well-fitted easy-system baseline or compare matched learning mechanisms. The analytical fixed-B floor is real, but it alone does not validate factorized learning.

H3 stays NO-GO. CartPole E05 has no usable numerical short-horizon margin; rotation H3 diagnostics are sampled checks, not global certificates. This follow-up does not rerun MPC for diagnostic variants, so historical solver latency/success cannot be transferred to them.

Next justified prerequisite: establish action-response fidelity on the constant-input-map positive control, and then preregister a stress/excitation protocol whose excess fitting error is small relative to the theoretical floor. A response-based optimization criterion would be a NEW protocol, not a retroactive change to the old gate. Do not start H2/H3 training yet.

## Reproduce and verify

Run from repository root:

```bash
export MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1
.venv/bin/python -m research.diagnostics.p2_followup --stage verify
.venv/bin/python -m research.diagnostics.p2_followup --stage e05
.venv/bin/python -m research.diagnostics.p2_followup --stage train
.venv/bin/python -m research.diagnostics.p2_followup --stage recompute
.venv/bin/python -m research.diagnostics.p2_followup_math
.venv/bin/python -m research.diagnostics.verify_p2_followup
.venv/bin/python -m research.diagnostics.summarize_p2_followup
```

Completed runs are reused by the training command; use a separate archived output path/code copy for a fresh training repetition. Recompute always reloads all 140 checkpoints and verifies data hashes. Imported source hashes are checked by verify_p2_followup. Gramians use float64 block exponentials and independent adaptive quadrature.

Artifacts: summary.json, mathematical_controls.json, e05_refresh.json, source_snapshot.json, recompute.json, verification.json, historical_recompute.json, and one checkpoint/result per run. See results.png for the plotted comparison.
