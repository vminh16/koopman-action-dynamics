# CartPole — 5% training data, seeds 42/43/44

## Completed training

All three manifests report `complete`, 100 epochs and 15,700 optimizer steps. Each run uses 500/10,000 trajectories and 10,000 sampled windows of length 15. Model: KoopmanLitModule, controllable, N=25, unmodulated input, Euler solver. Initial LR 0.001, AdamW, batch 64. Final epoch checkpoints were used; no test-based checkpoint selection.

| Seed | Final train prediction loss | Test MSE: open-loop H=15 | Test MSE: open-loop H=250 | Test MSE: re-encode H=250, K=50 |
|---:|---:|---:|---:|---:|
| 42 | 0.02497466 | 0.09289890 | 50.59788465 | 65.99446461 |
| 43 | 0.02418091 | 0.08918348 | 43.17358069 | 67.95760022 |
| 44 | 0.02600020 | 0.09360111 | 39.56642541 | 61.97731586 |

## Across-seed summary

Standard deviation below is the sample standard deviation across three run MSEs, not across test trajectories. The confidence interval is a two-sided Student-t interval (df=2); three seeds provide limited precision. Seeds change both trajectory subsets and initialization.

| Protocol | Mean MSE | Sample standard deviation | 95% t interval |
|---|---:|---:|---:|
| open_loop_H15 | 0.09189450 | 0.00237392 | [0.08599736, 0.09779163] |
| open_loop_H250 | 44.44596358 | 5.62472149 | [30.47338080, 58.41854637] |
| reencode_H250_K50 | 65.30979356 | 3.04836531 | [57.73723433, 72.88235280] |

## Evaluation protocol

All runs use the same complete set of 1,000 held-out test trajectories, starting from x0. MSE is unweighted, raw physical-state MSE averaged over four coordinates, time and trajectories; angles are neither wrapped nor normalized. Official model.forward runs in float32 on CPU; squared-error reduction uses float64. Evaluation was computed during aggregation because no evaluation files existed. Batched forward was cross-checked against a single-trajectory forward. H=15 means 0.3 s; H=250 means 5 s. Periodic re-encoding uses the preceding predicted state every 50 steps, with no ground-truth resets.

The training prediction loss applies temporal weights, so it is not numerically identical to the unweighted test MSE. These are locked-repo protocol results; exact paper comparability is not established here.

## Structure and independent float64 controllability checks

All checkpoints pass A=P Ac P^-1, B=P Bc and positive-definite P checks. The first four output reachability blocks have rank 4 for every seed; scaling A before forming the full 25-block matrix also gives output rank 4. The unscaled 25-block matrix returns rank 2 because of numerical dynamic range, and is not used to claim loss of output controllability.

| Seed | Output rank (first 4 blocks) | Gramian numerical rank (tol 1e-10) | Minimum eigenvalue Wy(0.3) |
|---:|---:|---:|---:|
| 42 | 4/4 | 3/4 | 4.38851276e-12 |
| 43 | 4/4 | 3/4 | 2.50216108e-12 |
| 44 | 4/4 | 3/4 | 4.74258936e-12 |

Gramian computation uses float64 block matrix exponentials, independently checked against adaptive quadrature. Positive minimum eigenvalues are extremely small; at the declared tolerance the short-horizon Gramian is numerically rank deficient. No positive margin is fabricated: rho_T is null. This indicates weak short-horizon reachability conditioning, not proof of exact mathematical uncontrollability or controllability of the nonlinear plant with bounded inputs.

## Provenance and limitations

Config content and SHA-256, checkpoint SHA-256, dataset hashes, effective hyperparameters and current sampler-source hash were verified against each manifest. Strict checkpoint loading passed. Each training manifest records clean source at commit `5701981ce63517637f9e7cdbb1dafc7bf68e9d6f`.

The previously recorded dataset integrity audit is **FAILED**: train transition maximum RK4 error 0.000160761215 exceeds the fixed 0.0001 tolerance. This aggregation does not regenerate data, relax tolerance, retrain models or certify paper reproduction.

Seed 42 has an earlier partial Lightning log (version_0, checkpoint epoch 12). The summary uses the completed manifest and final `checkpoints/C02-s42.pt`, not that partial checkpoint.

## Artifacts

- `cartpole_frac0.05_summary.json`: complete metrics, provenance and numerical diagnostics.
- `cartpole_frac0.05_summary.csv`: flat run table.
- `C02-s{42,43,44}/evaluation.json`: per-run evaluation.
- `C02-s{42,43,44}/controllability.json`: per-run Gramian/structure/rank diagnostics.
