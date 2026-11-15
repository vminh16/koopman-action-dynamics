# P1 CartPole code–paper mismatch audit — 2026-11-14

## Verdict and scope

**The current code reproduces the locked repository protocol, but does not establish reproduction of the paper's training objective or reported accuracy.** The strongest diagnosed blocker is a local input-response sign restriction in the CartPole operator. Two further verified issues are the different Gramian objective and a scheduler driven by step loss rather than epoch-mean loss. Controlled retraining has not been performed, so their individual contributions to final MSE are not quantified.

Audit base: `b5c0c0e30cc3be8a45d55fa576e2f82bf09bdce7`. No production source, configs, datasets or trained checkpoints were changed. Newly running C01/C03/C04 experiments were left untouched.

The primary reference is the local [paper PDF](../paper/20675_Learning_Koopman_Represe.pdf): Theorem 2 / Eq. (5)–(6), §3.3–3.4, Appendix F.1, Appendix G.1–G.2/Table 5 and Table 1. Table 1 lists CartPole 5% MSE **0.0024**; the current open-loop first-15-step mean is **0.09189450** (nominally 38.29×, **not an established matched-protocol ratio**). The PDF does not fully specify the Table 1 evaluator's window selection/reduction, optimizer/scheduler, epoch count or sampling refresh schedule.

## 1. Primary blocker: the input response has the wrong sign

Source: [operator](../../models/koopman_operator.py#L79), [SPD transform](../../models/koopman_operator.py#L86), [CartPole ODE](../../dynamics/cartpole.py#L24), [Euler rollout](../../models/model_builder.py#L186).

Current state order is `[x, x_dot, theta, theta_dot]`, with force `u` passed unchanged (`use_modulated_u=false`). At the upright origin, the analytic input vector is approximately

```
G(0) = [0, +0.97560976, 0, -1.46341463]^T.
```

The operator fixes the physical head of `Bc` to `e4`, uses `B=P Bc`, and makes `P_head=H_head H_head^T + eps I` with `eps>=0.1`. Because P is block diagonal,

```
B[3,0] = P_head[3,3] = sum(H_head[3,:]^2) + eps >= 0.1.
```

Thus **no amount of training can make this model's instantaneous angular-force coefficient negative** under the current conventions. The encoder cannot repair this: it sees x, not the current u. With the actual Euler rollout, the one-step input response is directly `dt * CB`, so the sign conflict is also observable at the forward API.

| Seed | Model d(theta_dot_dot)/du | True value at origin |
|---:|---:|---:|
| 42 | +0.791548133 | -1.463414550 |
| 43 | +0.810001016 | -1.463414550 |
| 44 | +0.778966427 | -1.463414550 |

A direct analytic coverage check also finds that the physical angular-input gain is negative at **100% of the first-15-step test predecessor states** (mean approximately -1.45595), and at 68.45052% of all train predecessor states. Thus the sign conflict is not confined to a single origin probe. This coverage calculation uses the same CartPole parameters and the analytic coefficient `-cos(theta) / [1.1 * 0.5 * (4/3 - 0.1*cos(theta)^2/1.1)]`; no fitting is involved. See `state_coverage.json`.

A deterministic black-box repro on C02-s42 returns `+0.791548133` versus oracle `-1.463414550` and intentionally fails a sign-consistency assertion. This is a representation-capacity issue, not proof of lost output controllability: output rank may remain 4/4 while the learned dynamics are wrong.

**Paper nuance:** Theorem 2 requires full-rank block matrices P1/P2, which permits a negative angular input coefficient. Appendix F.1 also describes choosing positive-definite blocks and explicitly acknowledges conservatism. Therefore the SPD choice is **not wholly undocumented by the paper**; the finding is its incompatibility with the current physical state/input convention, and an unresolved discrepancy with the reported CartPole experiment. It does not prove which private experimental implementation produced Table 1.

A diagnostic construction with general block-invertible P matches `G(0)` exactly and retains output rank 4. It is saved in `capacity_example.json`; it is not a trained replacement or a claim that a fixed B matches all CartPole angles.

## 2. Verified objective mismatch: Gramian shaping is almost absent in the measured gradient

Paper §3.3 defines

```
Rgram = 1 / lambda_min(Wy) + gamma * condition_number(Wy).
```

The executed [loss](../../training/loss.py#L224) is

```
softplus(0.03 - lambda_min(Wy)), weighted by 0.005.
```

This is a different function and has no condition-number penalty. For tiny lambda_min, it stays near `softplus(0.03)=0.7082597` instead of penalizing near-singularity as the reciprocal does.

On each final checkpoint, a deterministic batch of 64 training windows spanning the selected subset gives:

| Seed | Prediction gradient norm | Weighted Gramian gradient norm | Ratio |
| 42 | 2.91335e-02 | 7.07100e-12 | 2.42710e-10 |
| 43 | 3.67028e-02 | 9.06894e-12 | 2.47091e-10 |
| 44 | 6.32213e-02 | 2.10701e-12 | 3.33276e-11 |

These norms are over the operator parameters only. They show a negligible direct contribution to the summed gradient **at these measured checkpoints/batches**, not a universal statement about every earlier optimization step. TensorBoard provides supporting whole-run evidence: the logged controllability loss stays within about one float32 ULP of 0.7082597 for every completed C02 run.

Training computes Wy with float32 coupled Euler steps. Some eigenvalues become slightly negative numerically; the independent float64 block-exponential/quadrature computation is the more reliable diagnostic. Do not replace the loss with an unregularized reciprocal blindly: at eigenvalues around 1e-12 that objective can be extremely ill-conditioned.

## 3. Verified scheduler issue: initial 1e-3 becomes 1e-8, monitored on step loss

[training_step](../../training/train.py#L146) calls `self.log('train_total_loss', loss)` without specifying epoch aggregation. The installed Lightning validator reports `default_on_step=True`, `default_on_epoch=False` for training_step. [ReduceLROnPlateau](../../training/train.py#L182) monitors that metric once per epoch, so it is driven by the latest step loss rather than the epoch mean. Random window/batch composition can therefore influence LR decisions.

The saved Lightning checkpoints confirm **final LR=1e-8 for all three seeds**, versus the configured initial LR=1e-3. `epochs=100` is true, but it does not mean all 100 epochs used an appreciable learning rate. The manifest's `final_metrics` are last callback/step metrics, not established epoch averages.

This logging/monitoring behavior is verified. Whether constant LR or proper epoch-mean monitoring improves final accuracy needs an isolated retraining ablation. The paper's experimental tables do not establish a particular optimizer/scheduler, so AdamW/ReduceLROnPlateau should be labeled repo choices rather than claimed paper settings.

## 4. Other confirmed differences and uncertainties

- **Extra losses:** [training_step](../../training/train.py#L118) optimizes `Lpred + Lencoder_true + 0.5 Lencoder_pred + 0.005 Lcontrol`; paper §3.4 states prediction plus Gramian regularization. These encoder terms are additional objectives. In the measured batches their operator gradients are smaller than the prediction gradient, so they are not shown to be the dominant problem. `encoder_loss_flag=false` disables only `encoder_true`; `encoder_pred` is always added unless its coefficient is set to zero.
- **Sampling:** [dataset](../../my_utils/dataset.py#L41) chooses 20 distinct starts per selected trajectory once at construction and reuses them every epoch. Appendix G.1 describes a randomly sampled 15-step window, but does not explicitly settle the refresh frequency or windows-per-trajectory count. Fixed 20-window reuse is established code behavior; a requirement to redraw exactly once per epoch is **not established** from the PDF alone. The C02 runs have 10,000 windows and 15,700 updates over 100 epochs.
- **Matched settings:** N=25, 3 hidden ReLU layers of width 64, batch 64, initial LR 1e-3, H=15, dt=.02 and 10,000/1,000 trajectory counts match the stated Table 5/G.1 settings. Model class, config propagation, checkpoint shapes and hash contracts are not the currently diagnosed problem.
- **Evaluation:** the recorded primary metric uses the first 15 steps from x0 on every test trajectory and unweighted raw-coordinate MSE. The paper supports prediction-error evaluation and shows 15-step examples, but does not establish this exact Table 1 reduction/window policy. Current H=250 and K=50 predicted-state re-encoding are additional diagnostics, not proven Table 1 protocols. Paper §3.5's MPC resets use measured states and are not the same operation as the repo's predicted-state re-encoding.

## 5. Isolating Euler error: changing the rollout solver alone does not close the gap

Holding each checkpoint, encoder, physical inputs, test set and H=15 fixed, exact ZOH integration of the learned linear latent ODE gives:

| Seed | Current Euler MSE | Exact ZOH MSE |
| 42 | 0.09289890 | 0.09039338 |
| 43 | 0.08918348 | 0.08741307 |
| 44 | 0.09360111 | 0.09348360 |

The mean changes from **0.09189450 to 0.09043002** (about **1.59%** reduction). This rules out **inference Euler error alone** as an explanation of the nominal 38× discrepancy. It does not test the effect of retraining with a different solver. The exact-ZOH probe uses float64 linear algebra on the actual learned matrices; it is not a new trained checkpoint.

## 6. Dataset transition audit: explained numerical discrepancy, not the identified C02 training cause

The full audit failure was reproduced without changing tolerance. Exactly **1 of 2,500,000 train transitions** exceeds 1e-4: zero-based trajectory **3834**, control-step **248**, angle coordinate. The maximum error is **0.000160761215**; all-state local RMSE over the full train set is approximately **1.51869e-6**.

Replay from the same recorded predecessor and force:

| Integration | Error against stored next state |
|---|---:|
| Author dopri5, float32, original timestamps [4.96,4.98] | 0 (bit-for-bit) |
| Author dopri5, float32, local timestamps [0,.02] | 4.76837e-7 |
| dopri5, float64 | 1.60764e-4 |
| RK4, 1/2/4/16 substeps, float64 | about 1.60762e-4 |
| Independent SciPy DOP853, tight tolerances | 1.60762e-4 |

This demonstrates a reproducible discrepancy between the original float32 adaptive-solver result and the high-precision ODE result on this transition. It is not evidence of a shifted control index or random corruption: the original solver reproduces the stored value exactly; neighboring controls produce much larger errors.

**The offending trajectory is absent from all three C02 selected training subsets.** It therefore did not directly enter their training windows, and this specific tolerance failure cannot explain the C02 training gap. Other smaller numerical errors exist; this is not a claim of a globally exact dataset.

The old audit report remains **FAILED** at the preregistered threshold. This audit supplies an explanation instead of loosening the threshold or rewriting data. Generation seeds/provenance are still not established by replaying one transition.

## 7. Recommended next experiments, in causal order

1. Add a sign-response regression check and evaluate a block-invertible transform/input-orientation variant that can represent the physical input sign while retaining the rank construction. Keep original C02 checkpoints and label the variant explicitly; do not call an untested change a faithful paper reproduction.
2. Isolate a stable Gramian objective matching the stated paper intent and test its gradients against independent float64 values. Do not simultaneously change encoder losses/sampler/LR.
3. Compare proper epoch-mean scheduler monitoring against the original behavior with identical init/subset/window seeds and update budget. A constant-LR arm can be a separate experiment.
4. Separately ablate encoder penalties and fixed-window reuse; record sampling refresh and total optimizer updates. Changing every component together cannot identify a cause.
5. Define a paper-comparable evaluation contract before interpreting the Table 1 ratio. Use the held-out test set for final reporting, with a separate validation split for choices. No new test-based model selection occurred in this audit.

These are proposed follow-ups, not work performed in this audit. No additional model training was started.

## 8. Reproducible diagnostic evidence

Snapshots and probe scripts are under [outputs/reproduction/p1_code_paper_audit](../../outputs/reproduction/p1_code_paper_audit/). They include source/data/checkpoint SHA-256 hashes, the checkpoint/gradient probe, optimizer states, TensorBoard scalar summaries, the worst-transition replay and the full-dataset scope count. Probe scripts write `/tmp/p1_*.json`; captured JSON snapshots are stored alongside them.

From repository root:

```bash
# Intentionally exits 1 on the currently diagnosed sign conflict.
PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl .venv/bin/python \
  outputs/reproduction/p1_code_paper_audit/sign_repro.py

# Diagnostic measurements; these do not train or overwrite checkpoints.
PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python \
  outputs/reproduction/p1_code_paper_audit/p1_checkpoint_probe.py
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python \
  outputs/reproduction/p1_code_paper_audit/p1_transition_probe.py
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python \
  outputs/reproduction/p1_code_paper_audit/p1_dataset_scope_probe.py
```
