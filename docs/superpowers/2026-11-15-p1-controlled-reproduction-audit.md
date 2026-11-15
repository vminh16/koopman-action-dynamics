# P1 — controlled reproduction audit (2026-11-15)

**Verdict: ROOT_CAUSES_SUPPORTED; EXACT_PAPER_REPRODUCTION_NOT_CERTIFIED.**

28 observational-training runs completed at 100 epochs each (353,800 optimizer
updates), covering Pendulum and CartPole at 5% data. Original source, data and
checkpoints were preserved; variants have separate checkpoint/provenance files.

Full evidence: [causal report](../../outputs/reproduction/p1_root_cause_2026-11-15/REPORT.md),
[protocol](../../outputs/reproduction/p1_root_cause_2026-11-15/PROTOCOL.md),
[JSON](../../outputs/reproduction/p1_root_cause_2026-11-15/summary.json),
[plots](../../outputs/reproduction/p1_root_cause_2026-11-15/causal_results.png).

## Three-seed paired test results

All entries use 1,000 held-out trajectories, raw physical-state MSE over the first
15 steps. Variants were chosen using a separate validation split; the final
checkpoint is epoch 100. Numbers are mean ± sample standard deviation.

| Dataset | Paired baseline | Single-factor variant | Variant MSE |
|---|---:|---|---:|
| Pendulum | 0.08619425 ± 0.04338556 | Omit both encoder-consistency losses | 0.01447501 ± 0.01196399 |
| CartPole | 0.09179632 ± 0.00259836 | Negate physical transform block; retain random windows | 0.06004065 ± 0.00180977 |

Every seed improves in both pairs. These are new matched CPU/shuffle controls,
not claims of bitwise replay of older GPU/Lightning runs.

## Causal and mathematical conclusions

1. **Pendulum:** extra encoder objectives and scheduler-monitor semantics cause
   substantial convergence differences in the controlled runs. Removing the
   encoder penalties improves H=15 but worsens H=250 mean MSE from 51.17 to
   475.52. Combining changes does not necessarily improve the result.
2. **CartPole:** SPD P with fixed canonical B forces a positive angular-input
   coefficient, whereas the physical upright coefficient is negative. The
   negative-head intervention preserves output rank 4 and improves all three
   test pairs, but does not close the paper gap.
3. **Window selection matters:** a seed-42 2×2 validation experiment gives
   0.09105911 (original/random), 0.09586752 (original/start0), 0.05909300
   (negative/random), and 0.00084898 (negative/start0). Start0 changes both state
   coverage and unique-window diversity, so those contributions are not isolated.
4. **Diagnostic numerical match is not faithful replication:** negative-head +
   start0 reaches test MSE 0.00087518 ± 0.00012415 over three seeds, but deliberately
   violates the paper's random-window training description. It cannot certify
   paper reproduction. The start0 baseline has one seed only.
5. **Gramian:** removing the executed softplus term barely changes seed-42
   validation MSE on either dataset. Its discrepancy with the paper reciprocal/
   condition formula is real, but it is not established as the primary cause of
   the accuracy gap. Exact output rank and finite-horizon numerical conditioning
   remain separate questions.
6. **Input-map limitation:** for independent zero-mean input and z=[x,phi(x)],
   fixed B imposes constant physical input map CB. The unavoidable vector-field
   error includes input variance times Var[G(x)]. This is zero for Pendulum and
   nonzero over the CartPole training domain; it is not a bound on the reported
   multi-step state MSE or grounds to change the P2/H2/H3 gates.

Forward/gradient parity with the official model, geometry finite-difference
gradients, inverse identities, output ranks, held-out split separation and saved
checkpoint hashes were verified. Final paired Gramians were independently
checked with float64 block exponentials and adaptive quadrature. Analytic
linearization controls were evaluation-only; no oracle gradients trained models.

## Remaining requirements

Exact reproduction requires the actual Table 1 evaluation/window/normalization
contract and the experimental Gramian/transform settings. The public repository
was inspected read-only and snapshotted; its examples and evaluator do not
resolve these ambiguities. The existing dataset audit failure remains recorded
with its float32 solver explanation. This experiment covers 5% on two datasets,
not the full fraction grid. No paper PASS or H2/H3 readiness change is warranted.
