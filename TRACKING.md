# TRACKING — runs, evidence and replay

Shortened 2026-11-17. Every earlier entry (per-run tables, P0 config audit, P2 replay recipes,
artifact policy, exact commands) is kept verbatim in the
[archive](docs/archive/TRACKING_2026-11-17_full.md); links below point into it.

**Rules (AGENTS.md):** preregister question, script SHA-256, seeds, budgets, data identities
and decision rules *before* running; final checkpoint only, no test-based selection; never
overwrite a dataset path whose results were read; raw runs/data stay ignored in `outputs/`.
Status values: `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`, `NEEDS_INVESTIGATION`.
Machine note: 15 GB RAM, no swap. Run grids on CPU (`OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=`)
inside `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`; 8 CUDA processes OOM'd the desktop.

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/cartpole-mpl
.venv/bin/python -m unittest discover -s tests
```

## Index

| Date | Experiment | Verdict | Report | Preregistration / replay |
|---|---|---|---|---|
| 11-13 | P0 Pendulum config audit, seed probe | CLOSED: Koopman; lambda_control drift not the cause; bimodal seeds | [E01 manifest](outputs/reproduction/E01/manifest.yaml) | [archive §3.1–3.2](docs/archive/TRACKING_2026-11-17_full.md#31-config-audit-2026-11-13) |
| 11-13/14 | P2 stress tests E05–E09 (historical) | Gate P2 PASS; old H2 ratio 1.030 (under-fitted baseline) | [p2_failure_report.yaml](outputs/diagnosis/p2_failure_report.yaml) | [archive §3.3](docs/archive/TRACKING_2026-11-17_full.md#33-phase-p2-controlled-stress-tests-summary), [replay §4](docs/archive/TRACKING_2026-11-17_full.md#4-commands) |
| 11-14 | CartPole 5% C02 s42–44 | H15 `0.0919 ± 0.0024`; NEEDS_INVESTIGATION then | [summary](outputs/reproduction/cartpole_frac0.05_summary.md) | [archive §3.4](docs/archive/TRACKING_2026-11-17_full.md#34-cartpole-5--completed-runs-and-independent-evaluation-2026-11-14) |
| 11-15 | P1 causal audit, 28 runs | encoder losses/scheduler + SPD sign restriction matter | [report](outputs/reproduction/p1_root_cause_2026-11-15/REPORT.md) | [archive §3.5](docs/archive/TRACKING_2026-11-17_full.md#35-p1-controlled-causal-audit-2026-11-15) |
| 11-16 | P2 follow-up, 140 rotation runs | easy-system input map poorly fitted; one-step MSE gate insufficient | [report](outputs/diagnosis/p2_followup_2026-11-16/REPORT.md) | [archive §3.3.1](docs/archive/TRACKING_2026-11-17_full.md#331-p2-follow-up--e05-refresh-and-p1-derived-controls-2026-11-16) |
| 11-16 | P2 config check (22 runs) | compatible system learnable; original-coordinate heads fail | [report](docs/superpowers/2026-11-16-p2-config-check.md) | [archive §3.3.2](docs/archive/TRACKING_2026-11-17_full.md#332-p2-config-check--compatible-positive-control-2026-11-16) |
| 11-16 | P2 robust flow (21 runs) | compatible 5/5; original heads 0/5 at shared config | [report](docs/superpowers/2026-11-16-p2-robust-flow.md) | [archive](docs/archive/TRACKING_2026-11-17_full.md#p2-robust-flow-protocol--locked-before-execution-2026-11-16) |
| 11-16 | Drift/coverage + equal tuning (32 runs) | general head LR .01 5/5 → **baseline frozen** | [report](docs/superpowers/2026-11-16-p2-drift-coverage-tuning.md), [BASELINE](docs/BASELINE.md) | [archive](docs/archive/TRACKING_2026-11-17_full.md#driftcoverage--equal-budget-tuning--preregistered-2026-11-16) |
| 11-16 | P1 upstream-faithful 5% check (30 runs) | dense-P + start≤4 reaches CartPole 0.64× paper | [report](outputs/reproduction/p1_upstream_faithful_2026-11-16/REPORT.md) | [archive](docs/archive/TRACKING_2026-11-17_full.md#p1-upstream-code-faithful-check--preregistered-2026-11-16) |
| 11-16 | P1 fraction grid (96 runs) | `dense_s5` 7/8 cells, fails CartPole 1% → **P1 closed, mismatch explained** | [report](outputs/reproduction/p1_upstream_faithful_2026-11-16/REPORT_grid.md) | [below](#p1-fraction-grid) |
| 11-17 | H2 diagnosis (45 runs + CartPole bands) | fixed-B disease confirmed; G.5 halves it (test label `NO_GO_CHEAP_FIX_G5`) | [report](outputs/diagnosis/h2_diagnosis_2026-11-17/REPORT.md) | [below](#h2-diagnosis) |
| 11-17 | H2 tests vs G.5 (T1–T4) | `NOT_CONFIRMED_BEYOND_G5`: rotation residual survives G.5; G.5 does not fix CartPole sign flip; unicycle fails class-limit criterion; quadrotor test uninformative | [report](outputs/diagnosis/h2_g5_tests_2026-11-17/REPORT.md) | [below](#h2-tests-against-g5) |
| 11-17 | H2 gate follow-up (C1 unicycle, C2 quadrotor thrust rows, C3 CartPole cos-G.5) | `NOT_YET`: C1 misses class-limit rule 3/5 (needs 4/5); C2 insensitive; cos-G.5 does not fix CartPole | [report](outputs/diagnosis/h2_gate_2026-11-17/REPORT.md) | [below](#h2-gate-follow-up) |
| 11-17 | C4 CartPole confirmatory test (fresh seeds 45–47) | **PASS → H2 readiness GO** | [report](outputs/diagnosis/h2_cartpole_confirm_2026-11-17/REPORT.md) | [below](#c4-cartpole-confirmatory-test) |

## P1 fraction grid

Script `research/diagnostics/p1_upstream_faithful.py` (sha256 `9bf7ae531597077a`); 2 systems ×
1/5/10/30% × arms `spd_full`, `spd_s5`, `dense_s5`, `dense_full` × seeds 42–44, 100 epochs, CPU.
Rule: 3-seed mean test H15 ≤ 2× paper per cell; `dense_s5` all 8 → REPRODUCED_WITH_LABELED_CODE_VARIANT.
Result: 7/8, fails CartPole 1% (3.30×; seed 43 LR collapse). All 18 shared 5% runs match the
CPU 5% check exactly. Attempts 1–2 (8 CUDA processes → host OOM; 3 GPU processes → 3× slower
than CPU) left no completed run. `summary_grid.json` sha256 `91f5e65f2040dc4f`.

```bash
M=research.diagnostics.p1_upstream_faithful; L=outputs/reproduction/p1_upstream_faithful_2026-11-16/logs
.venv/bin/python -m $M --stage jobs | tac | systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  xargs -P 5 -L 1 sh -c '.venv/bin/python -m '$M' --stage train --system $0 --fraction $1 --arm $2 \
  --seed $3 --device cpu > '$L'/grid-$0-f$1-$2-s$3.log 2>&1'
.venv/bin/python -m $M --stage eval
```

## H2 diagnosis

Script `research/diagnostics/h2_diagnosis.py` (sha256 `87a7719915c08484`), tests
`tests/test_h2_diagnosis.py` (base arm bit-identical to the frozen baseline). Frozen config:
general head, LR .01 constant, 3000 updates, 32/8/16 trajectories, seeds 92–96. Arms: `base`
a ∈ {0,.5,1,2}; at a=2 `data128`, `latent12`, `latent24`, `g5` (+ g5 at a=0). D3: CartPole P1
`dense_s5`/`dense_full` 5% checkpoints, 10 000 random H15 windows (seed 20261117), |θ| bands.
Rules R1–R6 as preregistered in the [archive entry](docs/archive/TRACKING_2026-11-17_full.md#h2-readiness-diagnosis--preregistered-2026-11-17).

Result (`summary.json` sha256 `d3c12932c7fdc63a`): base response 0.0004 / 0.156 / 0.511 / 1.009;
excess over fixed-B floor ≈13% at a=2; data128 0.976, latent12 1.008, latent24 0.998; g5 0.460
(own scalar-gain floor 0.403). R1–R4, R6 hold; R5 holds → test label `NO_GO_CHEAP_FIX_G5`.
Scope note: R5 measures halving, weaker than RRD §5.4 ("G.5 fixes the failure"); G.5 still
misses the .05 level ~9×, so this is not a track decision. CartPole: `dense_s5` response error
0.008 upright → 3.199 hanging (sign flip); `dense_full` 0.424 → 0.762 (CB shrinks to the
sign-averaged gain, explaining its Table 1 miss).

```bash
M=research.diagnostics.h2_diagnosis; O=outputs/diagnosis/h2_diagnosis_2026-11-17
.venv/bin/python -m $M --stage data
.venv/bin/python -m $M --stage jobs | systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  xargs -P 4 -L 1 sh -c '.venv/bin/python -m '$M' --stage train --arm $0 --a $1 --seed $2 > '$O'/log-$0-a$1-s$2.txt 2>&1'
.venv/bin/python -m $M --stage eval
```

## H2 tests against G.5

Preregistered before running (full text in the [archive entry](docs/archive/TRACKING_2026-11-17_full.md#h2-tests-against-the-g5-comparator--preregistered-2026-11-17)).
Script `research/diagnostics/h2_g5_tests.py` (sha256 `99f5b581c2494254`), tests `tests/test_h2_g5_tests.py`.
T1 rotation a=2 base vs g5, seeds 102–106. T2 CartPole 5% `dense_full` + G.5, seeds 42–44, vs D3.
T3 unicycle v2 / T4 planar quadrotor v2 (held inputs; hover-centred thrust), latent n+4, LR
{.001,.01} per system×arm chosen on dev seeds 42–44 by validation H15 only, confirm 102–106;
response rows scaled by train std of ẋ. Rules: T1 residual = g5 > .05 and within 25% of its
scalar floor (≥4/5); T2 = g5 ≤ .5× dense_full in upright and hanging bands, all seeds;
T3/T4 = base disease (> .05, excess ≤ .25 floor, ≥4/5) and G.5 not fixing (> .05 on ≥4/5, mean
≥ .5× base). Verdict DISEASE_BEYOND_G5_CONFIRMED iff T1 ∧ (T3 ∨ T4) ∧ coverage p95 ≤ 3.

```bash
M=research.diagnostics.h2_g5_tests; O=outputs/diagnosis/h2_g5_tests_2026-11-17
run() { systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 xargs -P 4 -L 1 sh -c \
  '.venv/bin/python -m '$M' --stage train --job $0 $1 $2 $3 > '$O'/log-$0-$1-$2-$3.txt 2>&1'; }
.venv/bin/python -m $M --stage data
.venv/bin/python -m $M --stage jobs-dev | run
.venv/bin/python -m $M --stage select
.venv/bin/python -m $M --stage jobs-confirm | run
.venv/bin/python -m $M --stage eval
```

Result (57/57 complete; `summary.json` sha256 `0b69b2e4a90f3c4b`; selected LR .01 in all cells):
**NOT_CONFIRMED_BEYOND_G5**. T1 ✓ (base 0.916, g5 0.448 vs its floor 0.419). T2 ✗ — G.5 fixes
upright (0.424 → 0.043) but worsens hanging (0.762 → 0.972); pre-run prediction wrong. T3
unicycle: base 1.072 vs floor 0.873 (excess > 25% on ≥2 seeds → disease criterion not met), g5
0.584 (does not fix). T4 quadrotor uninformative: after row scaling the direction-dependent
ÿ row is 0.2% of E‖G‖², fixed-B floor 0.003; both models' errors are fitting errors. Coverage
≤ 2.59. Follow-ups need new preregistration: unicycle baseline calibration; quadrotor metric
that scores thrust-direction rows separately.

## H2 gate follow-up

**What and why.** The G.5 tests left the H2 gate open for three fixable reasons: the unicycle
baseline was not at its class floor (uncalibrated), the quadrotor metric was dominated by
constant torque rows, and G.5 on CartPole saw the unwrapped angle. This entry repairs each
and decides H2 readiness. Diagnostic only; no H2 method trained.
Script `research/diagnostics/h2_gate.py` (sha256 `06de7e0b450d9136`), test `tests/test_h2_gate.py`;
reuses `h2_g5_tests.py` unchanged. Data: v2 unicycle/quadrotor (regenerated per budget dir;
hashes equal the G.5-test data, e.g. unicycle train s42 `e7fdc518a8ef5362`).
- C1/C2: fixed-B and G.5 each get the same tuning grid LR {.003,.01} × updates {3000,12000}
  on dev seeds 42–44, selected by validation H15 only (fewest failures, lowest mean, then fewer
  updates, lower LR); confirm on new seeds 112–116. C1 scores all rows (as T3); C2 scores only
  the thrust-direction rows ÿ, z̈ (rows 3,4), scaled by train std of ẋ.
- C3: CartPole 5% `dense_full` + G.5 whose gain network gets cos θ in place of the unwrapped
  angle (labeled variant), seeds 42–44, 100 epochs, same windows/bands as D3.
Rules (fixed now): C1/C2 pass = fixed-B disease (> .05 and excess ≤ .25 floor on ≥4/5) ∧ G.5
does not fix (> .05 on ≥4/5, mean ≥ .5× fixed-B). C3 = cos-G.5 ≤ .5× `dense_full` in upright and
hanging bands, all seeds (informational: classifies CartPole as a scalar-gain case).
H2 readiness GO iff T1 (passed) ∧ (C1 ∨ C2) ∧ coverage p95 ≤ 3; else NOT_YET with reasons.
Predictions: C1 pass, C2 pass, C3 true.

```bash
M=research.diagnostics.h2_gate; O=outputs/diagnosis/h2_gate_2026-11-17
run() { systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 xargs -P 4 -L 1 sh -c \
  '.venv/bin/python -m '$M' --stage train --job $0 $1 $2 $3 $4 > '$O'/log-$0-$1-$2-$3-$4.txt 2>&1'; }
.venv/bin/python -m $M --stage data
.venv/bin/python -m $M --stage jobs-dev | run
.venv/bin/python -m $M --stage select
.venv/bin/python -m $M --stage jobs-confirm | run
.venv/bin/python -m $M --stage eval
```

Result (71 runs complete; `summary.json` sha256 `c2628c05525c97d3`): **H2 readiness NOT_YET.**
Selected (validation H15): unicycle fixed-B LR .003 / 12000 updates, G.5 .01 / 3000; quadrotor
both .01 / 12000. Replay control: the 12 dev runs at LR .01 / 3000 match the G.5-test dev runs
exactly (Δ val H15 = 0).
- C1 unicycle: fixed-B 1.071 ± 0.067 vs floor 0.937 (excess/floor .06/.17/.13/.34/.28 → 3/5 within
  25%, rule needs 4/5) → disease criterion fails narrowly; G.5 0.542 ≥ .5× fixed-B (does not fix;
  its scalar floor is 0.383). Calibration lowered seed spread (±0.18 → ±0.07) but not the mean.
- C2 quadrotor thrust rows: fixed-B floor only 0.009 — v2 hover data barely rotates the thrust
  direction, so the test is still insensitive. Both models fit poorly (fixed-B 0.95 ± 0.32,
  G.5 0.25 ± 0.20): a fitting problem on a near-constant input map, possibly the multi-input
  canonical structure; not evidence about H2.
- C3 CartPole cos-G.5: upright 0.036–0.055, hanging 0.85–0.88 vs `dense_full` 0.41–0.43 /
  0.75–0.78 → does not fix; the unwrapped-angle hypothesis is rejected. Prediction wrong.
- Post hoc (labeled, no rule change): CartPole's ẍ gain stays positive (0.91–0.98) while the θ̈
  gain flips sign (−1.46…+1.46), so the input *direction* changes between rows. A scalar gain
  on one fixed direction has normalized floor 0.052 / 0.072 / **0.908** in the upright / mid /
  hanging bands; G.5 sits at that structural floor. CartPole is therefore a candidate
  direction-change system; counting it needs a new preregistered test on fresh seeds.

## C4 CartPole confirmatory test

**What and why.** The gate follow-up found post hoc that CartPole's input direction changes
between rows (ẍ gain stays positive, θ̈ gain flips sign), so a scalar-gain G.5 cannot follow
it. This tests that claim on fresh models before it may count toward the H2 gate. Fourth
attempt to find a second system; the rule below is fixed before training.
Script `research/diagnostics/h2_cartpole_confirm.py` (sha256 `da3ee974e765d861`), test
`tests/test_h2_cartpole_confirm.py`. New locked configs
`configs/research/cartpole_controllable_win15_frac0.05_seed{45,46,47}.yml` (copies of seed 42,
only `run_id`/`seed`/`checkpoint` changed; new 500-trajectory subsets). Arms: fixed-B =
P1 `dense_full` harness; cos-G.5 = h2_gate C3 variant; 100 epochs each; validation = 200
trajectories outside the new subsets. Evaluation on the D3 windows (seed 20261117).
Floors (data only, computed before training), normalized per state: fixed-B best constant map
0.408 / 0.158 / 0.771, scalar-gain 0.052 / 0.072 / 0.908 (upright / mid / hanging).
Rule: C4 PASS iff in the hanging band every fixed-B seed is > .05 and ≤ 1.25× its floor, and
every cos-G.5 seed is > .05 and ≤ 1.25× its floor with mean ≥ .5× fixed-B. H2 readiness GO iff
T1 (passed) ∧ C4 PASS.

```bash
M=research.diagnostics.h2_cartpole_confirm; O=outputs/diagnosis/h2_cartpole_confirm_2026-11-17
.venv/bin/python -m $M --stage jobs | systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  xargs -P 3 -L 1 sh -c '.venv/bin/python -m '$M' --stage train --arm $0 --seed $1 > '$O'/log-$0-s$1.txt 2>&1'
.venv/bin/python -m $M --stage eval
```

Result (6/6 complete; `summary.json` sha256 `cf6970a703e833e8`): **C4 PASS, H2 readiness GO.**
Hanging band: fixed-B 0.717 / 0.713 / 0.801 (floor 0.771, ≤ 1.25×), cos-G.5 0.887 / 0.884 / 0.910
(floor 0.908; mean 0.894 ≥ .5× fixed-B 0.744). Upright: cos-G.5 0.015–0.030 vs fixed-B
0.398–0.457. Caveats: 3 seeds; fourth search for a second system; G.5 mid band 0.22–0.27
vs its floor 0.07 (not scored). [Report](outputs/diagnosis/h2_cartpole_confirm_2026-11-17/REPORT.md).

## Curated checkpoints (2026-11-17)

Tracked in Git (each with its `result.json` holding config, data hashes and checkpoint SHA-256);
everything else under `outputs/` is reproducible by the commands above and stays ignored.
- T1 rotation: `outputs/diagnosis/h2_g5_tests_2026-11-17/rotation/runs/{base,g5}-a2-s{102..106}/epoch300.pt`.
- C4 CartPole: `outputs/diagnosis/h2_cartpole_confirm_2026-11-17/{fixedB/cartpole-f0.05-dense_full,cartpole/cartpole-f0.05-g5cos-dense_full}-s{45,46,47}/checkpoint.pt`.
- P1 / D3 reference: `outputs/reproduction/p1_upstream_faithful_2026-11-16/runs/cartpole-f0.05-{dense_s5,dense_full}-s{42,43,44}/checkpoint.pt`.
