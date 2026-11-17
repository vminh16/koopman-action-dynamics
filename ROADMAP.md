# ROADMAP — `koopman-action-dynamics`

**Version:** 0.6 · 2026-11-17 · theory [RRD v0.5](docs/RRD_v0.5.md) · protocol [EXPERIMENT](EXPERIMENT.md) · evidence [TRACKING](TRACKING.md)
**Baseline paper:** Miao et al., *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026 ([local PDF](docs/paper/20675_Learning_Koopman_Represe.pdf); upstream code audited at `ec131bb`).
**Frozen comparator:** general-head fixed-B Koopman ([BASELINE](docs/BASELINE.md)); the author's G.5 input encoder is now a second comparator for H2.
Full v0.5 text (upstream audit tables, repository plan, old queue): [archive](docs/archive/ROADMAP_2026-11-17_full.md).

> **Do not drift.** RRD defines hypotheses, EXPERIMENT defines measurements, ROADMAP defines order and gates. H2 and H3 are independent tracks. Until a gate opens, only the author's Koopman method (and labeled diagnostic ablations of it, including G.5) is trained; no external baselines, no H2/H3 methods.

## Code boundaries

- Upstream code stays in `dynamics/`, `models/`, `training/`, `my_utils/`, `experiments/`, unchanged except the recorded `training/loss.py` NaN fix (zero-weighted condition-number term removed).
- Research code: `research/baseline/` (runners, audits), `research/benchmarks/` (rotation, unicycle, planar quadrotor), `research/diagnostics/` (all P1/P2/H2 diagnostics), `research/control/` (QP MPC). Tests in `tests/`.
- Generated data/runs stay ignored under `outputs/`; only reports, summaries and curated checkpoints are tracked.

## Phases

| Phase | Status | Outcome | Evidence |
|---|---|---|---|
| P0 Pendulum forensic audit | **CLOSED** | Local runs are real `KoopmanLitModule`; lambda_control drift recorded, not the cause; legacy generation seeds unknown. | [archive §P0](docs/archive/ROADMAP_2026-11-17_full.md#p0--local-pendulum-forensic-audit-completed-mismatch) |
| P1 reproduction | **CLOSED (mismatch explained)** | Table 1 scale is reached by dense-P (`91a47d0`) + upstream start≤4 sampler in 7/8 cells (fails CartPole 1%, one LR-collapsed seed); released block-SPD fails all CartPole cells. Exact reproduction NOT_CERTIFIED (evaluator contract unknown; CartPole transition audit fails by float32 solver error, documented). | [grid report](outputs/reproduction/p1_upstream_faithful_2026-11-16/REPORT_grid.md) |
| P2 baseline diagnosis | **CLOSED (2026-11-17)** | Stress systems implemented and audited; calibrated baseline frozen; fixed-B disease isolated (see H2). Small budgets by design. | [P2 report](outputs/diagnosis/p2_followup_2026-11-16/REPORT.md), [BASELINE](docs/BASELINE.md), [H2 diagnosis](outputs/diagnosis/h2_diagnosis_2026-11-17/REPORT.md) |
| Gate H2 | **READINESS GO (2026-11-17)** | Fixed-B disease beyond G.5 shown by preregistered rules on rotation (T1) and CartPole (C4, fresh seeds). | [C4 report](outputs/diagnosis/h2_cartpole_confirm_2026-11-17/REPORT.md) |
| Gate H3 | **NO-GO** | ρ_T ≈ 0.0068 but d_orient,p95 ≈ 0.12–0.13 > ρ_T; no certified correction window. Not revisited. | [archive §H3](docs/archive/ROADMAP_2026-11-17_full.md#gate-h3--independent-control-theorycertificate-hypothesis) |
| P3-H2 / P3-H3 | Not started | Method work only after the respective gate. | RRD §5–6 |
| P4–P5 | Not started | Independent reports; combination only if both tracks are supported. | — |

## Gate H2 — current evidence

1. **Disease confirmed (2026-11-17, rotation family).** Fixed-B response error grows 0.0004 → 1.009 for a = 0 → 2, sits at the best-constant-map floor, and is not reduced by 4× data or 2–4× latent dimension. On CartPole the same mechanism appears as a sign-flipped input gain when the pole hangs. [Report](outputs/diagnosis/h2_diagnosis_2026-11-17/REPORT.md).
2. **G.5 halves it but cannot rotate the input direction** (0.46, own floor 0.40). The preregistered label of that single test was `NO_GO_CHEAP_FIX_G5`; it is not a track decision.
3. **Tests against G.5** (rotation residual, CartPole, unicycle, planar quadrotor): see [G.5 tests](outputs/diagnosis/h2_g5_tests_2026-11-17/REPORT.md). Preregistered verdict `NOT_CONFIRMED_BEYOND_G5`: the rotation residual survives G.5 (G.5 at its own floor); G.5 does **not** fix CartPole's sign flip (better upright, worse hanging); unicycle fails badly for both models but the fixed-B baseline is not at its class floor (uncalibrated there); the quadrotor test was insensitive by design.

GO for P3-H2 additionally requires a design where factorized learning (H2-2) can be compared with a matched-class monolithic control-affine model (H2-3) and with G.5, at matched data/compute (RRD §5.4).

## Next

**Gate H2 readiness: GO (2026-11-17).** [C4](outputs/diagnosis/h2_cartpole_confirm_2026-11-17/REPORT.md) confirmed on fresh CartPole seeds that both fixed-B (0.71–0.80 vs floor 0.77) and G.5 (0.88–0.91 vs floor 0.91) sit at their class floors in the hanging band. With T1 (rotation) that is two systems. Unicycle narrowly missed (3/5); quadrotor v2 is insensitive. Caveat: fourth search for a second system, 3 seeds.

Next — P3-H2 (each step preregistered before training):
1. Write the H2-01 protocol: H2-2 factorized (drift branch + action branch, one latent state) vs H2-3 matched-class monolithic control-affine (shared trunk producing `a(z)`, `D(z)`) vs G.5 vs fixed-B; matched parameters/FLOPs, data, optimizer, updates and selection on validation only (RRD §5.3–5.4).
2. Systems: rotation a ∈ {0, 2} and CartPole 5% (full-range windows); metrics: response error by band against both floors, H15, and later closed-loop cost (H2d).
3. Decision: H2 supported only if H2-2 beats H2-3 (not just fixed-B/G.5); if H2-3 matches it, report NO-GO for factorization as the mechanism.

## Risks

| Risk | Action |
|---|---|
| Comparator too weak (SPD head, untuned) | Use general head + G.5, equal tuning budgets, selection on validation only. |
| Excitation/coverage too narrow | Report coverage; change data only as a new versioned dataset. |
| Response gain does not reach MPC | Measure closed-loop cost/regret separately (H2d); do not infer it. |
| Source license unclear for publication | No upstream LICENSE seen; clarify before publishing derived code. |

## Decision log (condensed)

- 2026-11-12 — v0.5: overlay layout, H2/H3 independent, no external baselines before diagnosis.
- 2026-11-13/14 — P0 closed; P2 run with the official module, Gate P2 PASS; old H2 ratio 1.030 later shown to be an under-fitted-baseline artifact.
- 2026-11-15 — P1 causal batch (28 runs): encoder losses/scheduler and the SPD sign restriction matter.
- 2026-11-16 — P2 calibration: general head LR .01 and compatible SPD LR .001 frozen as positive controls; P1 grid explains the Table 1 gap.
- 2026-11-17 — H2 diagnosis confirms the fixed-B disease; G.5 adopted as comparator; P2 closed; docs shortened (full texts in `docs/archive/`).
- 2026-11-17 — C4 passes on fresh seeds → H2 readiness GO. RRD moved to `docs/`, old versions to `docs/archive/rrd/`; `docs/HANDOFF.md` added.
