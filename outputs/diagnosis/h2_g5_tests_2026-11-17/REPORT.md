# H2 tests against the G.5 comparator — 2026-11-17

Preregistered in TRACKING.md. 57 runs (rotation 10, CartPole 3, unicycle/quadrotor 24 dev + 20
confirm), CPU, final checkpoints, oracle G(x) for evaluation only. `summary.json` sha256 prefix
`0b69b2e4a90f3c4b`; script `99f5b581c2494254`. LR selected on validation H15: .01 for all four
system×arm cells.

**Preregistered verdict: `NOT_CONFIRMED_BEYOND_G5`** (needs T1 and one of T3/T4; T3/T4 failed
the class-limit criterion).

| Test | Fixed-B (base) | G.5 | Floors (fixed-B / scalar) | Rule result |
|---|---|---|---|---|
| T1 rotation a=2 | 0.916 ± 0.042 | 0.448 ± 0.036 | 0.874 / 0.419 | **residual confirmed**: G.5 at its own floor, 9× above .05 |
| T2 CartPole, response error by band | upright 0.424, hanging 0.762 | upright **0.043**, hanging **0.972** | — | **G.5 does not fix** the sign flip (prediction was wrong) |
| T3 unicycle v2 | 1.072 ± 0.181 | 0.584 ± 0.079 | 0.873 / 0.351 | disease criterion **not met** (base above floor by ~23%); G.5 does not fix |
| T4 quadrotor v2 | 0.724 ± 0.616 | 0.355 ± 0.048 | 0.003 / 0.003 | **uninformative** (see below) |

Coverage p95 ≤ 2.59 everywhere. CartPole wrapped H15: G.5 0.041 / 0.123 / 0.139 vs
`dense_full` 0.077 / 0.147 / 0.162 (upright / mid / hanging).

## What this means

1. **Rotation:** the direction-rotation residual survives G.5 on new seeds. Fixed-B sits at its
   class floor; G.5 sits at its own floor (0.448 vs 0.419). Neither is an optimization failure.
2. **CartPole:** G.5 learns the upright gain (error 0.424 → 0.043) but gets the hanging regime
   *worse* (0.762 → 0.972), so it does not resolve the sign flip in practice. One untested
   explanation: G.5's gain network sees the raw, unwrapped angle (up to ±36 rad), not cos θ.
3. **Unicycle:** both models are far from correct (> .05 by 10–20×) and G.5 removes only ~45%.
   But the fixed-B baseline is ~23% *above* its class floor, so the strict "pure model-class
   limit" criterion fails: part of its error is fitting. The baseline was calibrated on
   rotation only and was not recalibrated for this system.
4. **Quadrotor:** the test cannot answer the question. After row scaling, the constant torque
   row carries 76% of E‖G‖² and the direction-dependent ÿ row only 0.2%, so the fixed-B floor
   is 0.003. Both models' large errors are fitting errors on near-constant rows (base varies
   0.2–1.6 across seeds). This is a metric/data design flaw, not evidence for or against H2.

## Pre-run predictions vs outcome

T1 yes → yes. T2 "G.5 fixes" → **no**. T3/T4 "G.5 does not fix" → yes for T3; T4 not measurable.

Next steps need new preregistrations (not reruns of this one): recalibrate the baseline on
unicycle (positive control as for rotation), and redesign the quadrotor metric to score the
thrust-direction rows separately.
