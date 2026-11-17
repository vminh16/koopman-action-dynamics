# H2 readiness diagnosis — 2026-11-17

Preregistered in TRACKING.md. 45 rotation runs (frozen general-head baseline, LR .01,
3000 updates each, seeds 92–96, final checkpoint, test split), plus CartPole P1-grid
checkpoints (no training). `summary.json` sha256 prefix `d3c12932c7fdc63a`; script `87a7719915c08484`.

## Rotation family (mean over 5 seeds, test)

| Arm | a | Response E‖G−r‖² | Fixed-B floor Var(G) | Excess | H15 / ignore-input | Coverage p95 |
|---|---:|---:|---:|---:|---:|---:|
| base | 0 | 0.0004 | 0.0000 | 0.0004 | 0.0009 | 0.64 |
| base | 0.5 | 0.156 ± 0.038 | 0.136 | 0.020 | 0.16 | 0.62 |
| base | 1 | 0.511 ± 0.101 | 0.449 | 0.062 | 0.54 | 0.62 |
| base | 2 | 1.009 ± 0.099 | 0.894 | 0.115 | **1.20** | 0.59 |
| data128 | 2 | 0.976 ± 0.030 | 0.894 | 0.082 | 0.98 | 0.22 |
| latent12 | 2 | 1.008 ± 0.098 | 0.894 | 0.114 | 1.35 | 0.59 |
| latent24 | 2 | 0.998 ± 0.087 | 0.894 | 0.104 | 1.35 | 0.59 |
| g5 | 0 | 0.0001 | 0.0000 | — | 0.0003 | 0.64 |
| g5 | 2 | 0.460 ± 0.037 | 0.894 | (below fixed-B floor) | 0.58 | 0.59 |

Scalar-gain floor of G.5 at a=2 (best fixed direction, per-state gain): 0.403.

**Preregistered verdict: `NO_GO_CHEAP_FIX_G5`.** R1 symptom ✓ (monotone, a=2 ≈ 2300× a=0),
R2 at fixed-B floor ✓ (excess ≈ 13% of floor), R3 data does not fix ✓, R4 latent does not
fix ✓, R6 coverage ✓, but R5 ✓: G.5 halves the response error (0.46 ≤ .5 × 1.01).

Interpretation (facts, no rule change):
1. The fixed-B disease is confirmed in isolation: the learned CB sits near the best constant
   map, error equals Var(G), and neither 4× data nor 2–4× latent dimension reduces it. At a=2
   the input term makes 15-step prediction worse than ignoring the input (ratio 1.20).
   Input error dominates drift error by ~50× (0.687 vs 0.0137).
2. The pre-run prediction "G.5 cannot help on rotation" was wrong as stated: a scalar
   state-dependent gain cannot rotate the input direction, but sign/scale changes capture
   ~55% of Var(G). G.5 is bounded by its own class floor (0.403); it reaches 0.46, so ≈0.40
   of response error (≈ 8× the .05 pass level) remains — a direction-rotation residual G.5
   cannot represent. Whether that residual justifies H2 needs a NEW preregistered test
   against the G.5 comparator; it does not overturn this verdict.

## Scope of the verdict (clarification added 2026-11-17, rules unchanged)

`NO_GO_CHEAP_FIX_G5` is the label of this single preregistered test, not an H2 track
decision. R5 as written ("G.5 ≤ .5× base") measures *partial improvement*; it is weaker than
RRD §5.4's NO-GO condition ("G.5 đã khắc phục failure mode"). G.5 at a=2 still fails the
.05 response pass level by ~9×, so the failure mode is reduced, not removed. Consequence:
the fixed-B disease is confirmed, G.5 becomes the stronger comparator any H2 claim must
beat, and the H2 gate stays open pending preregistered tests against G.5 on systems not yet
run in this protocol (rotation residual, unicycle, planar quadrotor, CartPole).

## CartPole (D3, P1 grid 5%, 10 000 random H15 windows, mean of seeds 42–44)

| Arm | Band \|θ\| | Normalized response error | H15 (wrapped) |
|---|---|---:|---:|
| dense_s5 (trained near upright) | [0, π/6) | 0.008 | 152 |
| | [π/6, π/2) | 0.679 | 206 |
| | [π/2, π] | 3.199 | 223 |
| dense_full (trained on all regimes) | [0, π/6) | 0.424 | 0.077 |
| | [π/6, π/2) | 0.173 | 0.147 |
| | [π/2, π] | 0.762 | 0.162 |

Supporting rule (hanging ≥ 1.5× upright, all seeds): PASS for both arms.
`dense_s5` has an accurate upright CB and a sign-wrong one when hanging (error ≈ 3.2,
close to 4 for exact sign flip); its H15 off-distribution diverges (extrapolation, not
attributable to the input map alone). `dense_full` compromises: its CB shrinks toward the
average of sign-flipped gains, so it is wrong even upright (0.42) — a mechanistic account
of why `dense_full` misses Table 1 (16.5×) while `dense_s5` matches it.
Since G.5 can flip sign, the CartPole symptom is exactly the kind G.5 may fix; not tested here.

Scope: 5 seeds, small synthetic budgets by design (diagnosis). No H2/H3 method trained,
historical gates unchanged.
