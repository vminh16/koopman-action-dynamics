# H2 gate follow-up — 2026-11-17

Preregistered in TRACKING.md. 71 runs (48 dev tuning, 20 confirm, 3 CartPole), CPU, final
checkpoints, oracle G(x) for evaluation only. `summary.json` sha256 `c2628c05525c97d3`.
**Preregistered verdict: H2 readiness `NOT_YET`.**

| Check | Fixed-B | G.5 | Floors (fixed-B / scalar) | Rule |
|---|---|---|---|---|
| C1 unicycle (calibrated: LR/updates tuned per arm) | 1.071 ± 0.067 | 0.542 ± 0.070 | 0.937 / 0.383 | disease criterion 3/5 seeds (needs 4/5) ✗; G.5 does not fix ✓ |
| C2 quadrotor, thrust rows only | 0.949 ± 0.321 | 0.248 ± 0.197 | 0.009 / 0.007 | insensitive: v2 hover data barely rotates thrust |
| C3 CartPole, G.5 fed cos θ (hanging band) | 0.746–0.778 | 0.846–0.875 | — / 0.908 (post hoc) | does not fix ✗ (prediction wrong) |

Replay control: dev runs shared with the G.5 tests reproduce them exactly.

## Reading

- **Unicycle** is the closest second system: both models are far from correct (> .05 by
  10–20×) and G.5 only halves the error, but fixed-B stays 6–34% above its constant-map floor
  on new seeds, so the strict "pure class limit" rule fails 3/5 vs 4/5.
- **Quadrotor v2** does not stress the input direction (floor 0.009). Its large errors are
  fitting failures on an almost constant input map; worth a separate look at the multi-input
  canonical structure, not at H2.
- **CartPole**: feeding cos θ did not help, so the unwrapped angle is not the cause. Post hoc
  (labeled): the ẍ gain stays positive while the θ̈ gain flips sign, so the input direction
  rotates between rows. A scalar gain on one fixed direction cannot follow it: normalized
  floor 0.908 in the hanging band, where all six G.5 models (two variants × three seeds)
  sit at 0.85–0.97. This is a candidate second system, but it was found after looking at
  the data and needs a fresh preregistered test before it counts.

Evidence so far: the fixed-B disease is real (rotation, CartPole, unicycle), and a residual
beyond G.5 is established on rotation and strongly indicated, not yet confirmed by rule, on
CartPole and unicycle.
