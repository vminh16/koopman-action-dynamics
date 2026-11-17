# C4 CartPole confirmatory test — 2026-11-17

Preregistered in TRACKING.md. Fresh locked configs (seeds 45–47, new 500-trajectory subsets),
6 runs × 100 epochs, CPU; D3 evaluation windows. `summary.json` sha256 `cf6970a703e833e8`.
**Verdict: C4 PASS → H2 readiness GO** (with T1 rotation residual).

Normalized response error per band (floors computed from data before training):

| | upright [0, π/6) | mid [π/6, π/2) | hanging [π/2, π] |
|---|---:|---:|---:|
| fixed-B floor (best constant map) | 0.408 | 0.158 | **0.771** |
| fixed-B s45 / s46 / s47 | 0.449 / 0.457 / 0.398 | 0.188 / 0.194 / 0.157 | **0.717 / 0.713 / 0.801** |
| scalar-gain floor (G.5 class) | 0.052 | 0.072 | **0.908** |
| cos-G.5 s45 / s46 / s47 | 0.024 / 0.030 / 0.015 | 0.253 / 0.271 / 0.219 | **0.887 / 0.884 / 0.910** |

Both model classes sit at their own floors in the hanging band on fresh seeds: fixed-B cannot
follow the state-dependent input, and G.5 fixes the upright regime but not the row-wise
direction change (ẍ gain stays positive while θ̈ gain flips sign). The disease beyond G.5 is
now shown on two systems (rotation, CartPole).

Caveats: 3 seeds; this was the fourth attempt to find a second system (the hypothesis came from
a post-hoc look at C3, then was tested on fresh models with a fixed rule); G.5 is above its
floor in the mid band (0.22–0.27 vs 0.07), which the rule does not score.
