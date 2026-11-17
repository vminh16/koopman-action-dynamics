# P1 upstream-code-faithful check — 2026-11-16

Preregistered in TRACKING.md. 30 runs (2 systems × 5 arms × seeds 42/43/44, 100 epochs),
final epoch, test H15 raw MSE on 1000 test trajectories from x0. Pass rule fixed before
running: 3-seed mean ≤ 2× paper.

| System | Arm | Operator | Sampler | Test H15 mean ± std | × paper | Learned CB[last] |
|---|---|---|---|---:|---:|---:|
| Pendulum | P1 baseline (ref) | block SPD | full range | 0.0862 ± 0.0434 | 2.71 | — |
| Pendulum | spd_s5 | block SPD | starts 0..4 | 0.0355 ± 0.0012 | 1.12 PASS | ≈1.97 |
| Pendulum | spd_s5_winfrac | block SPD | public code | 0.1847 ± 0.1260 | 5.81 | 1.78 / 0.33 / 0.34 |
| Pendulum | dense_s5 | dense P | starts 0..4 | 0.0184 ± 0.0003 | 0.58 PASS | ≈2.97 (true 3) |
| Pendulum | dense_s5_winfrac | dense P | public code | 0.0840 ± 0.1140 | 2.64 | 2.96 / 0.64 / 3.01 |
| Pendulum | dense_full | dense P | full range | 0.0147 ± 0.0024 | 0.46 PASS | ≈2.92 |
| CartPole | P1 baseline (ref) | block SPD | full range | 0.0918 ± 0.0026 | 38.2 | >0 |
| CartPole | spd_s5 | block SPD | starts 0..4 | 0.1140 ± 0.0028 | 47.5 | ≈+0.18 |
| CartPole | spd_s5_winfrac | block SPD | public code | 0.1122 ± 0.0025 | 46.8 | ≈+0.19 |
| CartPole | dense_s5 | dense P | starts 0..4 | 0.00154 ± 0.00018 | 0.64 PASS | ≈−1.48 (true ≈−1.46) |
| CartPole | dense_s5_winfrac | dense P | public code | 0.00170 ± 0.00032 | 0.71 PASS | ≈−1.48 |
| CartPole | dense_full | dense P | full range | 0.0396 ± 0.0027 | 16.5 | ≈−0.30 |

Findings: (1) CartPole needs BOTH the dense-P operator of upstream commit 91a47d0 (sign-free
input column) AND the upstream start≤4 sampler; either alone fails. (2) The released block-SPD
operator cannot reach CartPole Table 1 under any sampler tried. (3) Pendulum: the public SPD
operator with starts 0..4 matches the paper (1.12×); failures in winfrac arms are seed-dependent
LR collapse (step-level ReduceLROnPlateau → 1e-8) with poorly fitted CB. All runs: output rank n.
s5 arms use fewer updates (2000/4000) than full-range (7900/15700). Diagnostic only; C02/P1
evidence and H2/H3 gates unchanged. Table 1 evaluation contract still not pinned by authors.
