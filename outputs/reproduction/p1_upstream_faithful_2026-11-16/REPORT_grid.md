# P1 upstream-faithful full fraction grid — 2026-11-16

Preregistered in TRACKING.md. 96 runs (2 systems × 4 fractions × 4 arms × seeds 42/43/44,
100 epochs, CPU single-thread), final epoch, test H15 raw MSE on 1000 test trajectories
from x0. Pass rule per cell (fixed before running): 3-seed mean ≤ 2× paper.
Data: `summary_grid.json` (sha256 prefix `91f5e65f2040dc4f`); script prefix `9bf7ae531597077a`.

Cells: mean ± std (× paper; ✗ = fails ≤ 2×).

| System | Arm | 1% | 5% | 10% | 30% |
|---|---|---|---|---|---|
| Pendulum | spd_full | 0.55639 ± 0.49 (1.48×) | 0.086194 ± 0.043 (2.71× ✗) | 0.037152 ± 0.051 (3.26× ✗) | 0.006876 ± 0.00035 (1.13×) |
| Pendulum | spd_s5 | 0.56895 ± 0.19 (1.52×) | 0.035527 ± 0.0012 (1.12×) | 0.13778 ± 0.031 (12.09× ✗) | 0.014829 ± 0.00014 (2.43× ✗) |
| Pendulum | **dense_s5** | 0.35635 ± 0.046 (0.95×) | 0.018449 ± 0.00033 (0.58×) | 0.016267 ± 0.0021 (1.43×) | 0.0059723 ± 0.00023 (0.98×) |
| Pendulum | dense_full | 0.048962 ± 0.049 (0.13×) | 0.014726 ± 0.0024 (0.46×) | 0.009509 ± 0.0028 (0.83×) | 0.0070133 ± 0.00047 (1.15×) |
| CartPole | spd_full | 0.11335 ± 0.0056 (11.93× ✗) | 0.091796 ± 0.0026 (38.25× ✗) | 0.089155 ± 0.0015 (77.32× ✗) | 0.08361 ± 0.00068 (146.43× ✗) |
| CartPole | spd_s5 | 0.12243 ± 0.00049 (12.89× ✗) | 0.11401 ± 0.0028 (47.50× ✗) | 0.11079 ± 0.0011 (96.09× ✗) | 0.10756 ± 0.00033 (188.36× ✗) |
| CartPole | **dense_s5** | 0.031311 ± 0.047 (3.30× ✗) | 0.0015409 ± 0.00018 (0.64×) | 0.00098926 ± 8.7e-05 (0.86×) | 0.00011868 ± 6.7e-05 (0.21×) |
| CartPole | dense_full | 0.11189 ± 0.0087 (11.78× ✗) | 0.039613 ± 0.0027 (16.51× ✗) | 0.036708 ± 0.0006 (31.84× ✗) | 0.036322 ± 0.00087 (63.61× ✗) |

**Verdict (preregistered rule):** `dense_s5` passes 7/8 cells; fails **CartPole 1%** (3.30×).
Therefore NOT `REPRODUCED_WITH_LABELED_CODE_VARIANT` for the full grid; the failing cell is
CartPole 1%. Per seed: s42 `0.0024`, s44 `0.0055` (both under paper), s43 `0.086` with LR
collapsed to 1e-8 and angle input gain −0.24 (physical ≈ −1.46). At 1% there are 500 windows,
8 updates/epoch, 800 in total; upstream last-batch ReduceLROnPlateau decays LR before the input
column is learned (same mechanism as the 5% winfrac failures). Not rerun or seed-selected.

Findings:
1. The released block-SPD operator fails CartPole at every fraction and sampler, and the gap
   grows with data (12× → 146×): consistent with the sign-restricted input column, not with
   data scarcity.
2. Dense-P alone (`dense_full`) does not suffice on CartPole (12–64×); dense-P + start≤4
   sampler is required. On Pendulum dense-P passes with either sampler.
3. Replay control: all 18 shared 5% runs reproduce the CPU 5% check (`summary.json`) with
   test H15 difference 0.0. Output rank equals n in all runs.
4. Pendulum 1% passes for every arm only because the paper value (0.3747) is large; per-seed
   spread is wide (spd_full 0.16–1.10).

Scope: diagnostic, 3 seeds, final epoch, Euler harness of P1. Table 1 evaluation contract
still not confirmed by authors; CartPole dataset transition audit (RK4 1.6e-4 > 1e-4) is
untouched by this grid. C02/P1 evidence, P2 and H2/H3 gates unchanged.
