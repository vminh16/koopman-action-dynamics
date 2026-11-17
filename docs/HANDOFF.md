# Handoff — 2026-11-17

Read this first, then [AGENTS.md](../AGENTS.md). Scope and gates: [ROADMAP](../ROADMAP.md);
metric definitions: [EXPERIMENT](../EXPERIMENT.md); every run with commands and hashes:
[TRACKING](../TRACKING.md); theory: [RRD v0.5](RRD_v0.5.md).

## Where the project stands

| Phase | Status | One-line result |
|---|---|---|
| P0 | closed | Local Pendulum runs are real Koopman models. |
| P1 | closed, mismatch explained | Table 1 is reached by dense-P (`91a47d0`) + upstream start≤4 sampler in 7/8 cells; the released SPD operator cannot fit CartPole (input-sign restriction). Exact reproduction not certified. |
| P2 | closed | Stress systems built and audited; calibrated baseline frozen ([BASELINE](BASELINE.md)). |
| **H2** | **readiness GO** | Fixed-B fails when the input direction depends on state, and the author's G.5 scalar-gain encoder cannot fix it either: both sit at their class floors on rotation (T1) and CartPole hanging regime (C4, fresh seeds). |
| H3 | NO-GO | Certified correction margin too small. |

Plain-language core finding: the baseline assumes a push always acts in the same direction.
In CartPole, a push always accelerates the cart the same way but turns the pole the opposite
way once it hangs down. Fixed-B cannot represent this at all; G.5 can scale or flip the whole
push but not flip one row, so it fails on the same regime.

## Next step (not started)

Write and preregister **H2-01** before any code: H2-2 factorized (drift + action branches,
one latent) vs H2-3 matched-class monolithic control-affine vs G.5 vs fixed-B, matched
parameters/data/updates, selection on validation only; systems rotation a ∈ {0,2} and
CartPole 5% full-range windows; metrics = response error per band vs both floors, H15.
H2 is supported only if H2-2 beats H2-3. The user wants to review the H2-01 design before
implementation.

## Code map

| Path | What |
|---|---|
| `dynamics/`, `models/`, `training/`, `my_utils/`, `experiments/` | Upstream code. Only change: zero-weighted NaN term removed in `training/loss.py`. |
| `research/baseline/` | Explicit Koopman runner, data generator, checkpoint/data audits. |
| `research/benchmarks/` | Rotation family, unicycle, planar quadrotor (analytic `f0`, `G`). |
| `research/diagnostics/` | All diagnostics. Recent: `p1_upstream_faithful.py` (P1 grid), `h2_diagnosis.py`, `h2_g5_tests.py`, `h2_gate.py`, `h2_cartpole_confirm.py`. |
| `research/control/` | QP MPC. `research/methods/h2,h3/` are empty until method work starts. |
| `configs/research/` | Locked YAML; CartPole 5% seeds 45–47 were added for C4. |
| `docs/` | This file, BASELINE, RRD, paper PDF, `superpowers/` reports (paths are read by P2 scripts — do not move), `archive/` (full pre-2026-11-17 ROADMAP/EXPERIMENT/TRACKING, old RRD versions). |
| `outputs/` | Ignored except reports, `summary*.json`, selection files and curated checkpoints. |

## Pitfalls

- **Memory:** 15 GB RAM, no swap. Eight CUDA processes OOM-killed the desktop once. Run grids
  on CPU (`OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=`), 3–5 processes, inside
  `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`. CPU is ~3× faster than the
  RTX 2050 for these models. CUDA reports "unknown error" until the next reboot.
- **Loading checkpoints:** use the builder that made them — general head:
  `h2_diagnosis.make_model` / `p2_config_check.make_model(seed, general=True)`; dense-P:
  `p1_upstream_faithful.make_model`; cos-G.5: `h2_gate.cart_model`; multi-input:
  `h2_g5_tests.make_model`. Plain upstream `model(x0,u)` ignores the general head.
- **Do not edit** `p2_config_check.py`, `p2_drift_coverage.py`, `p2_robust_flow.py`,
  `h2_g5_tests.py`: later scripts and saved source snapshots depend on their hashes.
  Newer scripts reuse them by temporarily swapping module globals.
- Run every data-generation stage sequentially before parallel training (shared files).
- Quadrotor v2 data barely rotates thrust; the multi-input fixed-B model also fits poorly even
  there (open question about the canonical multi-input structure).
- My predictions were wrong three times in this round; keep preregistering rules before runs.

## Open issues

Unicycle narrowly misses the class-limit rule (3/5 seeds); CartPole dataset transition audit
fails at 1e-4 (float32 solver, documented); Table 1 evaluator contract unknown; no upstream
LICENSE seen (check before publishing).
