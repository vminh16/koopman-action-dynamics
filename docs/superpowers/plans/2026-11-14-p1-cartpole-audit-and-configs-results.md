# CartPole audit and locked configurations: execution results

Plan: [2026-11-14-p1-cartpole-audit-and-configs.md](2026-11-14-p1-cartpole-audit-and-configs.md).

Implemented the audit engine, 12 locked `.yml` files (four fractions, three
seeds), and structure, corruption, configuration, operator and dry-run tests.
The runner uses the existing author module and rejects effective hyperparameter
drift; no training or dataset regeneration was performed.

## Actual audit result

**FAILED at the planned absolute transition tolerance `1e-4`.** Do not certify
these data as passing the mathematical audit. The report is saved at
`outputs/reproduction/cartpole_data_audit.yaml`; the CLI exits with code 1.

- Train: 10,000 trajectories, 2,500,000 transitions checked; maximum local RK4
  error `0.0001607612153478044`.
- Test: 1,000 trajectories, 250,000 transitions checked; maximum local RK4
  error `0.00008444161522547233` (passes).
- Shapes, finite values, actuator bounds, initial state bounds and
  mutation/autograd checks pass. Minimum train/test initial-state distance is
  `0.00019525498711533363`; no state-trajectory hash collisions.
- Both dataset SHA-256 values match the plan and locked configurations.

At the worst train transition, the recorded predecessor is approximately
`[0.9891561, -0.1212785, 32.6944962, 7.0126972]`. Two RK4 half-steps yield error
`0.000160762378`, essentially unchanged. This rules out a simple explanation
based on the single RK4 step's truncation error; it does not establish the
original data generation error's cause.

## Decisions

- Preserve data, hashes and `tol=1e-4`; report the observed failure rather than
  widen the tolerance or regenerate the locked dataset. These configs reproduce
  the existing data, but the integrity gate remains unresolved.
- Inspect `module.model.operator.core` in the operator test: the plan's
  `.operator` is the wrapper, rather than the actual canonical operator.
- Report empirical actuator moments descriptively; do not demand exact
  population moments from finite random samples.
- Work on a new branch in the existing checkout, preserving pre-existing P2
  changes. Only CartPole task files are included in the commit.

## Verification

`PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl .venv/bin/python -m unittest discover tests -v`:
Final run after the review fix: 79 tests passed in 21.763 seconds. This includes the actual complete-data audit
failure regression and CLI failure-report/exit-code checks. The implementation
suite passing does not change the dataset audit's failed verdict.

## Independent code review

### Standards

Zero actionable findings. Research-owned implementation stays in `research/`,
with upstream modules unchanged; locked standalone YAML files follow the
existing provenance convention.

### Spec

One runtime contract gap found and fixed: an edited CartPole config specifying
`latent_dim=16` could previously pass the effective checker if the module was
also built with 16. The checker now requires `latent_dim=25` for the CartPole
baseline. The regression test was observed failing before the guard and passing
afterward (four focused config tests pass). The full suite then passed 79/79.
No deferred minor findings.
