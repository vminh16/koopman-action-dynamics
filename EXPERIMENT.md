# EXPERIMENT — Protocol & Evidence Registry

**Version:** 0.6 · 2026-11-17 · theory [RRD v0.5](docs/RRD_v0.5.md) · order/gates [ROADMAP](ROADMAP.md) · runs/replay [TRACKING](TRACKING.md)
**Reference:** Miao et al., ICLR 2026 ([local PDF](docs/paper/20675_Learning_Koopman_Represe.pdf)); upstream code `ec131bb`.
**Evidence status:** P0 CLOSED; P1 CLOSED (mismatch explained, exact reproduction NOT_CERTIFIED); P2 CLOSED; H2 readiness GO (disease beyond G.5 on rotation and CartPole); H3 NO-GO.
Full v0.5 text (P0 checklist, P1 records, status logs, templates): [archive](docs/archive/EXPERIMENT_2026-11-17_full.md).

> **Invariant.** Until a gate opens, only the ICLR 2026 model (and labeled diagnostic variants such as G.5) is trained, on original and stress data. Simulator oracles (true `f`, `G_x`, same-state resets) are for **evaluation only**, never for training or model selection. Learned-surrogate controllability is not a plant guarantee.

## 1. Questions

| Question | Evidence | Stage / status |
|---|---|---|
| E0 Was the Pendulum run a structured Koopman model? | provenance, `state_dict`, config | P0 — yes |
| E1 Is Table 1 reproducible? | fraction grid, provenance | P1 — with labeled variant (dense-P + start≤4), 7/8 cells |
| E2–E4 Does fixed-B mis-predict action response when the input map depends on state? | same-state response, `CB` vs `G_x`, rotation sweep, coverage | P2/H2 — yes on rotation and CartPole |
| E5 Is the Koopman margin useful for certificates? | `ρ_T`, `d_orient`, `d*` | H3 — no (NO-GO) |
| E6 Does factorized learning beat a matched-class model? | H2-2 vs H2-3 vs G.5 | P3-H2, after gate |
| E7 Certified and useful for planning? | certified vs free, QP/NMPC | P3-H3, after gate |

Claims hierarchy: (a) algebraic property; (b) prediction; (c) counterfactual fidelity; (d) MPC usefulness; (e) deployment. Never infer (d)/(e) from (a)/(b).

## 2. P0/P1 findings that constrain later work

- Upstream `run_train.py` builds an MLP; research runners instantiate `KoopmanLitModule` explicitly. Upstream sampler limits window starts to 0..4; the paper text says random windows over the trajectory. Table 1 is reached only with the code sampler.
- The released block-SPD operator restricts the sign of the physical input column; CartPole's angular input gain is negative, so this operator cannot fit it. Use sign-free heads (general head, dense-P) as comparators.
- Table 1 H15 from `x0` evaluates only the near-upright CartPole regime (input gain −1.46…−1.39); over a full trajectory the gain changes sign (−1.46…+1.46).
- Angles are stored unwrapped; raw-MSE long horizons mix angle wrap with dynamics error. Pendulum's input map is constant, so Pendulum cannot test fixed-B limitations.

## 3. Data

| ID | System | Role | Status |
|---|---|---|---|
| D-PEND | Pendulum 5000/1000 × 250, dt .02 | reproduction | hashed; legacy generation seed unknown |
| D-CART | CartPole 10000/1000 × 250, dt .02 | reproduction | hashed; transition audit fails at 1e-4 (float32 solver, documented) |
| D-ROT | rotation family `G=[cos ax₁, sin ax₁]`, 8–128 traj × 50, RK4 | stress (H2 main) | versioned per protocol in `outputs/diagnosis/*` |
| D-UNI v1 / D-QUAD v1 | unicycle, planar quadrotor, 30 × 50, i.i.d. uniform inputs | historical P2 | weak excitation (unicycle barely moves; quadrotor tumbles) |
| D-UNI v2 / D-QUAD v2 | same plants; held inputs (ω ±3), hover-centred thrust mg/2 ± .5 N | H2 G.5 tests | heading spread ≈1 rad, tilt ±1.2 rad |

Protocols: **A observational** (primary); **B controlled intervention** (optional after H2 gate, equal budget for all methods); **C shift** (future). Changing data means a new versioned dataset, never overwriting a path whose results were read.

## 4. Metrics — precise definitions and oracle policy

### 4.1. Prediction fidelity

For fixed held-out trajectories with x0 and physical input sequence:

\[
\mathrm{MSE}_{H}=\frac{1}{NHn}\sum_{i=1}^{N}\sum_{k=1}^{H}\|\hat x^{(i)}_k-x^{(i)}_k\|_2^2.
\]

Report one-step (with re-encode **every** observed state as separate teacher-forced evaluation), free-running rollout MSE at H, and periodic re-encoding separately. Do not call re-encoding an unconstrained open-loop prediction. Report per-state standardized MSE, raw-state MSE and error-vs-horizon curve; do not mix angles without explicit wrapping convention.

### 4.2. Physical same-state action-response and counterfactual accuracy

With dynamics oracle available **only for diagnosis**:

\[
\Delta_f(x;u,u_0)=f(x,u)-f(x,u_0),\qquad
\widehat\Delta_f(x;u,u_0)=C\{f_z(\phi(x),u)-f_z(\phi(x),u_0)\}.
\]

\[
E_{\rm response}=\frac1K\sum_i\|\widehat\Delta_f(x_i;u_i,u_{0i})-\Delta_f(x_i;u_i,u_{0i})\|_2^2.
\]

For a linear fixed-B Koopman surrogate, `f_z(z,u)=Az+Bu`, so `\widehat\Delta_f=CB(u-u0)`; **not** an assertion that rollout action effects are independent of initial state after model re-encoding. Report normalized error with epsilon denominator, action-pair distribution, and confidence intervals. For comparisons across controls, use matched action budget and same `x` with simulator reset **evaluation only**.

### 4.3. Lifted-map and manifold consistency (diagnostics, not training supervision)

For control-affine true systems `f(x,u)=f_0(x)+G_x(x)u`, at valid `x`:

\[
G_z^\star(x)=J_\phi(x)G_x(x),\quad
E_{\rm lift}(x)=\|B-G_z^\star(x)\|_F,\quad
E_{\rm drift}(x)=\|A\phi(x)-J_\phi(x)f_0(x)\|_2.
\]

On predicted latent state `z`: `E_manifold(z)=||z-φ(Cz)||`. Report both physical `||CB-G_x||` and lifted errors; even when physical `G_x` constant, `JφG_x` can vary. Record nondifferentiability handling of ReLU encoder (`Jφ` exists a.e.); benchmark oracle units and state support. Never equate a sampled maximum with a global supremum.

### 4.4. Controllability diagnostics and H3 feasibility

For a frozen continuous-time `(A,B,C)` and declared horizon `T`:

\[
W_y(T)=\int_0^T Ce^{As}BB^\top e^{A^\top s}C^\top ds,
\quad Q_y(T)=\int_0^T Ce^{As}e^{A^\top s}C^\top ds,
\]
\[
\rho_T=\sqrt{\lambda_{\min}(W_y(T))}/\sqrt{\lambda_{\max}(Q_y(T))}.
\]

Log `lambda_min`, `cond`, `rho`, numerical quadrature/grid, float64-vs-default, horizon and Gramian physical output convention. Avoid uncontrolled `T=15*dt` overwrite in upstream loss during analysis: use independent implementation or explicitly record the effective horizon. For a candidate state-dependent lifted map `G_z^*(x)`, record:

\[
d_{\rm orient}(x)=\|(I-\Pi_{G_z^\star(x)})B\|_2
\]

(with full-rank or pseudoinverse convention stated). Orientation metric is a **necessary diagnostic**, not a sufficient H3 certificate. `d_star` is the *constrained factorization optimization* defined in RRD §6, dependent on pre-declared singular-value bounds on `M`; numerical output is not an exact global feasibility proof. **Do not use `χ=r_A/ρ_T` as a GO/NO-GO rule.**

**P2 verification conventions (2026-11-14):**
- `d_star` keeps the sign of `M`: for single-input systems it is solved exactly over both sign branches (`t=1/M`), so a negative factor is not silently discarded. Multi-input systems report the proven least-squares lower bound and the feasible SLSQP upper bound separately, plus solver status; neither is a global certificate.
- `d_orient_max` and `d_orient_p95` are distinct keys. Gate comparisons use the p95 key; the max is reported for tail inspection only. Reporting a max under a p95 key is a protocol bug.
- `ρ_T` and the factorization bounds use the checkpoint `dt` and the *same* MPC horizon (`T = H_mpc · dt`); the continuous-time Gramian horizon must not be `15·dt` by accident. The upstream loss horizon is recorded separately as an effective hyperparameter.
- The Gramian is computed exactly with a float64 Van Loan block exponential (`scipy.linalg.expm`); a trapezoidal quadrature is retained only as an integration-sensitivity check, not as the reported value.
- The upstream `controllability_loss` carried a `beta=0` condition-number term. It contributed zero to the value but its backward pass divided by a numerically zero minimum eigenvalue, producing NaN gradients after ~230 full-budget steps (observed on planar quadrotor seed 45). The zero-weighted term was removed; the returned value is unchanged.



### 4.5. Control utility, candidate ranking and solver efficiency

For fixed candidate pool `\mathcal A` and true/evaluated costs from held-out simulator:

\[
R_{\rm cand}=J(\arg\min_{A\in\mathcal A}\hat J(A))-\min_{A\in\mathcal A}J(A).
\]

This is *candidate-set regret*, not global regret. Report closed-loop goal success/cost, RMS tracking error, physical input energy, total constraint violations and state constraint violations. Explicitly label solver regime:

1. **Exact baseline QP:** lifted LTI and affine constraints under convex quadratic cost; only here assert global convex optimum under stated assumptions.
2. **Frozen-scheduling QP:** QP on a frozen predicted trajectory; **approximation** to state-dependent dynamics, not globally optimal nonlinear MPC.
3. **NMPC:** SQP/shooting/direct collocation for nonlinear model; report local convergence, initialization, timing and failures.

Report wall-clock latency (median, P95), solver iterations, success/timeouts, objective/constraint tolerances, physical-action bounds, MPC horizon/`dt`, hardware. Compare under equal action bounds and evaluate plant with same dynamics and seeds. **Do not claim controller-reproduction complete:** upstream `experiments/run_control.py` is empty.

**P2 verification conventions (2026-11-14):**
- Candidate ranking uses one time-varying pool generated from the held-out simulator, identical for every model at that `(a, seed)`; each MIMO input axis is excited. The learned model encodes that pool once and rolls out the latent LTI solver; the RK4 plant is compared on the same states, action sequence and target. Re-encoding is measured separately and never folded into the free-running score.
- E08 handles solver failure, stale solutions, residuals and out-of-bounds actions explicitly. When the applied action violates bounds it is projected into `[umin, umax]`; zero is a valid fallback command only if `0 ∈ [umin, umax]`. Raw and applied violations, residuals, solver status, latency and fallback count are logged.
- The same E08 check compares exact ZOH and Euler discretization of the learned lifted step and records the training-time Euler convention, so a discretization effect cannot be mistaken for a modeling effect.
- E07 runs a fixed initial-state sweep for Unicycle and Planar Quadrotor through the learned MPC and the target linearization oracle on the same plant/horizon/bounds/cost, and reports regret, tracking error, effort and failures.


### 4.6. Practical identifiability and data coverage (H2)

For `y=dot x=f0(x)+G(x)u`, the population relation

\[
G(x)=\operatorname{Cov}(y,u\mid x)\operatorname{Cov}(u,u\mid x)^{-1}
\]

requires positive definite conditional input covariance and suitable exogeneity/noise assumptions. It is **not** a finite-sample estimator guaranteed to work on high-dimensional sparse trajectories. Record local design-matrix singular values, local input covariance condition numbers, spatial distance to train data, MPC-vs-random state coverage, action-excitation bin counts and outcomes with reduced excitation. Factorization alone does not imply causal identification.

## 5. Experiment registry

| ID | Content | Status |
|---|---|---|
| E00–E04 | P0 audit, CartPole data/reproduction | done (P0/P1 closed) |
| E05–E09 | baseline margin, rotation sweep, unicycle/quadrotor, MPC harness, identifiability | done (historical P2; E06/E07 re-done with calibrated baseline below) |
| P1-grid | 96-run fraction grid, upstream-faithful variants | done |
| P2-calib | config check, robust flow, drift/coverage tuning → frozen baseline | done |
| H2-D | fixed-B disease diagnosis: rotation sweep + data/latent/G.5 controls + CartPole bands | done — disease confirmed |
| H2-G5 | T1 rotation residual, T2 CartPole + G.5, T3 unicycle v2, T4 quadrotor v2 | done — `NOT_CONFIRMED_BEYOND_G5` (T1 ✓, T2 G.5 fails, T3 baseline uncalibrated, T4 insensitive) |
| H2-gate | C1 unicycle calibrated, C2 quadrotor thrust rows, C3 CartPole cos-G.5 | done — `NOT_YET` |
| H2-C4 | CartPole confirmatory, fresh seeds 45–47 | done — PASS → **H2 readiness GO** |
| H2-01/02, H3-01/02 | methods | only after the respective gate |

## 6. Experimental controls and acceptance criteria

**Reproduction:** determine paper's precise dataset budget, window sampling, optimizer, seed, loss flags, solver, and evaluation horizon. Log any mismatch rather than silently repair. Where source code is inconsistent, maintain: (i) untouched upstream snapshot, (ii) independent corrected runner, (iii) explicit diff and protocol classification. Do not call a local fix an exact upstream reproduction without qualification.

**Data integrity:** train/validation/test split by trajectory and by regime if applicable; checksum each `.pt`; avoid regenerating same path after results are viewed; fixed angle conventions and physical units. Oracle `f,G` cannot enter Protocol A train loop or model selection based on private test states.

**Statistical reporting:** pre-register at least 5 seeds for confirmatory ML experiments when feasible (smoke and protocol debugging may use one); report mean/std, confidence interval or bootstrap at trajectory level, plus all failures and negative results. Pre-register model/parameter budget, training updates, learning-rate schedule, ODE tolerances, horizon and MPC solver accuracy. No test-set hyperparameter tuning.

**H2 GO:** failure persists on at least one controlled state-dependent stress test after checking coverage/solver; matched-class monolithic and factorized comparison can be designed without privileged labels. H2 failure is not inferred from Pendulum-only or zero-action-only tests.

**H3 GO:** (i) verified proof/assumptions, (ii) nontrivial certified correction capacity with appropriate constrained factorization, (iii) a viable tractable controller (exact QP, clearly labeled approximate QP, or NMPC with acceptable compute/utility). Controllability rank alone is insufficient. `rho` can be tiny; that is a valid NO-GO outcome.

**Comparison semantics:** acceptance thresholds and compute budget are to be recorded **before** full experiments, not retrofitted to observed performance. P0/P1 success is about **protocol reproducibility**, not evidence of H2/H3 novelty.

## 7. Artifacts and provenance

Every experiment is preregistered in TRACKING (question, script hash, seeds, budgets, rules) before running, writes per-run `result.json` with data/checkpoint SHA-256, and is evaluated by reloading checkpoints with `strict=True`. Raw runs and synthetic data are ignored in Git; reports, `summary.json` and curated final checkpoints are tracked. Historical P2 recompute: `python -m research.diagnostics.recompute_p2 --report outputs/diagnosis/p2_failure_report.yaml --tolerance 1e-6`.
