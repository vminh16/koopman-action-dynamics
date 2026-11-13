# ROADMAP.md — Research Execution & Repository Architecture
## `koopman-action-dynamics` · RRD v0.5 / EXPERIMENT v0.5

**Version:** 0.5 · 2026-11-12  
**Working paper title:** *Learning Action-Dependent Koopman Dynamics for Prediction and Control*.  
**Primary baseline:** Miao et al., *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026.  
**Upstream audit commit:** `ec131bb549c8f7b01519a99acca237b9fdc9c826` (public `KYMiao/Controllable-Koopman`, static review).  
**Current state:** P0 Pendulum audit completed (verdict: CONFIG_DRIFT_CONFIRMED_PERFORMANCE_CAUSE_NOT_CONFIG, seed variance bimodal); P2 controlled stress tests & baseline diagnosis COMPLETED using official baseline `KoopmanLitModule` (Gate P2: PASS; Gate H2 Fixed-B limitation: NOT_CONFIRMED, Readiness: NO-GO; Gate H3: NO-GO due to lack of certificate margin window). H2/H3 tracks remain paused without model training.

> **Do not drift:** RRD defines scientific hypotheses. EXPERIMENT defines evidence and measurements. ROADMAP defines execution order, code boundaries and release gates. H2 and H3 are independent research tracks. Through P2, only **the author's Koopman method** is trained, including when that same method is run on new synthetic systems; do not reproduce external model baselines at this stage.

---

## 0. Decisions and repository name

**Recommended repository:** `koopman-action-dynamics` (availability on GitHub not checked).  
**Research direction:** expressive and controllability-aware Koopman latent dynamics with two independent hypotheses:

- **H2 — Dynamics-Factorized Learning:** compare factorization as a *learning mechanism* against matched-model-class alternatives; do not claim two branches themselves are novel.
- **H3 — Certified State-Dependent Input Maps:** characterize and measure fidelity–controllability–MPC tractability trade-off; an abstract surrogate certificate alone does not suffice.

**Primary code strategy: minimal-change overlay.** Keep upstream top-level modules in their original paths so all absolute imports remain valid. Put all new researcher-owned code under `research/`, reusable command wrappers under `scripts/`, and documents at root. No mass move/rename, no bulk formatting of original files, no import rewrites in P0/P1.

**License/provenance caveat:** no `LICENSE` file was visible in the audited upstream tree. Before publishing a derivative repository with original source copied, verify reuse/distribution permissions and retain the author's citation/history. Public access on GitHub does not by itself give an open-source license. If permissions remain unclear, prefer a separate repository containing only your code plus a documented upstream checkout/submodule (Alternative B below).

---

## 1. Audit of real upstream code and minimum required adjustments

### 1.1. Confirmed by static inspection (not runtime)

| Priority | Upstream source | Finding | Decision |
|---|---|---|---|
| **BLOCKER** | `experiments/run_train.py` | Active model constructor is `MLPLitModule`; `KoopmanLitModule` commented. Flag `--koopman_mode` alone **does not** make this a Koopman run. | First verify existing Pendulum checkpoint; write an **independent Koopman runner** under `research/baseline/` using official modules. Leave upstream untouched. |
| **BLOCKER** | `experiments/run_eval.py` | Loader constructs `KoopmanLitModule` unconditionally. | Validate checkpoint class/key compatibility before evaluation. |
| HIGH | `experiments/run_train.py` | `--lr` parsed but not passed to current MLP module; script selects `30%` of window dataset. | Explicit run manifest and effective optimizer LR/data fraction in new runner. |
| HIGH | `my_utils/dataset.py` | Window starts limited to `0..4` by hardcoded `max_start=5`. | Preserve behavior only for exact-script audit; a corrected sampler lives in research layer and is separately labeled as a protocol variant. |
| HIGH | `training/loss.py` | Controllability-loss horizon argument overwritten with `15*dt`. | Record effective training horizon; use independent float64 Gramian diagnostics for H3. |
| HIGH | `experiments/run_control.py` | Empty. | Build verified MPC evaluation **outside upstream** before any closed-loop claims; paper's table is not reproduced by this file. |
| MEDIUM | `dynamics/cartpole.py` | In-place angle overwrite on input state in `dynamics()`. | Test for mutation/autograd before dataset generation or derivatives; if fixed, do so in a documented adapter/fork patch, not silently. |
| MEDIUM | `my_utils/dataset.py` | Simulator data saved to `data/{system}_{split}.pt`; generation CLI does not seed itself. | Seed externally + hash data; retain flat `data/` path for compatibility. |
| HIGH | `experiments/run_train.py` | Save checkpoint vào `checkpoints/` nhưng không tự tạo folder; ghi config chỉ sau lệnh save. | Kiểm tra tồn tại checkpoint thực tế và tạo folder từ wrapper trước khi train; tránh xem log epochs là successful saved run. |
| MEDIUM | `README.md` | Example `window_len=50`; reproduction notes indicate paper uses 15. | Freeze and document **paper-vs-repo protocol** instead of treating README examples as canonical paper settings. |
| MEDIUM | Repo tree | No tracked environment lockfile/LICENSE visible; `.pyc` tracked upstream. | Add own env lockfile and gitignore; don't rewrite upstream history in bulk. |

**Important:** The audited code is the upstream commit named above. Your local repository may contain modifications. All conclusions about what your Pendulum run *actually did* remain open until local verification.

### 1.2. Upstream modules to reuse (no copy, no imports rewritten)

| Path | Role | How research code should call it |
|---|---|---|
| `dynamics/__init__.py` | `SYSTEM_REGISTRY`, four registered simulators | Use unchanged for official Pendulum/CartPole reproduction. |
| `dynamics/{pendulum,cartpole,mountaincar,grn}.py` | Continuous-time physical dynamics | Import for official datasets and oracle **evaluation**; do not embed oracle gradients in observational training. |
| `my_utils/dataset.py`, `my_utils/simulate.py` | Generation, `.pt` files, window dataset | Wrapper for deterministic generation; audit legacy window sampler. |
| `models/encoder.py` | `StructuredEncoder`, `UEncoder` | Shared observed-state lift and existing Appendix G.5 input encoder. |
| `models/koopman_operator.py` | Free/controllable operator, multi-input support | Extract/inspect actual `(A,B)`, maintain canonical parameterization; no baseline edits. |
| `models/model_builder.py` | Builds `KoopmanModel`, ODE solver | Reuse from research runner; configure via explicit validated dict. |
| `training/train.py`, `training/loss.py` | Lightning modules, prediction/encoder/Gramian losses | Prefer reuse for faithful baseline; corrected variants require labeled protocol. |
| `training/evaluate.py` | Open-loop/re-encode metrics | Reuse after checking metric semantics, do independent recomputation. |
| `experiments/*.py` | Upstream CLIs (not uniformly coherent) | Preserve as reference, **not** as unquestioned production pipeline. |

---

## 2. Recommended repository structure — minimum import disruption

```text
koopman-action-dynamics/
├── README.md                         # Your project README + attribution + quickstart
├── RRD_v0.5.md                     # Theory; update by explicit versioned revisions
├── EXPERIMENT.md                    # Measured evidence, experiment registry
├── ROADMAP.md                       # This document
├── .gitignore                       # Ignore outputs, data, checkpoints, pycache
├── environment/                     # Your reproducibility lock / environment manifest
│   └── README.md
├── docs/
│   ├── UPSTREAM_AUDIT.md            # Optional: code findings and local patch register
│   └── decision_log.md
│
├── dynamics/                        # UPSTREAM — keep path, source names, imports
├── models/                          # UPSTREAM — keep path
├── training/                        # UPSTREAM — keep path
├── my_utils/                        # UPSTREAM — keep path
├── experiments/                     # UPSTREAM — keep path
│
├── research/                        # ALL USER-OWNED IMPLEMENTATION
│   ├── __init__.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── verify_checkpoint.py     # Identify MLP vs Koopman, consistency checks
│   │   ├── train_koopman.py         # Correct, explicit baseline runner
│   │   ├── eval_koopman.py          # Evaluation wrapper and reproducibility manifest
│   │   └── model_adapter.py         # Thin adapter only, no duplicated physics/model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate.py              # Seeds, splits, hashes, manifests
│   │   └── validate.py
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── registry.py              # Research registry; DO NOT edit upstream registry
│   │   ├── rotation_family.py
│   │   ├── unicycle.py
│   │   └── planar_quadrotor.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   ├── action_response.py
│   │   ├── lifted_input_map.py
│   │   ├── controllability_margin.py
│   │   ├── coverage.py
│   │   └── candidate_regret.py
│   ├── control/
│   │   ├── __init__.py
│   │   ├── qp_mpc.py                # Baseline LTI QP and controlled solver tests
│   │   └── solver_checks.py
│   ├── methods/
│   │   ├── h2/                      # EMPTY until H2 gate
│   │   └── h3/                      # EMPTY until H3 gate
│   └── common/
│       └── provenance.py
│
├── configs/
│   └── research/                    # Your configs; don't overwrite generated upstream YAML
├── scripts/
│   ├── audit_local.sh
│   ├── reproduce_pendulum.sh
│   └── reproduce_cartpole.sh
├── tests/
│   ├── test_upstream_interface.py
│   ├── test_dataset_contract.py
│   ├── test_checkpoint_class.py
│   ├── test_gramian_numeric.py
│   └── test_benchmark_oracles.py
│
├── data/                             # Existing upstream expects flat data/{system}_*.pt
├── checkpoints/                     # Existing upstream output; gitignored
├── lightning_logs/                  # Existing upstream output; gitignored
└── outputs/                         # Research results & immutable manifests; large gitignored
```

**Compatibility rule:** the only dependencies of `research/` on upstream are public import surfaces, e.g. `from training.train import KoopmanLitModule`, `from models.model_builder import build_model`, `from dynamics import SYSTEM_REGISTRY`; upstream never imports from `research/`. Use `PYTHONPATH=.` from project root. Do not create another top-level package named `models`, `training`, `dynamics` inside `research/` that causes ambiguous imports.

**Ownership rules:** For each original file, keep it byte-for-byte unchanged whenever practical; new modules go to `research/`. If a verified correctness fix *must* touch upstream, make one minimal focused patch with test, note changed lines and a clear commit message. No upstream mass refactor until P1 is verified. Changes in `research/` must be reviewable separately from source of [P1].

**Alternative B — physical vendor separation:** `vendor/Controllable-Koopman/` as pinned Git submodule; execute wrappers with `PYTHONPATH=vendor/Controllable-Koopman:.` from root (or run upstream scripts from submodule directory). This keeps upstream imports textually unchanged but complicates cwd-relative `data/`, `configs/`, `checkpoints/`, and environment setup. Prefer only if public redistribution rights require keeping original source outside your repository, or when upstream updates become frequent. **Do not migrate during P0/P1.**

---

## 3. Phase sequence and evidence gates

### P0 — Local Pendulum forensic audit (COMPLETED: MISMATCH)

**Goal:** determine what was trained and whether existing result is valid baseline evidence.

- [x] `git rev-parse HEAD`, `git status --short`, model class, effective flags and optimizer LR recorded.
- [x] Checkpoint keys and architecture compatibility audited; the completed local runs use `KoopmanLitModule`.
- [x] Data hashes, sampled window range, train fractions, train/test split, time step, model dimensions and evaluation script recorded.
- [x] Re-evaluated open-loop on 1,000 test trajectories; results and source manifest saved.
- [x] Protocol mismatch classified honestly; exact paper reproduction remains a follow-up, without overwriting existing runs.

**Output:** [manifest](outputs/reproduction/E00/manifest.yaml), checkpoint audit and `MISMATCH` verdict.
**Exit:** no remaining ambiguity about the local Pendulum run type, config or provenance. The run is not claimed as exact paper reproduction.

### P1 — Dataset and original-model reproduction (CartPole NEXT)

- [ ] Freeze author commit and package versions. Audit paper vs README settings (`window_len`, optimizer, sampling, Gramian effective horizon).
- [ ] Use author's simulator to **generate** CartPole dataset, not assume an external download. First smoke: 10–20 trajectories; then independent train/test with dataset manifest and seeds.
- [ ] Validate sample shape, no NaNs, physical input range, state mutation in CartPole ODE, `.pt` hashes, train/test separation.
- [ ] Implement the *correct Koopman* training entrypoint in `research/baseline/train_koopman.py`, instantiate `KoopmanLitModule`, pass effective LR and record all sampling decisions. Reuse official encoder/operator/loss code; label divergence from upstream entrypoint.
- [ ] Evaluate Pendulum (if needed) and CartPole with same protocol as paper; distinguish open-loop/periodic re-encode; repeat runs for robust reported results.
- [ ] Confirm Gramian matrix/rank using independent numeric test and float64, checking learned `(A,B)` are really structured/controllable.

**Output:** reproducibility table, source/data/model hashes, saved checkpoints and variance across seeds.  
**Exit:** author model truly reproduced or precise reason/documentation of remaining mismatch; do not hide failed replication.

### P2 — Baseline diagnosis on original + research stress-test systems

**Goal:** isolate failure modes without training any H2/H3 methods.

- [x] Build `rotation_family.py`: state-dependent input-direction with tunable rotation parameter `a`; analytic `f0,Gx`; test `a=0` and a preregistered sweep of stronger rotations.
- [x] Build Unicycle: full-state, heading-dependent input direction. Declare controls as `(v,ω)` and state `(x,y,θ)`; do not silently change to steering-angle vehicle model.
- [x] Build Planar Quadrotor: orientation-dependent thrust, mass/gravity/inertia, full state and actuator bounds; compare in same control setting.
- [x] Use **the same original Koopman architecture** trained on the new simulator datasets; report parameters, data budget, action distribution and coverage.
- [x] Measure response error, `CB` vs physical `Gx`, `B` vs lifted `JφGx`, manifold/drift errors, `d_orient`, constrained `d_star`, `ρ_T` vs `T/N`, state coverage, candidate regret.
- [x] Implement/test baseline convex QP MPC in `research/control/` (upstream `run_control.py` is empty); verify constraints and objective on an analytically simple LTI toy before using learned models.
- [x] Plot stress-strength vs response error, prediction error and closed-loop cost. Separate **model-class limitation**, **data coverage** and **solver numerical issues**.

**Output:** `outputs/diagnosis/` with explicit failure-mode report and positive/negative controls.  
**Exit:** a testable reason to improve model, or evidence to pivot. Negative Pendulum/CartPole result alone is **not** a veto.

### Gate H2 — Independent architecture/learning hypothesis

**Gate Verdict: NOT_CONFIRMED (Readiness: NO-GO).**
- Measured response growth ratio $E_{\text{response}}(a=2.0)/E_{\text{response}}(a=0.0) = 0.95 \pm 0.02 < 1.5$ (non-monotone across $a \in \{0.0, 0.5, 1.0, 2.0\}$) under the official baseline `KoopmanLitModule`.
- Coverage ($\text{state distance p95} \le 0.58 \le 3.0$) and optimization ($\text{one-step MSE} \approx 2.8 \times 10^{-4} \le 0.05$) pass pre-registered gates.
- Previous 4.82x degradation report was an artifact of fitting an unconstrained linear surrogate on misaligned transitions.
- Track H2 remains paused until/unless a modified excitation protocol or longer training reveals an identifiable failure mode.

### Gate H3 — Independent control-theory/certificate hypothesis

**Gate Verdict: NO-GO.**
- Controllability margin $\rho_T \approx 0.0235 > 0$ is well-defined, and QP MPC achieves 100% solver success with P95 latency $< 5$ ms.
- However, orientation error $d_{\text{orient\_p95}} \approx 0.125-0.131 > \rho_T$ and bounded factorization distance $d^*_{\text{p95}} \approx 0.125 > \rho_T$.
- No certified correction window exists under singular-value bounds $\sigma(M) \in [0.5, 2.0]$.
- Track H3 remains closed on this baseline architecture.

### P3-H2 — Research method, only after H2 gate

Train input-affine one-branch, factorized two-branch, and matched-class monolithic control-affine ablations with identical dataset/sampler/solver/param and compute constraints. Evaluate counterfactual action fidelity, prediction and MPC; report practical identifiability. Controlled intervention training is an **optional separate protocol** with equal privileges to all compared models. This track can succeed or fail independently of H3.

### P3-H3 — Research method, only after H3 gate

Implement S1 (existing input scaling), S2 (free state-dependent input map), S3 (certified constrained map), with fixed baseline definitions and independently tested certificates. Evaluate actual finite-horizon tractability: full NMPC, approximate scheduled QP, potential virtual-input QP with conservative physical bounds. Record validity conditions of `M(z)` singular values and `ΔB` norm, stability/conditioning, `δ/T/N` sweeps and extra solver cost. Do not inherit original global QP guarantee when dynamics is nonlinear in state.

### P4–P5 — Research evidence and optional combination

P4: independent H2 and H3 reports, statistical significance, failure cases, baselines if publication competition requires them, novelty audit. P5 optional: only if both tracks are independently supported, open a **new** scoped RRD for H2+H3 or online adaptation. No automatic feature creep.

---

## 4. Immediate work queue and commands

**Order (not a time estimate):**

1. **Verify existing Pendulum:** get local commit and actual script, review checkpoint keys + effective LR; avoid rerunning unless invalid.
2. **Set up Git boundary:** establish upstream provenance, create `research/` and test scaffolding; do **not** move original folders.
3. **CartPole smoke dataset:** use generator only after seed/mutation audit; test `n=20`, `T=250` before full dataset.
4. **CartPole official Koopman reproduction:** use new thin runner, retain paper-vs-repo protocol variants.
5. **Controlled rotation toy:** write only benchmark/oracle and diagnosis, **not** H2/H3 models.
6. **MPC harness:** implement baseline QP and unit-test mathematical correctness before reporting closed-loop claims.

Safe inspection commands for existing checkout:

```bash
# In repository root
pwd
git rev-parse HEAD
git status --short
git diff --stat
python --version
python -m pip freeze > outputs_or_local_environment.freeze.txt
python -m my_utils.dataset --help
# WARNING: scripts in upstream are NOT interchangeable as currently written.
# Do NOT assume `python experiments/run_train.py --koopman_mode controllable`
# really instantiates the Koopman model before verifying local implementation.
```

Do not commit `outputs_or_local_environment.freeze.txt` if it exposes local paths or private package indexes; create sanitized lock/manifest in `environment/` instead.

**Recommended `.gitignore` additions** (new generated artifacts, preserve source):

```gitignore
__pycache__/
*.py[cod]
.venv/
.env
.env.*
data/*.pt
checkpoints/
lightning_logs/
outputs/
figures/
*.log
```

Keep `.gitkeep` in empty output directories if desired. **Do not remove already tracked upstream `.pyc` in the same code-change commit**; clean them in a separate housekeeping commit after baseline audit.

---

## 5. Git provenance and branching policy

Preferred if licensing and attribution are resolved: fork the original project and preserve its history, then keep researcher code isolated in `research/`. Record `UPSTREAM_COMMIT` in README/manifest and keep upstream remote. Example for a fork/clone that already points to author's project:

```bash
# Example only: inspect remotes before changing them
git remote -v
git remote rename origin upstream
# git remote add origin git@github.com:<YOUR_USER>/koopman-action-dynamics.git
# git push -u origin main
```

If using GitHub's fork feature, origin may already be your fork and upstream remote configured: **do not run remote rename mechanically**. If rights are not clarified, use a separate user-owned repo with an upstream checkout/submodule and document the environment-boundary workaround; do not implicitly license or relicense the author's files.

Branch/commit convention (suggested): `audit/pendulum-run`, `repro/cartpole`, `diagnostics/input-rotation`, `track/h2-*`, `track/h3-*`. Keep method-implementation commits separate from dataset/protocol fixes. No rename of upstream package until source provenance and P1 metrics are locked.

**Code verification contract for later review:** a PR/commit should include changed files, exact command, expected outputs, reproducibility manifest, unit tests and checkpoint/data provenance. Review order: source audit → shape/physics/math tests → training/evaluation reproducibility → statistical/control interpretation. A successful program exit alone does not meet acceptance.

---

## 6. Stop conditions, risks and decisions to record

| Risk | Detection | Action |
|---|---|---|
| Pendulum is MLP | `model.net` keys / script ctor | Reclassify run; reproduce Koopman correctly; preserve old results |
| Paper protocol inaccessible/ambiguous | README-vs-paper/code mismatches | Label protocol variant, inspect authors' sources, don't invent exact claims |
| CartPole input mutation/gradient error | Mutation unit test, autograd anomaly | Isolate minimal repair, record diff and test |
| Dataset coverage too narrow | Window starts, local `Cov(u\mid x)`, state-distance plots | Fix data protocol as explicit variant; don't call branch failing evidence |
| Fixed input direction not harmful in closed loop | Candidate regret and MPC still good | Report negative result; pivot stress case rather than tune to cherry-pick |
| H3 certificate margin vacuous | `rho`/bounded factorization report | Stop or scope to local/trajectory certificate after new mathematical review |
| H3 destroys control efficiency | NMPC P95 latency, infeasibility vs baseline QP | Do not claim Koopman/QP advantage; pivot or stop |
| Source license unclear for public repo | No verified original LICENSE | Clarify rights, use fork mechanism or private/local checkout as appropriate |

**Decision log:**
- 2026-11-12: v0.5 synchronized to RRD: shared P0–P2, H2 and H3 independent, no external baselines before diagnostics.
- 2026-11-12: Pendulum reported finished; verification remains open due to upstream `MLPLitModule`/`KoopmanLitModule` entrypoint mismatch.
- 2026-11-12: Select minimal-change overlay for import stability; new code in `research/` and untouched upstream roots.
- 2026-11-13: P0 Pendulum config and seed variance probe completed. Effective config is verified; seed variance is bimodal.
- 2026-11-13: P2 re-run with official baseline `KoopmanLitModule` (fixing surrogate substitution, transition off-by-one, benchmark dt, and multi-input operator device/H buffer bugs). Gate P2 PASS. Gate H2 Fixed-B limitation NOT_CONFIRMED (readiness NO-GO); Gate H3 NO-GO. H2/H3 tracks remain closed.

---

## 7. References

- [RRD v0.5](RRD_v0.5.md) — definitions, theory and independent hypotheses.
- [EXPERIMENT v0.5](EXPERIMENT.md) — exact metrics, experiment registry, data protocols.
- [Original ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/e3781fb5d2c17ea7283cc2a543793bdb-Paper-Conference.pdf).
- [Author code at audited commit](https://github.com/KYMiao/Controllable-Koopman/tree/ec131bb549c8f7b01519a99acca237b9fdc9c826).
