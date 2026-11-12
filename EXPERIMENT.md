# EXPERIMENT.md — Experimental Protocol & Evidence Registry
## Expressive and Controllability-Aware Koopman Latent Dynamics

**Version:** 0.5 · 2026-11-12  
**Theory source:** `RRD_v0.5.md` (normative scientific scope)  
**Execution partner:** `ROADMAP.md` (phases, gates, repository policy)  
**Research tracks:** H2 — Dynamics-Factorized Learning; H3 — Controllability-Preserving State-Dependent Input Maps. **Keep separate.**  
**Reference:** Miao et al., *Learning Koopman Representations with Controllability Guarantees* (ICLR 2026).  
**Upstream repository:** `https://github.com/KYMiao/Controllable-Koopman`  
**Audited upstream `main` commit:** `ec131bb549c8f7b01519a99acca237b9fdc9c826` (static inspection on 2026-11-12).  
**Evidence status:** Người thực hiện báo đã chạy Pendulum. **Chưa xác minh** lệnh chạy, loại mô hình, checkpoint, dữ liệu hoặc MSE; không ghi “reproduced”.

> **Nguyên tắc bất biến:** P0–P2 chỉ huấn luyện và kiểm tra **mô hình ICLR 2026** (trên dữ liệu gốc và stress-test mới), không chạy external model baselines. Simulator oracle cho phản thực và đạo hàm **chỉ dùng diagnosis/evaluation** trong Protocol A. Không phát triển/huấn luyện H2 hoặc H3 trước các gate tương ứng. Không coi learned-surrogate controllability là bảo đảm của plant thật.

---

## 1. Questions, evidence tiers, and experiment boundaries

| Question | Evidence | Stage |
|---|---|---|
| E0. Mô hình Pendulum đã chạy thực sự là structured Koopman? | Script provenance, `state_dict` keys, config, evaluation log | **P0, trước mọi việc khác** |
| E1. Có tái lập kết quả paper bằng cùng protocol? | Pendulum/CartPole metric, provenance, ≥ nhiều seed nếu cần | P1 |
| E2. Prediction fidelity có che lấp action-response error ở action khác? | Same-state paired-action oracle, candidate regret, visited-state coverage | P2 |
| E3. Fixed input operator sai ở physical/lifted space thế nào? | `CB` vs `G_x`, `B` vs `JφG_x`, drift/manifold residual | P2 |
| E4. Khi input direction đổi theo state, failure tăng có hệ thống không? | Controlled rotation-strength sweep, uncertainty, coverage | P2 |
| E5. Koopman margin có đủ hữu ích không? | `ρ_T`, orientation, bounded-factorization, numerical stability, QP trade-off | P2; H3 gate |
| E6. Factorization giúp **learn** tốt hơn cùng model class? | Matched-capacity monolithic vs two-branch | P3-H2 |
| E7. H3 vừa certified vừa hữu ích khi planning? | Certified-vs-free, actual QP/NMPC, physical bounds | P3-H3 |

**Claims hierarchy:** (a) exact algebraic property; (b) observed prediction; (c) counterfactual fidelity; (d) MPC usefulness; (e) deployment. Không suy ra mức (d)/(e) chỉ từ (a)/(b). 

---

## 2. P0 — Audit của mã nguồn chính thức và kết quả Pendulum đã chạy

### 2.1. Findings from upstream static inspection (must reconcile against LOCAL COMMIT)

| ID | Source location | Verified upstream behavior | Risk / Required test |
|---|---|---|---|
| U01 — **Critical** | `experiments/run_train.py` | `model = MLPLitModule(...)` là dòng **đang chạy**; `KoopmanLitModule` được comment | Một run đặt `koopman_mode=controllable` vẫn có thể đang train **MLP**, không phải Koopman. Kiểm tra checkpoint trước khi dùng kết quả. |
| U02 — **Critical** | `experiments/run_eval.py` và `experiments/analyze_controllability.py` | Cả hai khởi tạo `KoopmanLitModule` khi load checkpoint | Checkpoint MLP từ U01 không tương thích với evaluator Koopman. Không được coi eval thành công nếu loader bị sửa mà không ghi lại. |
| U03 — High | `experiments/run_train.py` | `--lr` được parse và lưu config nhưng không truyền vào `MLPLitModule`; mặc định module là `1e-3` | CLI LR không nhất thiết là LR đã train. Với runner mới phải truyền explicit `lr`. |
| U04 — High | `experiments/run_train.py` | Dữ liệu sau tạo cửa sổ bị lấy ngẫu nhiên `30%`; không phải lựa chọn tỷ lệ từ CLI | Phải phân biệt **30% of sampled windows** với % trajectories trong paper. |
| U05 — High | `my_utils/dataset.py`, `SampledMultiTrajectoryDataset` | `max_start = 5` cố định; chỉ chọn start trong `range(5)` bất kể T/window; `max_windows_per_traj=20` không vượt 5 | Mẫu training tập trung đầu trajectory. Không tự ý gọi là protocol paper. |
| U06 — High | `training/loss.py`, `controllability_loss` | `T = 15*dt` ghi đè đối số `T`; Gramian tính bằng Euler | Ghi horizon thực; kiểm định margin bằng implementation độc lập/float64. |
| U07 — High | `experiments/run_control.py` | File rỗng (0 byte) | Repo chính thức **không cung cấp MPC entrypoint hoạt động** trong file này; không claim đã reproduce MPC nếu chưa có code/protocol. |
| U08 — Medium | `my_utils/dataset.py` | Dữ liệu được **sinh** rồi lưu `data/{system}_{split}.pt`; seed không được thiết lập trong CLI generator | Dataset không có download URL trong repo; phải tạo mới với seed và hash; không trộn data versions. |
| U09 — Medium | `dynamics/cartpole.py` | `dynamics()` ghi `x[...,2] = theta` in-place | Rủi ro autograd/mutation; thử nghiệm kiểm tra không mutate input và xử lý bằng adapter riêng nếu cần. |
| U10 — Medium | `README.md`, `run_train.py` | README đưa `window_len=50` và đường dẫn ví dụ `configs/...`; code sinh config sau training | Đối chiếu paper-vs-code; không tự nhận `50` là window paper. |
| U11 — Medium | Repo tree | Có các `.pyc` được track; không thấy `requirements.txt`/`pyproject.toml` hoặc `LICENSE` trong commit đã xem | Lock môi trường, `.gitignore` artifact mới; **kiểm tra quyền tái phân phối code trước khi public**. |
| U13 — High | `experiments/run_train.py` | `torch.save(..., "checkpoints/...")` nhưng script không tạo `checkpoints/` trước; `configs/` chỉ được tạo **sau** bước lưu | Trên checkout sạch, lần train có thể kết thúc bằng lỗi ghi checkpoint; xác minh file có thật và log hoàn chỉnh. |
| U12 — Method | `training/train.py`, `KoopmanLitModule` | Optimizer AdamW + learned encoder/operator; `training/loss.py` dùng weighted prediction, encoder consistency, Gramian penalty | Audit flags, weights, state dimensions và solver trước khi so paper; **không** giả định README khớp paper. |

**Scope of audit:** code review tĩnh từ upstream commit được ghi ở trên, **chưa chạy runtime trên máy người dùng**. Nếu local fork khác commit, audit theo local diff trước khi kết luận.

### 2.2. Pendulum verification checklist (retroactive audit)

- [ ] Ghi `git rev-parse HEAD`, `git status --short`, `git diff --stat` và thông tin environment (`python`, PyTorch, Lightning, torchdiffeq, OS/GPU).
- [ ] Lưu **lệnh thực tế** đã train/eval, các log tham số (`lr`, epochs, window, seed, loss flags).
- [ ] Kiểm tra tên lớp model đã khởi tạo **bằng runtime log/source**, không suy luận từ `--koopman_mode`.
- [ ] Kiểm tra `state_dict`: Koopman có các keys về `model.encoder`/`model.operator`; MLP thường có `model.net`. Khi tên khác, kiểm tra mã nguồn thực tế.
- [ ] Ghi số trajectories dùng để train, số sampled windows, split theo trajectory, seed của generator và sampler.
- [ ] Eval lại checkpoint với đúng model class, cùng horizon/metric/split paper, tách open-loop vs re-encode.
- [ ] Nếu Pendulum là MLP hoặc khác protocol: đổi status thành `completed_unverified` hoặc `not_reproduced`; **không xóa log cũ**, chạy lại structured Koopman từ một runner nghiên cứu tách biệt.

**Required Pendulum record (YAML/JSON):** `experiment_id`, `kind`, `upstream_commit`, `local_commit`, `entrypoint`, `model_class`, `model_config`, `data_hashes`, `train_seed`, `eval_seed`, `dataset_fraction_definition`, `n_windows`, `dt`, `ode_solver`, `epochs`, `optimizer_lr_effective`, `checkpoint_sha256`, `eval_protocol`, `metric_table`, `status`, `notes`. Chưa có trường nào được điền bằng suy đoán.

---

## 3. Dataset protocol — Không “download” nhầm synthetic data

### 3.1. Data sources and splits

**Paper reproduction dataset:** dùng simulator có trong `dynamics/` và generator `my_utils/dataset.py`; lưu file `data/{system}_{split}.pt` đúng layout upstream. `SYSTEM_REGISTRY` hiện chỉ gồm `pendulum`, `cartpole`, `mountaincar`, `grn`; paper có các môi trường khác nhưng chưa có implementation tương ứng trong tree đã kiểm tra. Không giả định sẵn Reacher/Franka data.

| Dataset ID | System | Role | Status | Nguồn/điều kiện |
|---|---|---|---|---|
| D-PEND-TRAIN/TEST | Pendulum | Upstream reproduction | Có run được báo cáo; chưa kiểm định data | Dữ liệu địa phương cần fingerprint |
| D-CART-TRAIN/TEST | CartPole | **Next official reproduction** | Chưa xác nhận tồn tại | Sinh từ `CartPoleDynamics` chính commit khóa |
| D-ROT-* | 2D input-direction rotation family | Controlled stress strength sweep | Chưa tạo | `research/benchmarks/`, analytic oracle |
| D-UNI-* | Unicycle | Orientation-dependent input | Chưa tạo | Ràng buộc state/action và miền góc rõ |
| D-QUAD-* | Planar Quadrotor | Orientation-dependent thrust, multi-input | Chưa tạo | Ghi state variables, thrust bounds, parameter regime |
| D-ACRO-* | Acrobot | Optional extension | Không ưu tiên | Chỉ nếu stress đơn giản không đủ |

**Reference experiment counts (to VERIFY from paper before full run):** Pendulum thường 5.000 train / 1.000 test; CartPole 10.000 train / 1.000 test; 250 steps/trajectory; paper training windows được ghi là 15 steps, README là 50. Những con số này là **reference candidates** từ RRD/paper discussion, không phải bằng chứng rằng dữ liệu local đã đúng.

**Next dataset workflow:** (i) khóa physics+seed+sampling; (ii) smoke test 10–20 trajectories và shape/dtype/finite checks; (iii) sinh `cartpole_train.pt` và `cartpole_test.pt` độc lập; (iv) hash SHA-256; (v) không overwrite dữ liệu Pendulum. Có thể dùng entrypoint gốc (ví dụ kiểm tra `python -m my_utils.dataset --help`); runner mới chỉ bọc seed/manifest, không thay simulator.

**Illustrative upstream generator commands (AFTER audit; without seed lock these are NOT full scientific runs):**

```bash
PYTHONPATH=. python -m my_utils.dataset --system cartpole --num 20 --T 250 --split train
# Sau khi có seed+generator manifest:
# PYTHONPATH=. python -m my_utils.dataset --system cartpole --num 10000 --T 250 --split train
# PYTHONPATH=. python -m my_utils.dataset --system cartpole --num 1000  --T 250 --split test
```

**Dataset contract:** each trajectory has `x0`, `u_seq`, `x_seq`; inspect shapes (`x0` `[1,n]`, `u_seq` `[T,m]` or equivalent, `x_seq` `[T,1,n]` upstream); ensure `T ≥ window_len`, time alignment `x(t+dt)` vs `u(t)`, bounded inputs, no NaNs, no shared trajectories across splits. Log numerical integrator, `dt`, physical units, state sampling ranges and data generator commit.

### 3.2. Three distinct data protocols

- **A — Observational (PRIMARY):** random admissible actions/trajectories for training; held-out trajectories for metrics. Same-state simulator resets and true `f,G_x` allowed **only** for diagnostics/evaluation. This is the claimable observational-learning setting.
- **B — Controlled-intervention training (OPTIONAL AFTER H2 GATE):** deliberately repeat states with diverse actions **in training only**, give all compared methods same intervention budget; report as a **different information setting**. Never compare A vs B as an architecture-only gain.
- **C — Shift/adaptation (FUTURE):** split by physics regimes, enforce no test-regime leakage. Not in P0–P2 or H2/H3 core claims.

---

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

### 4.6. Practical identifiability and data coverage (H2)

For `y=dot x=f0(x)+G(x)u`, the population relation

\[
G(x)=\operatorname{Cov}(y,u\mid x)\operatorname{Cov}(u,u\mid x)^{-1}
\]

requires positive definite conditional input covariance and suitable exogeneity/noise assumptions. It is **not** a finite-sample estimator guaranteed to work on high-dimensional sparse trajectories. Record local design-matrix singular values, local input covariance condition numbers, spatial distance to train data, MPC-vs-random state coverage, action-excitation bin counts and outcomes with reduced excitation. Factorization alone does not imply causal identification.

---

## 5. Experiment registry (pre-register before runs)

| ID | Phase / checkpoint | Data | Training? | Required outputs |
|---|---|---|---|---|
| E00 | Local Pendulum forensic verification | Existing Pendulum | NO | Class/ckpt/script/provenance PASS or FAIL |
| E01 | Repo protocol + loss/solver audit | Static upstream + smoke | NO | Reproduction manifest, effective config |
| E02 | CartPole smoke generation | `D-CART` 20 samples | NO models | Data integrity, generator hash |
| E03 | Pendulum faithful reproduction if E00 invalid | D-PEND | **Official Koopman only** | Eval model, logs, checkpoint |
| E04 | CartPole faithful reproduction | D-CART | **Official Koopman only** | Multi-seed if resources permit, paper table |
| E05 | Baseline margin & Jacobian diagnostics | D-PEND/D-CART | NO new models | `rho`, `G_z*`, `E_lift`, `d_orient` |
| E06 | Controlled rotation toy sweep | D-ROT | **Official Koopman only** | Response-vs-rotation, coverage |
| E07 | Unicycle & planar quadrotor | D-UNI/D-QUAD | **Official Koopman only** | Failure-mode report, open-loop vs MPC |
| E08 | MPC harness verification | Held-out | NO new models | QP formulation unit tests, bounded action |
| E09 | H2 identifiability study | Stress data | NO new methods first | Data excitation and theoretical diagnostics |
| G-H2 | Independent decision | E05–E09 | NO | GO / PIVOT / NO-GO with evidence |
| G-H3 | Independent decision | E05–E09 + theory | NO | Proof audit, margin and tractability assessment |
| H2-01 | Monolithic input-affine vs factorized | Protocol A | **New methods after G-H2 only** | Matched class/params/data/solver |
| H2-02 | Controlled-intervention optional | Protocol B | AFTER H2-01 | Information-budget ablation |
| H3-01 | S1, S2 and certified S3 | Protocol A | **New methods after G-H3 only** | Proof/certificate/compute trade-off |
| H3-02 | Planning and margin sweeps | Protocol A | AFTER H3-01 | delta/T/N/M/cost curves |

**Do not conflate:** E06/E07 train *the original official model on new environments*; this does **not** constitute H2/H3 model training. External models DKO/MAKO/etc are literature references until the later publication-competitiveness phase.

---

## 6. Experimental controls and acceptance criteria

**Reproduction:** determine paper's precise dataset budget, window sampling, optimizer, seed, loss flags, solver, and evaluation horizon. Log any mismatch rather than silently repair. Where source code is inconsistent, maintain: (i) untouched upstream snapshot, (ii) independent corrected runner, (iii) explicit diff and protocol classification. Do not call a local fix an exact upstream reproduction without qualification.

**Data integrity:** train/validation/test split by trajectory and by regime if applicable; checksum each `.pt`; avoid regenerating same path after results are viewed; fixed angle conventions and physical units. Oracle `f,G` cannot enter Protocol A train loop or model selection based on private test states.

**Statistical reporting:** pre-register at least 5 seeds for confirmatory ML experiments when feasible (smoke and protocol debugging may use one); report mean/std, confidence interval or bootstrap at trajectory level, plus all failures and negative results. Pre-register model/parameter budget, training updates, learning-rate schedule, ODE tolerances, horizon and MPC solver accuracy. No test-set hyperparameter tuning.

**H2 GO:** failure persists on at least one controlled state-dependent stress test after checking coverage/solver; matched-class monolithic and factorized comparison can be designed without privileged labels. H2 failure is not inferred from Pendulum-only or zero-action-only tests.

**H3 GO:** (i) verified proof/assumptions, (ii) nontrivial certified correction capacity with appropriate constrained factorization, (iii) a viable tractable controller (exact QP, clearly labeled approximate QP, or NMPC with acceptable compute/utility). Controllability rank alone is insufficient. `rho` can be tiny; that is a valid NO-GO outcome.

**Comparison semantics:** acceptance thresholds and compute budget are to be recorded **before** full experiments, not retrofitted to observed performance. P0/P1 success is about **protocol reproducibility**, not evidence of H2/H3 novelty.

---

## 7. Results, artifacts and provenance templates

Directory convention (actual project tree in `ROADMAP.md`):

```text
outputs/
  reproduction/{experiment_id}/{seed}/manifest.yaml
  reproduction/{experiment_id}/{seed}/metrics.json
  diagnosis/{experiment_id}/{seed}/metrics.json
  diagnosis/{experiment_id}/{seed}/plots/
  h2/{experiment_id}/{seed}/
  h3/{experiment_id}/{seed}/
```

Each experiment outputs `status={planned,running,failed,complete_verified}`, immutable config and seed, upstream/local commit IDs, environment versions, data checksum, checkpoint checksum, model class, metrics per trajectory and aggregated, stdout/stderr, failure codes, resource use and notes on deviations. `complete_verified` requires a second pass of metric recomputation from stored checkpoints, not merely finishing epochs.

**P0 evidence to request during future code verification:** current repository URL/branch/commit or zip, exact Pendulum command, local changed files, `data/` manifest or checksums (not whole huge data initially), YAML config, checkpoint `state_dict` key summary, stdout training+evaluation, effective package versions. Never upload secrets or private tokens.

---

## 8. Source and theory links

- [RRD v0.5](RRD_v0.5.md), especially §4–6, §9–11; RRD governs hypotheses and theorem assumptions.
- [Official ICLR 2026 publication](https://proceedings.iclr.cc/paper_files/paper/2026/file/e3781fb5d2c17ea7283cc2a543793bdb-Paper-Conference.pdf).
- [Upstream source snapshot](https://github.com/KYMiao/Controllable-Koopman/tree/ec131bb549c8f7b01519a99acca237b9fdc9c826).
- For originality and theory audit see references for Koopman inputs, controllability, LPV-MPC and nonlinear input transformations in `RRD_v0.5.md`. This protocol does not assert novelty.

**Status log — 2026-11-12:** v0.5 updated from H2-only plan. Pendulum self-reported completed, pending forensic verification; baseline-next is CartPole data+training. H2/H3 gates remain independent, with controlled stress tests prior to method design.
