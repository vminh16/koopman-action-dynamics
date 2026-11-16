# Experiment Tracking

Tài liệu này ghi lại các run phục vụ tái lập Pendulum và CartPole. Không ghi kết quả
paper vào cột kết quả của repo. Kết quả CartPole 5% nằm ở §3.4; kết quả
Pendulum và chẩn đoán P2 được giữ riêng. [Trạng thái P0/P1 hiện tại](docs/superpowers/2026-11-15-p0-p1-status.md): P0 forensic audit đã đóng; đợt 28 đối chứng P1 đã hoàn tất, exact paper reproduction chưa được chứng nhận.

P2 follow-up 2026-11-16: 140 paired runs và E05 refresh đã hoàn tất; H2/H3 vẫn NO-GO. [Báo cáo mới](outputs/diagnosis/p2_followup_2026-11-16/REPORT.md).


## 1. Reference values from paper — Pendulum

| Train fraction | Paper prediction MSE |
|---:|---:|
| 1% | 0.3747 |
| 5% | 0.0318 |
| 10% | 0.0114 |
| 30% | 0.0061 |

Các giá trị trên chỉ là reference. Kết quả của repo phải ghi kèm seed, config,
dataset và commit.

## 2. Protocol khóa cho Pendulum

| Field | Value |
|---|---|
| System | `pendulum` |
| Train trajectories | 5,000 |
| Test trajectories | 1,000 |
| Trajectory length | 250 |
| `dt` | 0.02 |
| Model | controllable Koopman |
| Latent dimension | 16 |
| Encoder | 3 layers, hidden size 64 |
| Prediction window | 15 |
| Batch size | 64 |
| Learning rate | `1e-3` |
| Control loss | enabled |
| Evaluation | open-loop, no re-encoding |
| Primary metric | mean physical-state MSE |

Train fractions phải được chọn theo trajectory:

| Fraction | Number of train trajectories |
|---:|---:|
| 1% | 50 |
| 5% | 250 |
| 10% | 500 |
| 30% | 1,500 |

## 3. Run registry

Status hợp lệ: `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`,
`NEEDS_INVESTIGATION`.

| Run ID | Fraction | Seed | Epochs | Status | Config | Checkpoint | Eval MSE | Notes |
|---|---:|---:|---:|---|---|---|---:|---|
| B01-s42 | 5% | 42 | 100 | COMPLETED | [config](configs/pendulum_mode-controllable_win15_frac0.05_dz16_hs64_nl3_ul2_uh64_MLPmodel.yaml) | [checkpoint](checkpoints/pendulum_mode-controllable_win15_frac0.05_dz16_hs64_nl3_ul2_uh64_MLPmodel.pt) | 0.106817 | Legacy config drift; see §3.1 |
| B01-s42-locked | 5% | 42 | 100 | COMPLETED | [config](configs/research/pendulum_controllable_win15_frac0.05_seed42.yaml) | [checkpoint](checkpoints/B01-s42-locked.pt) | 0.106817 | Locked config; exact reproduction of B01 |
| B01-s43 | 5% | 43 | 100 | COMPLETED | [config](configs/research/pendulum_controllable_win15_frac0.05_seed43.yaml) | [checkpoint](checkpoints/B01-s43.pt) | 0.105053 | Seed variance probe; see §3.2 |
| B01-s44 | 5% | 44 | 100 | COMPLETED | [config](configs/research/pendulum_controllable_win15_frac0.05_seed44.yaml) | [checkpoint](checkpoints/B01-s44.pt) | 0.009492 | Seed variance probe; see §3.2 |
| B02-s42 | 10% | 42 | 100 | COMPLETED | [config](configs/pendulum_mode-controllable_win15_frac0.1_dz16_hs64_nl3_ul2_uh64_MLPmodel.yaml) | [checkpoint](checkpoints/pendulum_mode-controllable_win15_frac0.1_dz16_hs64_nl3_ul2_uh64_MLPmodel.pt) | 0.099285 | Legacy config drift; see §3.1 |
| B02-s42-locked | 10% | 42 | 100 | COMPLETED | [config](configs/research/pendulum_controllable_win15_frac0.1_seed42.yaml) | [checkpoint](checkpoints/B02-s42-locked.pt) | 0.099285 | Locked config; exact reproduction of B02 |
| B02-s43 | 10% | 43 | 100 | COMPLETED | [config](configs/research/pendulum_controllable_win15_frac0.1_seed43.yaml) | [checkpoint](checkpoints/B02-s43.pt) | 0.006967 | Seed variance probe; see §3.2 |
| B02-s44 | 10% | 44 | 100 | COMPLETED | [config](configs/research/pendulum_controllable_win15_frac0.1_seed44.yaml) | [checkpoint](checkpoints/B02-s44.pt) | 0.007141 | Seed variance probe; see §3.2 |
| B03-s42 | 30% | 42 | 100 | COMPLETED | [config](configs/pendulum_mode-controllable_win15_frac0.3_dz16_hs64_nl3_ul2_uh64_MLPmodel.yaml) | [checkpoint](checkpoints/pendulum_mode-controllable_win15_frac0.3_dz16_hs64_nl3_ul2_uh64_MLPmodel.pt) | 0.006827 | Legacy config drift; see §3.1 |
| B03-s43 | 30% | 43 | 100 | PLANNED |  |  |  |  |
| B03-s44 | 30% | 44 | 100 | PLANNED |  |  |  |  |

### Completed results summary

| Fraction | Trajectories | Final train prediction loss | Test MSE | Paper MSE | Difference vs paper |
|---:|---:|---:|---:|---:|---:|
| 5% | 250 | 0.022495 | 0.106817 | 0.0318 | 3.36x |
| 10% | 500 | 0.047502 | 0.099285 | 0.0114 | 8.71x |
| 30% | 1,500 | 0.005205 | 0.006827 | 0.0061 | 1.12x |

The reported test MSE is the open-loop mean over 1,000 test trajectories with
`window_len=15` and `reencode=False`. Training losses were read from the final
TensorBoard event for each fraction. This historical summary table reports
seed 42 only; results for seeds 43/44 at 5% and 10% are recorded in §3.2.

### Current assessment

- The expected monotonic trend is present: `MSE(5%) > MSE(10%) > MSE(30%)`.
- The 30% result is close to the paper reference.
- The 5% and 10% results with seed 42 are substantially higher than the paper
  references, but seeds 43/44 are at or below them (§3.2). The gap is dominated
  by seed/protocol variance, not by the config drift.
- P0 is complete with verdict `CONFIG_DRIFT_CONFIRMED_PERFORMANCE_CAUSE_NOT_CONFIG`;
  the local runs are valid protocol-variant evidence.
- Aggregate reproduction remains open: 3 seeds are too few to report a mean
  because the outcome is bimodal.
- The legacy checkpoints/configs do not include an explicit seed field; the
  locked configs and manifests now do.

P0 audit manifest: [E01 manifest](outputs/reproduction/E01/manifest.yaml).

### 3.1 Config audit (2026-11-13)

The legacy configs record `lambda_control: 0.01`, but `experiments/run_train.py`
builds a subset `model_config` that omits this key. `KoopmanLitModule` therefore
used its default `0.005`. The training `hparams.yaml` for all three runs
confirms the effective value. No other drift was found in lr, loss flags,
dimensions, or optimizer.

Actions taken:

- New locked configs in `configs/research/` are the single source. The runner
  passes that exact dict to the model and aborts if any recorded value differs
  from the effective hyperparameter.
- Each manifest records seed, sampler version/source hash, git commit and
  working-tree state, dataset SHA256, and effective optimizer/scheduler/loss
  weights (`outputs/reproduction/B0*-s42-locked/manifest.yaml`).
- Existing checkpoints PASS the class/dimension/dataset-hash audit
  (`research/baseline/verify_checkpoint.py`). All three are
  `KoopmanLitModule`: encoder `2->64->64->64->14`, operator
  `A_free (15,16)`, `B_free (14,1)`, `H_head (2,2)`, `H_tail (14,14)`.
- Locked 5%/10% reruns reproduce B01/B02 exactly: the TensorBoard training
  curves are identical and eval MSE is `0.106817` / `0.099285`. The config
  drift is a provenance defect, not the performance cause.
- Metric probe on the same checkpoints: current first-window eval gives
  `0.106817` / `0.099285`; 20 random windows per trajectory gives `0.422812` /
  `0.368205`; all 236 windows over 100 trajectories gives `0.344123` /
  `0.300952`. The current metric is already optimistic, so the paper gap is not
  an evaluation-window artifact.
- Paper Table 10 with input encoder is `0.0067` / `0.0065`, but local
  `use_modulated_u=false`; compare against Table 1 (`0.0318` / `0.0114`).

Next single-variable checks, in order:

1. Seeds 43/44 at 5% and 10% to measure run-to-run variance. **Done, see §3.2.**
2. Constant learning rate / no `ReduceLROnPlateau`: the scheduler decays toward
   `1e-6..1e-8` while epoch-mean loss is flat.
3. Per-epoch random window resampling. The paper samples a window per
   trajectory; the current dataset fixes 20 windows when constructed.
4. Only if 1–3 fail, and as a separately labeled variant: remove the
   encoder-consistency losses, which are not in the paper's stated objective.

### 3.2 Seed variance probe (2026-11-13)

Same locked protocol, only `seed` changed. Seed also selects the 5%/10%
trajectory subset, so this measures full protocol variance, not just weight-init
variance. Eval is deterministic: `random.seed(42)` over 1000 test trajectories.

| Fraction | Seed 42 | Seed 43 | Seed 44 | Mean | Paper (Table 1) |
|---|---:|---:|---:|---:|---:|
| 5% | 0.106817 | 0.105053 | 0.009492 | 0.0738 | 0.0318 |
| 10% | 0.099285 | 0.006967 | 0.007141 | 0.0378 | 0.0114 |

Findings:

- Seed 42 is an outlier at both fractions. Seeds 43/44 are at or below the
  paper reference; 10% matches the paper's input-encoder numbers
  (`0.0067`/`0.0065`) even though `use_modulated_u=false`.
- The outcome is bimodal (training either converges well or stalls), so 3 seeds
  do not give a usable mean. The seed-42 10% run's train loss also rises from
  `0.0415` (step 7859) to `0.0518` (end), which points at optimizer/schedule
  instability rather than a config error.
- The P0 "3.4x/8.7x worse than paper" claim should be read as one unlucky seed,
  not a reproducible gap.

Next: extend to at least 5-10 seeds at 10% to estimate the failure rate, and
separate the data-subset seed from the model-init seed (currently one `seed`
controls both).

### 3.3. Phase P2 Controlled Stress Tests Summary

Phase P2 đã được thực hiện lại toàn diện và nghiêm ngặt với baseline chính thức `KoopmanLitModule` (thay thế surrogate baseline cũ), khắc phục triệt để lỗi lệch bước thời gian $(x_t, u_t) \to x_{t+1}$ có bao gồm $x_0$, đồng bộ $dt$ theo benchmark, và ghi nhận provenance đầy đủ. Kết quả P0 hoàn toàn được giữ nguyên không thay đổi. Full run hiện dùng 5 seeds, manifest theo run, artifact có hash, và cổng `p2_verification` chỉ PASS sau khi recompute từ checkpoint+dataset khớp báo cáo.

**Kết quả đo đạc chính thức (5 seeds 42–46; 30 train / 10 eval trajectories; 25 epochs; MPC horizon 12 $\Rightarrow T=0.24$ s cho rotation):**
- **Rotation family sweep ($a \in [0.0, 2.0]$):**
  - $a=0.0$: 1-step MSE $(2.82 \pm 0.05) \times 10^{-4}$, normalized $E_{\text{response}} = 1.009 \pm 0.008$, $E_{\text{phys}} = 1.005 \pm 0.004$, $E_{\text{lift}} = 1.362 \pm 0.086$, $d_{\text{orient,p95}} = 0.131 \pm 0.006$, $\rho_T = 0.00680 \pm 0.00010$.
  - $a=0.5$: 1-step MSE $(2.84 \pm 0.05) \times 10^{-4}$, $E_{\text{response}} = 1.021 \pm 0.007$, $E_{\text{phys}} = 0.996 \pm 0.011$, $E_{\text{lift}} = 1.306 \pm 0.081$, $d_{\text{orient,p95}} = 0.131 \pm 0.006$, $\rho_T = 0.00674 \pm 0.00005$.
  - $a=1.0$: 1-step MSE $(2.84 \pm 0.05) \times 10^{-4}$, $E_{\text{response}} = 1.032 \pm 0.004$, $E_{\text{phys}} = 0.991 \pm 0.022$, $E_{\text{lift}} = 1.251 \pm 0.075$, $d_{\text{orient,p95}} = 0.127 \pm 0.005$, $\rho_T = 0.00675 \pm 0.00008$.
  - $a=2.0$: 1-step MSE $(2.86 \pm 0.04) \times 10^{-4}$, $E_{\text{response}} = 1.050 \pm 0.003$, $E_{\text{phys}} = 1.007 \pm 0.026$, $E_{\text{lift}} = 1.292 \pm 0.182$, $d_{\text{orient,p95}} = 0.121 \pm 0.005$, $\rho_T = 0.00674 \pm 0.00006$.
- **Unicycle:** 1-step MSE $(1.002 \pm 0.027) \times 10^{-4}$, $E_{\text{phys}} = 1.280 \pm 0.048$, $E_{\text{lift}} = 1.290 \pm 0.047$, $d_{\text{orient,p95}} = 0.244 \pm 0.012$, candidate regret $0.041 \pm 0.046$.
- **Planar Quadrotor:** 1-step MSE $0.3821 \pm 0.0032$, $E_{\text{phys}} = 35.07 \pm 0.19$, $E_{\text{lift}} = 35.16 \pm 0.19$, $d_{\text{orient,p95}} = 0.331 \pm 0.111$, candidate regret $113.5 \pm 40.8$.
- **Convex QP MPC (rotation, 5 seeds):** OSQP thành công 100%, 0 fallback, vi phạm input $=0$, p95 latency $\approx 3.8-3.9$ ms $< 50$ ms. Chi phí: $168.4 \pm 4.5$ ($a=0.0$) vs $267.9 \pm 4.4$ ($a=2.0$), so với zero-control $247.8$ và LTI-oracle $124.2$ / $380.9$. E08 xác nhận ZOH vs Euler khớp nhau ($\text{RMSE } 0.0180$ vs $0.0181$), residual primal/dual được log, và fallback chỉ dùng lệnh chiếu hợp lệ.
- **E09 excitation control:** giảm excitation làm input covariance suy biến (singular value $=0$, condition $=\infty$) nhưng $E_{\text{response}}$ gần như không đổi ($1.216$ vs $1.235$), tức chưa thấy failure mode do excitation trong ngân sách này.
- **E05:** Pendulum checkpoint+config+dataset hash load và chạy được; CartPole được ghi nhận `missing_checkpoint` (không có config/checkpoint đã verify), nên E05 là `partial` chứ không bị ẩn.
- **Phán quyết Gates chính thức:**
  - **Gate P2: PASS**; **P2 verification: PASS** sau recompute độc lập từ dataset+checkpoint (SHA-256, strict load, tolerance $10^{-6}$).
  - **Gate H2 (Fixed-B limitation): NOT_CONFIRMED** (tỷ lệ tăng trưởng đáp ứng $E_{\text{response}}(a=2.0)/E_{\text{response}}(a=0.0) = 1.030 < 1.5$; đơn điệu nhưng rất yếu).
  - **Gate H2 Readiness: NO-GO** (chưa đủ bằng chứng xác nhận giới hạn fixed-B để mở track H2).
  - **Gate H3: NO-GO** (biên điều khiển $\rho_T \approx 0.0068$, trong khi $d_{\text{orient,p95}} \approx 0.121-0.131 > \rho_T$; không tồn tại cửa sổ chứng chỉ khả thi với $\sigma(M) \in [0.5, 2.0]$).
- Báo cáo chẩn đoán chi tiết: [p2_failure_report.yaml](outputs/diagnosis/p2_failure_report.yaml); artifact và manifest theo run: `outputs/diagnosis/runs/<run_id>/`.
- **Sửa lỗi số học upstream:** `training/loss.py` có số hạng condition-number nhân bởi `beta=0`; giá trị loss không đổi nhưng backward đi qua $1/(\lambda_{\min}+10^{-8})$ và sinh NaN gradient sau ~230 bước full-budget (gặp trên quadrotor seed 45). Đã bỏ số hạng chết này; training 25 epochs hoàn tất.


### 3.3.1. P2 follow-up — E05 refresh and P1-derived controls (2026-11-16)

**Execution: COMPLETED; evidence verification: PASS; H2/H3 remain NO-GO.**

- E05 now explicitly registers the locked `.yml` CartPole C02 config. Refreshed
  Pendulum seed 42 and C02 seeds 42/43/44: strict checkpoint loading, dataset hashes,
  repeat computation and later recomputation all match. The historical snapshot
  above remains unchanged; its `missing_checkpoint` entry is superseded by this
  separate E05 refresh. CartPole still has `rho_T=null` at T=.3 s under the numerical
  rank tolerance, consistent with P1's independently verified algebraic output rank 4.
- Completed **140 paired rotation runs / 65,000 optimizer updates**:
  five seeds 42–46, four strengths; five diagnostic arms at 25 epochs and baseline/
  combined arms at 100 epochs. Reused frozen P2 datasets, windows and shared probe.
  All 40 longer runs reproduce their first 25 epochs exactly. Explicit shuffle RNG
  differs from historical Lightning ordering; compare new paired baselines.
- The constant-map positive control a=0 still has normalized response error
  **1.014627** for baseline at 100 epochs and **0.954208** for the combined diagnostic
  variant (five-seed means). Ignoring inputs gives response error 1. Thus small
  one-step MSE does not demonstrate learned action-response fidelity.
- Independent decomposition separates the weighted fixed-map floor (0 at a=0,
  0.845222 at a=2) from learned-map fitting error. The old one-step-MSE threshold
  .05 admits an exact-drift predictor that ignores inputs (MSE about .000269).
  Original SPD heads also cannot exactly represent the a=0 map [1,0]^T because
  their second physical input coefficient is at least .1. That restriction alone
  only proves a .01 response-error floor, not the observed error near 1.
- The weak stress ratio therefore does **not isolate** fixed-B limitation from
  baseline fitting/parameterization. The mathematical floor is verified, while
  superiority of a new learning mechanism remains untested. Historical gates are
  preserved; no H2/H3 method was trained. Unicycle/Quadrotor were not retrained in
  this follow-up, and MPC results are not transferred to diagnostic variants.
- Checks: 140 checkpoint metric recomputations, source/data/checkpoint hashes,
  float64 output rank 2 on all rotation checkpoints, Gramian block exponentials
  versus adaptive quadrature, E05 recomputation and **80 tests passed**.

Evidence: [full report](outputs/diagnosis/p2_followup_2026-11-16/REPORT.md),
[summary](outputs/diagnosis/p2_followup_2026-11-16/summary.json),
[verification](outputs/diagnosis/p2_followup_2026-11-16/verification.json),
[figure](outputs/diagnosis/p2_followup_2026-11-16/results.png).

### 3.3.2. P2 config check — compatible positive control (2026-11-16)

Protocol mới đã chạy 22 runs / 18.100 updates; known-answer và finite-difference
checks PASS; 32 checkpoint evaluations và 15 test evaluations recompute khớp.
Baseline/loss gốc học được compatible control, nhưng config được chọn bằng
validation seed42 (LR=.01, constant,100 epochs) chỉ đạt 4/5 seeds. Cả SPD/general
trên hệ tọa độ gốc chưa đạt với ngân sách đã khóa. Stress không chạy vì
prerequisite 5/5 chưa đạt; không đổi H2/H3 gates. Không kết luận các config khác
đều thất bại: LR=.001/1000 đạt trên seed tune nhưng chưa được xác nhận đa seed.
[Protocol và báo cáo ngắn](docs/superpowers/2026-11-16-p2-config-check.md).

### 3.4. CartPole 5% — completed runs and independent evaluation (2026-11-14)

**Training status: COMPLETED (C02-s42/s43/s44). Reproduction assessment: NEEDS_INVESTIGATION.**

Mỗi run dùng 500/10.000 train trajectories (5%), 20 windows/trajectory,
10.000 windows dài 15 bước; 100 epochs và 15.700 optimizer steps. Model là
`KoopmanLitModule`, controllable, `N=25`, encoder 3 layers × hidden 64,
`use_modulated_u=false`, batch 64, AdamW với LR ban đầu `0.001`,
`weight_decay=1e-5`, scheduler `ReduceLROnPlateau`, ODE solver Euler, `dt=0.02`.
Runner `research.baseline.train_koopman` truyền toàn bộ locked YAML và LR vào
module, dùng encoder/operator/loss của tác giả; wrapper khác entrypoint upstream
ở config/provenance và ghi quyết định sampling. Đây là protocol repo đã khóa,
chưa được chứng nhận tương đương hoàn toàn với paper.

| Run ID | Seed | Epochs | Training status | Config | Checkpoint | Final train prediction loss | Test MSE open-loop H=15 |
|---|---:|---:|---|---|---|---:|---:|
| C02-s42 | 42 | 100 | COMPLETED | [config](configs/research/cartpole_controllable_win15_frac0.05_seed42.yml) | [checkpoint](checkpoints/C02-s42.pt) | 0.02497466 | 0.09289890 |
| C02-s43 | 43 | 100 | COMPLETED | [config](configs/research/cartpole_controllable_win15_frac0.05_seed43.yml) | [checkpoint](checkpoints/C02-s43.pt) | 0.02418091 | 0.08918348 |
| C02-s44 | 44 | 100 | COMPLETED | [config](configs/research/cartpole_controllable_win15_frac0.05_seed44.yml) | [checkpoint](checkpoints/C02-s44.pt) | 0.02600020 | 0.09360111 |

#### Evaluation protocol and seed aggregates

Đánh giá mới được tính từ checkpoint epoch cuối trên cùng toàn bộ 1.000 test
trajectories, bắt đầu từ `x0`; không chọn checkpoint theo test MSE. Metric là
MSE không trọng số trên bốn chiều trạng thái vật lý, thời gian và trajectories;
không wrap góc hoặc normalize state. Model forward float32, giảm tổng sai số
float64. Training prediction loss có trọng số thời gian nên không đồng nhất
với test MSE này. Rollout batch đã đối chiếu với một trajectory chạy riêng.

| Protocol | Seed 42 | Seed 43 | Seed 44 | Mean ± sample std | 95% t CI (df=2) |
|---|---:|---:|---:|---:|---|
| Open-loop H=15 (0.3 s) | 0.09289890 | 0.08918348 | 0.09360111 | 0.09189450 ± 0.00237392 | [0.08599736, 0.09779163] |
| Open-loop H=250 (5 s) | 50.59788465 | 43.17358069 | 39.56642541 | 44.44596358 ± 5.62472149 | [30.47338080, 58.41854637] |
| Re-encode H=250, K=50 | 65.99446461 | 67.95760022 | 61.97731586 | 65.30979356 ± 3.04836531 | [57.73723433, 72.88235280] |

`std` và CI tính trên ba seed, không phải trên test trajectories; ba seed
cho ước lượng bất định còn hạn chế. Seed thay đổi cả train subset và model init.
Re-encode dùng trạng thái **dự đoán** trước đó mỗi 50 bước; không reset bằng
ground truth. Sai số tăng mạnh ở horizon dài và re-encode K=50 kém hơn open-loop
H=250 ở cả ba seed. Không so các MSE này trực tiếp với reference Pendulum ở §1.

#### Structure and independent float64 Gramian/rank checks

Cả ba checkpoint pass strict load, `A=P Ac P^-1`, `B=P Bc`, và `P` positive
definite. Kiểm tra độc lập `rank(C[B, AB, A²B, A³B])=4`; chuẩn hóa scale của
`A` trước khi lập 25 khối cũng cho output rank 4. Rank 2 từ ma trận 25 khối
chưa chuẩn hóa chịu ảnh hưởng dynamic range nên không dùng kết luận mất
output controllability.

| Seed | Output rank (first 4 blocks) | Wy(0.3 s) numerical rank, tol=1e-10 | Minimum eigenvalue Wy |
|---:|---:|---:|---:|
| 42 | 4/4 | 3/4 | 4.38851276e-12 |
| 43 | 4/4 | 3/4 | 2.50216108e-12 |
| 44 | 4/4 | 3/4 | 4.74258936e-12 |

Gramian tính float64 bằng block matrix exponential và đối chiếu adaptive
quadrature độc lập. Eigenvalue nhỏ nhất dương nhưng rất nhỏ: `Wy` gần suy biến
về mặt số học tại horizon 0.3 s; `rho_T=null` theo tolerance đã công bố.
Output rank đủ không đồng nghĩa biên điều khiển tốt trong horizon ngắn, hoặc
bảo đảm điều khiển hệ phi tuyến thật dưới giới hạn input.

#### Provenance, artifacts and remaining gates

- Training source: `5701981ce63517637f9e7cdbb1dafc7bf68e9d6f`; cả ba manifest ghi source clean.
- Train SHA-256: `f016889ea0090a91c31c545ec37ca25fcaee8d6658d2364bebabb8e0ecea449b`.
- Test SHA-256: `9b8f99b2445f8c6e737f683d544e3efe357a9f21b75e006948541cfdbd5ef35e`.
- Đã đối chiếu config/checkpoint/dataset hashes, effective hyperparameters và sampler-source hash với từng manifest.
- Manifest: [s42](outputs/reproduction/C02-s42/manifest.yaml), [s43](outputs/reproduction/C02-s43/manifest.yaml), [s44](outputs/reproduction/C02-s44/manifest.yaml). Mỗi thư mục còn có `evaluation.json` và `controllability.json`.
- Summary: [Markdown](outputs/reproduction/cartpole_frac0.05_summary.md), [JSON](outputs/reproduction/cartpole_frac0.05_summary.json), [CSV](outputs/reproduction/cartpole_frac0.05_summary.csv).
- Seed 42 từng có log chưa hoàn tất ở `version_0` (checkpoint epoch 12); bảng dùng checkpoint cuối `C02-s42.pt` của run hoàn tất, không dùng checkpoint dở dang.
- **Dataset audit vẫn FAILED:** train RK4 max error `0.000160761215 > 0.0001`; không đổi dữ liệu/hash/tolerance. Shape, finiteness, actuator/initial bounds, mutation/autograd và train/test separation đã pass. [Audit report](outputs/reproduction/cartpole_data_audit.yaml).
- Chưa có kết quả CartPole 1%, 10%, 30% trong đợt này. Cần xử lý/giải thích audit transition và đối chiếu paper protocol trước khi kết luận reproduction PASS.
- Các checkpoint CartPole mới chưa được đưa vào E05/recompute P2. Dòng `missing_checkpoint` ở §3.3 mô tả snapshot P2 trước đợt train này; P2 report và gates không tự thay đổi.

### 3.5. P1 controlled causal audit (2026-11-15)

**28 runs × 100 epochs; 353.800 optimizer updates. Verdict: ROOT_CAUSES_SUPPORTED,
EXACT_PAPER_REPRODUCTION_NOT_CERTIFIED.** Các checkpoint/dataset C02 cũ được giữ
nguyên; các run mới có namespace riêng dưới
`outputs/reproduction/p1_root_cause_2026-11-15/`.

Đối chứng cùng init/subset/windows và thứ tự minibatch; 200 validation trajectories
nằm ngoài hợp của ba tập train. Bảng sau là test MSE H=15 trên toàn bộ 1.000
trajectories, mean ± sample std qua seed 42/43/44, checkpoint epoch cuối:

| Dataset | Paired baseline | Variant | Variant MSE |
|---|---:|---|---:|
| Pendulum 5% | 0.08619425 ± 0.04338556 | Bỏ cả hai encoder-consistency penalties | 0.01447501 ± 0.01196399 |
| CartPole 5% | 0.09179632 ± 0.00259836 | Đổi dấu toàn bộ physical transform block; giữ random windows | 0.06004065 ± 0.00180977 |

- Cả ba seed đều cải thiện trong từng cặp. Baseline đối chứng mới dùng CPU và
  shuffle generator tường minh; không thay thế số liệu lịch sử ở §3.4.
- Pendulum: bỏ encoder loss cải thiện H=15 nhưng làm H=250 kém đi (mean MSE
  51.17 → 475.52). Sửa scheduler monitor sang epoch mean cũng cải thiện seed 42;
  các thay đổi không có tác dụng cộng tuyến tính, nên không áp dụng hàng loạt.
- CartPole: ràng buộc SPD buộc angular input gain dương, trái với gain âm gần
  upright. Đối chứng âm physical block vẫn giữ output rank 4/4; sửa dấu một
  mình chưa đủ đạt reference paper.
- Đối chứng sampler seed 42, validation MSE: original/random 0.09105911;
  original/start0 0.09586752; negative/random 0.05909300; negative/start0
  0.00084898. Start0 thay cả miền trạng thái và số windows duy nhất.
- Negative/start0 lặp ba seed cho test H=15 **0.00087518 ± 0.00012415**, nhưng
  đây là **DIAGNOSTIC_ONLY_NOT_FAITHFUL_PAPER_REPRODUCTION** vì khác random-window
  sampling được paper mô tả. Không lấy số thấp này để đánh dấu P1 PASS.
- Bỏ riêng softplus Gramian penalty gần như không đổi validation MSE seed 42;
  khác biệt objective với paper vẫn tồn tại nhưng chưa chứng minh là nguyên
  nhân chính của prediction gap. Rank/Gramian float64 được kiểm tra độc lập.
- Chưa có protocol Table 1 đủ tường minh để chứng nhận tái lập chính xác; không
  sửa verdict dataset audit cũ hoặc các gates P2/H2/H3.

Bằng chứng: [audit note](docs/superpowers/2026-11-15-p1-controlled-reproduction-audit.md),
[full report](outputs/reproduction/p1_root_cause_2026-11-15/REPORT.md),
[protocol](outputs/reproduction/p1_root_cause_2026-11-15/PROTOCOL.md),
[JSON](outputs/reproduction/p1_root_cause_2026-11-15/summary.json),
[plots](outputs/reproduction/p1_root_cause_2026-11-15/causal_results.png).

## 4. Commands

### 4.0. Artifact policy and replay after cleanup (2026-11-16)

Code lives in `research/`; generated results stay ignored under `outputs/`.
Cleanup removes Git tracking, **not local files**. Keep original Pendulum/CartPole
inputs and root checkpoints; legacy input-generation provenance is incomplete, so
never replace them with freshly generated data and claim identical reproduction.
CartPole inputs must be supplied separately with the hashes in locked configs.
No Git history was rewritten; old commits still contain archived artifacts.
Cleanup verification: 407 generated files (14.86 MiB) removed from Git tracking;
tracked contents drop from 46.23 to about 31.4 MiB. All 322 previously tracked `.pt`
files remain byte-identical locally. 80 tests pass. A temporary index-only checkout
(with original CartPole inputs supplied separately) restored all 36 historical P2
dataset hashes and recomputed P2, regenerated P1 evaluation from 28 retained
checkpoints, and rebuilt/verified all 22 config-check runs. The 140-run follow-up
training was not repeated during cleanup; its replay code is unchanged.

| Evidence retained in Git | Rebuilt/ignored artifacts | Replay below |
|---|---|---|
| P0/C02 checkpoints, configs, audit reports | TensorBoard, plots, new fraction runs | Existing locked train/eval commands in §4 |
| P1: all 28 final checkpoints, manifests/results, protocol, concise summary | Progress duplicates, per-trajectory evaluations, large summaries/plots | P1 |
| Historical P2: report, checkpoints, E09 control data | Deterministic E06/E07 datasets and plots | Restore P2 inputs, then historical recompute |
| P2 follow-up: summary/provenance; a=0 baseline/combined e100 seeds42–46 | Remaining 130 runs and figure | P2 follow-up |
| Config check: protocol/selection/gates; 5 confirmations, seed45 original controls, .001 long-budget candidates and tiny overfit | Other tuning/comparison runs, synthetic inputs | Config check |

Checkpoint-bearing retained run records are kept with **every referenced checkpoint**.
`recompute` alone needs all inputs/runs; replay missing artifacts before verifying.
Old reports/source hashes describe their capture time. P1 active scripts moved from
`outputs/reproduction/.../*.py` to `research/diagnostics/p1_audit/`; captured protocols
remain unchanged. Curate future essential checkpoints explicitly, not whole folders.

Common environment (repository root; installed dependencies from `requirements.txt`):

```bash
export PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1
```

**Restore historical P2 synthetic inputs.** Regenerate in a temporary directory,
verify every SHA against the frozen report, and copy only missing files. Existing
inputs are verified, never overwritten. This also supplies the follow-up's inputs.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil, tempfile, yaml
from research.baseline.common import ROOT, sha256_file
from research.baseline.data_generator import generate_benchmark_dataset
report = yaml.safe_load((ROOT/'outputs/diagnosis/p2_failure_report.yaml').read_text())
checked = 0
with tempfile.TemporaryDirectory() as temporary:
    for group in ('rotation_sweep', 'benchmark_evaluations'):
        for name, entry in report[group].items():
            system = 'rotation_family' if group == 'rotation_sweep' else name
            kwargs = {'a': entry['a']} if system == 'rotation_family' else {}
            datasets = [entry['datasets']['eval'], *entry['datasets']['train'].values()]
            for item in datasets:
                source, digest, _ = generate_benchmark_dataset(
                    system, item['num_trajectories'], item['T'], item['dt'],
                    item['seed'], save_dir=temporary, sys_kwargs=kwargs)
                assert digest == item['sha256'], item['path']
                target = ROOT/item['path']
                if target.exists():
                    assert sha256_file(target) == digest, target
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                checked += 1
# Derive a local-path report for clones at a different filesystem location.
# The frozen report, scientific values and artifact hashes remain unchanged.
for group in ('rotation_sweep', 'benchmark_evaluations'):
    for entry in report[group].values():
        for seed, recorded in entry['seeds'].items():
            checkpoint = recorded['training']['checkpoint']
            run = Path(entry['datasets']['train'][seed]['path']).parent.parent
            checkpoint['path'] = str(ROOT/run/'checkpoints'/Path(checkpoint['path']).name)
(ROOT/'outputs/diagnosis/p2_replay.yaml').write_text(yaml.safe_dump(report))
print(f'{checked} historical dataset hashes verified; local replay report written')
PY
.venv/bin/python -m research.diagnostics.recompute_p2 \
  --report outputs/diagnosis/p2_replay.yaml
```

**P1 causal audit.** All 28 checkpoints are retained. Rebuild derived evaluation
and reports without retraining:

```bash
.venv/bin/python research/diagnostics/p1_audit/experiment.py --verify
.venv/bin/python research/diagnostics/p1_audit/verify_geometry.py
for system in pendulum cartpole; do
  .venv/bin/python research/diagnostics/p1_audit/evaluate_final.py --system "$system"
done
.venv/bin/python research/diagnostics/p1_audit/evaluate_start0.py
.venv/bin/python research/diagnostics/p1_audit/summarize.py
.venv/bin/python research/diagnostics/p1_audit/plot_results.py
```

To retrain missing P1 checkpoints, use the **same** output paths and 100 epochs;
completed run records skip training. Restore both checkpoint and metadata together,
or use a separate output/archive for a fresh repetition.

```bash
p1=research/diagnostics/p1_audit
for seed in 42 43 44; do
  for system in pendulum cartpole; do
    .venv/bin/python "$p1/experiment.py" --system "$system" --arm baseline --seed "$seed"
  done
  .venv/bin/python "$p1/experiment.py" --system pendulum --arm noenc --seed "$seed"
  .venv/bin/python "$p1/geometry_experiment.py" --system cartpole --arm negative_head --seed "$seed"
  .venv/bin/python "$p1/sampling_experiment.py" --arm negative_head_start0 --seed "$seed"
done
for arm in constant epochmean noenc_epochmean; do
  .venv/bin/python "$p1/experiment.py" --system pendulum --arm "$arm" --seed 42
done
.venv/bin/python "$p1/geometry_experiment.py" --system pendulum --arm general_head --seed 42
for arm in signed noenc epochmean; do
  .venv/bin/python "$p1/experiment.py" --system cartpole --arm "$arm" --seed 42
done
for arm in general_head general_head_epochmean general_head_noenc; do
  .venv/bin/python "$p1/geometry_experiment.py" --system cartpole --arm "$arm" --seed 42
done
for system in pendulum cartpole; do
  .venv/bin/python "$p1/gramian_experiment.py" --system "$system" --seed 42
done
.venv/bin/python "$p1/sampling_experiment.py" --arm baseline_start0 --seed 42
```

Static numerical probes: `.venv/bin/python research/diagnostics/p1_audit/static/<probe>.py`;
`sign_repro.py` intentionally fails on the original wrong-sign parameterization.
`p1_tb_probe.py` requires original ignored Lightning logs; its retained JSON is
forensic evidence, not a promise to recreate unknown historical training.

**P2 follow-up, 140-run protocol.** Restore historical inputs above first; 10 retained
run/checkpoint pairs skip training, missing 130 runs are reconstructed locally.

```bash
.venv/bin/python -m research.diagnostics.p2_followup --stage verify
.venv/bin/python -m research.diagnostics.p2_followup --stage e05
.venv/bin/python -m research.diagnostics.p2_followup --stage train
.venv/bin/python -m research.diagnostics.p2_followup --stage recompute
.venv/bin/python -m research.diagnostics.p2_followup_math
.venv/bin/python -m research.diagnostics.recompute_p2 \
  --report outputs/diagnosis/p2_replay.yaml \
  --output outputs/diagnosis/p2_followup_2026-11-16/historical_recompute.json
.venv/bin/python -m research.diagnostics.verify_p2_followup
.venv/bin/python -m research.diagnostics.summarize_p2_followup
```

**Config check, 22-run conditional protocol.** Restore inputs for retained runs
first (the training runner skips completed results). The same split seeds, configs,
selection rule and 5/5 gate apply; a blocked stress stage remains a valid outcome.

```bash
.venv/bin/python - <<'PY'
from research.diagnostics.p2_config_check import generate
for seed in range(42, 47):
    for compatible in (True, False):
        for split in ('train', 'validation', 'test'):
            generate(seed, split, compatible)
generate(42, 'train', solver='euler', count=2, T=30)
PY
.venv/bin/python -m research.diagnostics.p2_config_check --stage all
.venv/bin/python -m research.diagnostics.p2_config_check --stage recompute
.venv/bin/python -m research.diagnostics.report_p2_config_check
```

Full historical P2 retraining is optional, rather than a prerequisite for reuse of
retained checkpoints. Use a separate output directory to preserve the frozen report:

```bash
.venv/bin/python -m research.diagnostics.run_p2_diagnostics \
  --output_dir outputs/diagnosis/new_p2_run
```

Training reproducibility depends on the recorded package versions/device; a fresh
run is a new provenance record, not a promise of bitwise-identical checkpoints.
Dataset restoration above requires exact hashes and stops on version/solver drift.


Run from the repository root with the project environment activated:

```bash
source .venv/bin/activate
mkdir -p checkpoints configs figures
```

Train a locked run (new canonical entrypoint):

```bash
PYTHONPATH=. python -m research.baseline.train_koopman \
    --config configs/research/pendulum_controllable_win15_frac0.05_seed42.yaml \
    --dry-run   # validate config/data/model contract first
PYTHONPATH=. python -m research.baseline.train_koopman \
    --config configs/research/pendulum_controllable_win15_frac0.05_seed42.yaml \
    2>&1 | tee /tmp/B01-s42-locked.train.log
```

Evaluate and verify the resulting checkpoint:

```bash
PYTHONPATH=. python experiments/run_eval.py \
    --config configs/research/pendulum_controllable_win15_frac0.05_seed42.yaml \
    --window_len 15 \
    --max_traj 1000 \
    --num_plot 0

PYTHONPATH=. python -m research.baseline.verify_checkpoint \
    --config configs/research/pendulum_controllable_win15_frac0.05_seed42.yaml \
    --hparams outputs/lightning_logs/B01-s42-locked/version_0/hparams.yaml
```

Run the config contract self-check:

```bash
PYTHONPATH=. python -m tests.test_config_contract
```

Run the P2 diagnosis (quick = smoke only; full writes a verifiable run):

```bash
PYTHONPATH=. python -m research.diagnostics.run_p2_diagnostics --quick
PYTHONPATH=. python -m research.diagnostics.run_p2_diagnostics
PYTHONPATH=. python -m research.diagnostics.recompute_p2 \
    --report outputs/diagnosis/p2_failure_report.yaml --tolerance 1e-6
```

Legacy entrypoint (`experiments/run_train.py`) is retained only for the old
runs. It writes the YAML **after** training from CLI arguments, not from the
effective module, so do not use it for new evidence.

Create the log directory before using `tee`:

```bash
mkdir -p logs
```

## 5. Per-run record

Copy this section for every completed or failed run.

### RUN_ID

#### Identity

| Field | Value |
|---|---|
| Run ID |  |
| Date/time |  |
| Git commit |  |
| Branch |  |
| Python |  |
| PyTorch |  |
| CUDA/GPU |  |
| Host |  |
| Seed |  |

#### Dataset

| Field | Value |
|---|---|
| Train file |  |
| Test file |  |
| Train trajectories used |  |
| Test trajectories evaluated |  |
| Train fraction |  |
| Dataset SHA256 |  |
| Trajectory selection seed | 42 |
| Window length | 15 |
| Windows per trajectory | 20 |

#### Configuration and artifacts

| Artifact | Path |
|---|---|
| YAML config |  |
| Checkpoint |  |
| Training log |  |
| Evaluation log |  |
| Figures |  |

#### Results

| Metric | Value |
|---|---:|
| Mean MSE |  |
| Median trajectory MSE |  |
| MSE standard deviation |  |
| MSE 95% CI |  |
| MSE horizon 1 |  |
| MSE horizon 5 |  |
| MSE horizon 15 |  |
| Training time |  |
| Peak GPU memory |  |

#### Reproduction assessment

- Status: `PASS` | `NEEDS_INVESTIGATION` | `FAIL`
- Comparison with paper:
- Deviations from paper:
- Suspected causes:
- Follow-up:

## 6. Required commands for provenance

Run these at the start of each experiment:

```bash
git rev-parse HEAD
git status --short
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)"
sha256sum data/pendulum_train.pt data/pendulum_test.pt
```

### Recorded provenance for completed runs

| Field | Value |
|---|---|
| Dataset train SHA256 | `564c6da6dfbb61f2aecc09f8b881a65274d14aeb8ba1ae59a29c49ea1602057f` |
| Dataset test SHA256 | `c7c9da6ab89d0549a531ba060109caa09642e187e841535b8dae807d6e0a5020` |
| Git HEAD when recorded | `ec131bb549c8f7b01519a99acca237b9fdc9c826` |
| Python environment | `.venv`, Python 3.10.12 |
| PyTorch | 2.14.1+cu130 |
| GPU | NVIDIA GeForce RTX 2050 |
| Trajectory-selection seed | 42 |

## 7. Acceptance rule

Do not mark Pendulum reproduction as `PASS` from one run or one seed.

Mark B01/B02/B03 as complete only when:

1. The requested trajectory fraction is correct.
2. The test set remains untouched during training and selection.
3. Seeds 42, 43 and 44 have completed.
4. Open-loop evaluation uses the same fixed test set.
5. Mean, standard deviation and confidence interval are recorded.
6. The trend is consistent with the paper:

```text
MSE(5%) > MSE(10%) > MSE(30%)
```

If the absolute values differ, record the difference and investigate the
optimizer, loss coefficients, ODE solver, sampling protocol and checkpoint
selection before claiming reproduction.

### P2 robust-flow protocol — locked before execution (2026-11-16)

Goal: reliable baseline/experimental comparisons, not matching paper scores.
[Protocol](docs/superpowers/2026-11-16-p2-robust-flow.md). Dev seeds42–44;
LR .001/.01, fixed1000 epochs, original losses, 8/8/16 train/val/test trajectories.
Select on multi-seed validation, then freeze for NEW confirmation seeds52–56
(compatible, original SPD/general). Nonlinear general sweep .5/1/2 opens only
when dev and both prerequisite confirmation groups pass. No test-based selection.
Source/data hashes: `outputs/diagnosis/p2_robust_flow_2026-11-16/source_snapshot.json`
and compact `summary.json`; RNG offsets10000/20000/30000+seed, windows seed=seed,
shuffle700000+seed. Raw runs/datasets stay ignored; curate only final confirmation
checkpoints and concise evidence. No H2/H3 model training or historical gate edits.

```bash
export PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1
sha256sum research/diagnostics/p2_robust_flow.py research/diagnostics/p2_config_check.py
.venv/bin/python -m research.diagnostics.p2_robust_flow --stage all
.venv/bin/python -m research.diagnostics.p2_robust_flow --stage verify-saved
```

`all` reuses completed local raw runs; on a fresh checkout it rebuilds ignored
inputs and raw runs. `verify-saved` regenerates inputs and verifies only curated
final checkpoints, with the source and frozen protocol hashes checked first.

Post-test diagnostic for this protocol (no reselection or gate changes): decompose
physical vector-field error on the five compatible test splits into drift, input
and cross terms. This is not a bound on rollout MSE. Reuse archived checkpoints;
source/data identities are in the summary and source snapshot above.

```bash
.venv/bin/python - <<'PY'
import torch
from research.diagnostics import p2_robust_flow as m
m.c.OUT=m.OUT
torch.set_num_threads(1)
rows=m.load(m.OUT/'summary.json')['runs']; result={}
for row in rows:
    if not row['name'].startswith('confirm-compatible-'): continue
    train,_=m.c.generate(row['seed'],'train',True)
    test,identity=m.c.generate(row['seed'],'test',True)
    model=m.c.make_model(row['seed'])
    checkpoint=row['archived_checkpoint']
    assert m.sha256_file(m.ROOT/checkpoint['path'])==checkpoint['sha256']
    model.load_state_dict(torch.load(m.ROOT/checkpoint['path'],weights_only=True))
    X=torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in test])
    U=torch.cat([r['u_seq'] for r in test]).double()
    T=torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in train]).double()
    with torch.no_grad():
        A,B=m.c.matrices(model); z=model.model.encoder(X).double()
        drift=(z@A.double().T)[:,:2]-X.double()@torch.tensor(m.c.F.T)
        inputs=U@(B[:2].double()-torch.tensor([[0.],[1.]],dtype=torch.float64)).T
        total=float((drift+inputs).square().mean())
        d=float(drift.square().mean()); i=float(inputs.square().mean())
        cross=float((2*drift*inputs).mean())
        assert abs(total-d-i-cross)<1e-10
        scale=T.std(0).clamp_min(1e-8)
        coverage=torch.cdist(X.double()/scale,T/scale).min(1).values
    result[str(row['seed'])]={'test_data':identity,'drift_mse':d,'input_mse':i,
        'cross_term':cross,'total_vector_field_mse':total,
        'nearest_train_distance_p95':float(torch.quantile(coverage,.95))}
m.c.save(m.OUT/'posthoc_vector_field.json',{'scope':'POSTHOC_DIAGNOSTIC_ONLY_NO_RESELECTION',
    'runs':result})
print(result)
PY
```

Robust-flow result: 21 finite runs /63,000 updates; 63 checkpoint-validation
recomputations and15 curated-final validation/test verifications. Compatible
validation5/5, test response5/5, joint prediction+response4/5 (seed56 H15ratio .284>.25).
Original SPD/general each0/5; nonlinear sweep blocked as declared. Posthoc
vector-field decomposition shows larger drift than input error on compatible test;
seed56 drift=.01430 vs input=.000360. No config reselection. 82 tests PASS.
See [short report](docs/superpowers/2026-11-16-p2-robust-flow.md) for the flow decision:
coverage/drift controls and equal tuning budgets before architecture comparisons,
not paper-score chasing. Only15 essential final checkpoints are curated in Git;
raw runs and synthetic data remain ignored.

Robust-flow review: Standards and Spec found no blocking issues; the pre-run
development3/3 prerequisite is explicit in the frozen protocol.

### Drift/coverage + equal-budget tuning — preregistered (2026-11-16)

[Protocol](docs/superpowers/2026-11-16-p2-drift-coverage-tuning.md). A: compatible,
seeds72–76, train8 vs32; both640 presented windows, batch64,300epochs=3000updates
and192000presentations. N8 repeats160 unique windows4×; N32 has640unique windows.
First8 observations/windows paired. LR=.001 constant, original losses. B: original
SPD/general, each LR{.001,.01} ×dev42–44 ×3000updates with32train; choose separately
by failures/worst H15ratio/worst response/LR, then confirm82–86. Test read after
selection is frozen; no nonlinear or H2/H3 training. Total32runs/96000updates.
Validation8/test16; seed offsets10000/20000/30000, window seed=seed, shuffle700000+seed.
Source/data hashes are recorded in `outputs/diagnosis/p2_drift_coverage_2026-11-16/`
`source_snapshot.json` and `summary.json` before/after execution. Keep raw results,
data and intermediate weights ignored; curate20 final data/confirmation controls.

```bash
export PYTHONPATH=. MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1
sha256sum research/diagnostics/p2_drift_coverage.py research/diagnostics/p2_config_check.py
.venv/bin/python -m research.diagnostics.p2_drift_coverage --stage all
.venv/bin/python -m research.diagnostics.p2_drift_coverage --stage verify-saved
```

`all` restores ignored inputs and reconstructs missing raw runs. `verify-saved`
regenerates train/val/test and checks20 curated finals, source/protocol hashes and
selection metadata; full development checkpoint reproduction needs `all`.

Drift/coverage result:32 finite runs /96,000updates /6,144,000presentations;
64validation checkpoint reloads,20curated finals verified on train/val/test,84testsPASS.
At equal budgets, N8 vsN32 compatible test joint pass1/5 vs5/5; H15 mean MSE
.000509121→.00000973021 (52.3×), drift .0171776→.000234165 (73.4×).
Per-head tuning original system selectsLR=.01 for both: general dev3/3 and new
confirmation/test5/5; SPD dev0/3 and confirmation/test0/5. Freeze N32 compatible
LR=.001 and original-general LR=.01 as positive controls for subsequent stress;
retain upstream-SPD as a labeled weaker diagnostic, not the sole comparator.
[Report](docs/superpowers/2026-11-16-p2-drift-coverage-tuning.md). No nonlinear/H2/H3
training or historical gate changes in this round. Next: controlled nonlinear
stress with these baselines, then matched-class architecture comparisons.

Drift/coverage review (Standards/Spec): no blocking findings; historical runners
and model/loss code unchanged. Curate only20 final controls plus compact metadata.

Fresh-export check PASS: `verify-saved` restored35 initially absent synthetic
datasets and verified20 curated final checkpoints without raw run records.
