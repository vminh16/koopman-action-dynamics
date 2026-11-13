# Experiment Tracking

Tài liệu này ghi lại toàn bộ run phục vụ tái lập Pendulum. Không ghi kết quả
paper vào cột kết quả của repo.

## 1. Reference values from paper

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
| B01-s42 | 5% | 42 | 100 | COMPLETED | [config](configs/pendulum_mode-controllable_win15_frac0.05_dz16_hs64_nl3_ul2_uh64_MLPmodel.yaml) | [checkpoint](checkpoints/pendulum_mode-controllable_win15_frac0.05_dz16_hs64_nl3_ul2_uh64_MLPmodel.pt) | 0.106817 | Open-loop, 1,000 test trajectories |
| B01-s43 | 5% | 43 | 100 | PLANNED |  |  |  |  |
| B01-s44 | 5% | 44 | 100 | PLANNED |  |  |  |  |
| B02-s42 | 10% | 42 | 100 | COMPLETED | [config](configs/pendulum_mode-controllable_win15_frac0.1_dz16_hs64_nl3_ul2_uh64_MLPmodel.yaml) | [checkpoint](checkpoints/pendulum_mode-controllable_win15_frac0.1_dz16_hs64_nl3_ul2_uh64_MLPmodel.pt) | 0.099285 | Open-loop, 1,000 test trajectories |
| B02-s43 | 10% | 43 | 100 | PLANNED |  |  |  |  |
| B02-s44 | 10% | 44 | 100 | PLANNED |  |  |  |  |
| B03-s42 | 30% | 42 | 100 | COMPLETED | [config](configs/pendulum_mode-controllable_win15_frac0.3_dz16_hs64_nl3_ul2_uh64_MLPmodel.yaml) | [checkpoint](checkpoints/pendulum_mode-controllable_win15_frac0.3_dz16_hs64_nl3_ul2_uh64_MLPmodel.pt) | 0.006827 | Open-loop, 1,000 test trajectories |
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
TensorBoard event for each fraction. Only seed 42 is currently available.

### Current assessment

- The expected monotonic trend is present: `MSE(5%) > MSE(10%) > MSE(30%)`.
- The 30% result is close to the paper reference.
- The 5% and 10% results are substantially higher than the paper references and
  require investigation before claiming reproduction.
- P0 is complete with verdict `MISMATCH`; the local runs are valid
  protocol-variant evidence, not exact paper reproduction.
- Aggregate reproduction remains open because only seed 42 has been run and the
  low-data results do not match the reference.
- The saved checkpoints/configs do not include an explicit seed field; the
  training script currently uses its fixed seed value `42`.

P0 audit manifest: [E00 manifest](outputs/reproduction/E00/manifest.yaml).

## 3.1. Phase P2 Controlled Stress Tests Summary

Phase P2 đã hoàn tất thực nghiệm chẩn đoán và phân tách lỗi (Gate P2 PASS):
- **Rotation family sweep ($a \in [0.0, 2.0]$):** $E_{\text{response}}$ tăng từ $0.2397$ lên $1.1561$ ($4.82\times$), $\|CB - G_x\|_F$ tăng từ $0.4896$ lên $1.0207$, trong khi 1-step prediction MSE giữ ở mức rất thấp $\approx 10^{-5}$.
- **Unicycle:** $E_{\text{lift}} = 1.5617, E_{\text{phys}} = 1.5096, d_{\text{orient}} = 0.5347, \rho_T = 2.80 \times 10^{-3}$.
- **Planar Quadrotor:** $E_{\text{lift}} = 53.3568, E_{\text{phys}} = 35.3904, d_{\text{orient}} = 1.0677, \rho_T = 4.64 \times 10^{-4}$.
- **Convex QP MPC:** Đã kiểm thử trên hệ LTI và mô hình Koopman; tracking cost tăng $27.5\%$ trên hệ quay ($a=1.0$ vs $a=0.0$).
- **Phán quyết Gate H2 / H3:** Đã chứng minh giới hạn cấu trúc model-class của toán tử $B$ tĩnh khi hướng tác động xoay theo trạng thái; đủ bằng chứng mở Gate H2 và Gate H3.
- Báo cáo chẩn đoán chi tiết: [p2_failure_report.yaml](outputs/diagnosis/p2_failure_report.yaml).

## 4. Commands

Run from the repository root with the project environment activated:

```bash
source .venv/bin/activate
mkdir -p checkpoints configs figures
```

Train one run:

```bash
PYTHONPATH=. python experiments/run_train.py \
    --system pendulum \
    --koopman_mode controllable \
    --control_loss_flag \
    --latent_dim 16 \
    --num_layers 3 \
    --hidden_size 64 \
    --window_len 15 \
    --train_fraction FRACTION \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    2>&1 | tee logs/RUN_ID.train.log
```

Evaluate the resulting checkpoint:

```bash
PYTHONPATH=. python experiments/run_eval.py \
    --config configs/CONFIG.yaml \
    --window_len 15 \
    --max_traj 1000 \
    --num_plot 5 \
    2>&1 | tee logs/RUN_ID.eval.log
```

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
