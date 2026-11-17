# Baseline đã chốt

**Quyết định:** dùng **general-head fixed-B Koopman** làm baseline mạnh cho
controlled nonlinear stress tiếp theo. Giữ upstream-SPD như đối chứng code gốc;
không dùng nó làm comparator duy nhất khi chưa qua positive control.

**Bằng chứng bổ sung (P1 grid 2026-11-16):** head SPD đã phát hành không đạt Table 1
CartPole ở mọi fraction (12–188×) vì ép dấu cột input; head sign-free (dense-P `91a47d0`)
đạt ≤2× ở 7/8 ô. Điều này ủng hộ việc dùng head không ép dấu làm comparator.
[Grid report](../outputs/reproduction/p1_upstream_faithful_2026-11-16/REPORT_grid.md).

**Comparator thứ hai cho H2 (2026-11-17):** input encoder G.5 của tác giả (`use_modulated_u`,
hệ số vô hướng `g(x)⊙u`) cùng config/ngân sách, chọn LR bằng validation. Mọi kết luận H2 phải
vượt cả fixed-B lẫn G.5. Config ở đây chỉ đã hiệu chỉnh cho rotation; unicycle/quadrotor cần
hiệu chỉnh riêng trước khi so sánh. [H2 diagnosis](../outputs/diagnosis/h2_diagnosis_2026-11-17/REPORT.md),
[G.5 tests](../outputs/diagnosis/h2_g5_tests_2026-11-17/REPORT.md).

## Cấu hình đã xác nhận

| Thành phần | Main comparator: original coordinates | Compatible sanity control |
|---|---|---|
| Module | `KoopmanLitModule` | `KoopmanLitModule` |
| Physical head | `P0 @ exp(M)`, locally general/invertible | SPD upstream |
| LR | `0.01`, cố định | `0.001`, cố định |
| Optimizer | AdamW, weight decay `1e-5` | Như main |
| Budget | **3.000 updates / 192.000 window presentations** | Như main |
| Data | 32 train / 8 validation / 16 test trajectories, mỗi trajectory50 steps | Như main |
| Sampler | 20 fixed windows/traj, window15, batch64 | Như main |
| Epochs trong protocol này | 300 (640 windows, 10 updates/epoch) | 300 |
| Architecture | `controllable`; state2, input1, latent6; 3 hidden layers ×64 ReLU | Như main |
| Dynamics / solver | `dz/dt=Az+Bu`, `use_modulated_u=False`; Euler, dt=.02 | Như main |
| Loss | Prediction gamma=.8; encoder-true + .5 encoder-pred; .005 softplus(.03−λmin(W)), coupled Euler W over15dt | Như main |

Không dùng oracle/analytic derivatives làm training loss. Source/protocol hashes nằm trong [source snapshot](../outputs/diagnosis/p2_drift_coverage_2026-11-16/source_snapshot.json);
config và per-run data identities nằm trong [summary](../outputs/diagnosis/p2_drift_coverage_2026-11-16/summary.json).

**Scope:** cấu hình trên đã xác nhận cho synthetic positive controls của protocol
này; không mặc định chuyển latent6/LR/data32 sang Pendulum, CartPole hay mọi hệ khác.
Với task mới, giữ baseline identity và cấp tuning budget ngang nhau, chọn config
bằng development/validation, khóa trước khi đọc test.

## Checkpoint và cách dùng

- Main: `outputs/diagnosis/p2_drift_coverage_2026-11-16/runs/confirm-general-s{82..86}/epoch300.pt`.
- Compatible: `outputs/diagnosis/p2_drift_coverage_2026-11-16/runs/coverage-n32-s{72..76}/epoch300.pt`.
- Giữ cả năm seeds; không chọn lại seed/checkpoint theo điểm test.
- General checkpoint phải dùng `p2_config_check.make_model(seed, general=True)`
  rồi strict-load; inference bằng `p2_config_check.matrices/rollout/metric`.
  Forward của module upstream chưa được thay head, nên không dùng trực tiếp
  `model(x0,u)` hoặc `model.model.operator.get_AB()` cho general checkpoint.
- Replay/verify: các lệnh drift/coverage trong [TRACKING.md](../TRACKING.md).

## Bằng chứng và điều kiện đi tiếp

32 runs, equal-update/equal-presentation controls, tuning riêng ngang ngân sách.
Main general và compatible N32 đều đạt **5/5 validation và test seeds mới**;
upstream-SPD trên original coordinates **0/5** trong search/budget đã thử.
[Báo cáo số liệu và giới hạn](superpowers/2026-11-16-p2-drift-coverage-tuning.md).

**Readiness: READY_FOR_CONTROLLED_NONLINEAR_STRESS.** Đóng đợt calibration này;
chưa train kiến trúc H2/H3 và không sửa historical gates. Trên stress phải đo
response floor/excess fitting error, drift và rollout riêng. Sau đó mới so sánh
learning mechanisms trong matched model class.

General head là deviation có nhãn từ public SPD implementation; không gọi nó là
exact paper reproduction. Mục tiêu là baseline đáng tin và phép so sánh công bằng,
không ép điểm bằng paper hay tuyên bố general head luôn tốt hơn mọi config SPD.
