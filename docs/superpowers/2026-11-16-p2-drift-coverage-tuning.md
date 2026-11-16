# P2: drift/coverage và tuning ngang ngân sách

## Protocol khóa trước chạy

Mục tiêu: định vị sai số drift/generalization và kiểm tra baseline công bằng,
không ép trùng điểm paper. Giữ nguyên mọi source và gates lịch sử.

- Reuse frozen `p2_config_check`: encoder/operator/loss gốc, latent6, dt=.02,
  Euler learner/RK4 data, AdamW WD1e-5, batch64, H15, 20 windows/traj, T50.
- **A — data diversity:** compatible G=e2, seeds MỚI72–76, LR=.001 cố định.
  So sánh8 và32 independent train trajectories. Cả hai640 presented windows,
  300 epochs=3000 updates=192000 sample presentations. Nhóm8 có160 unique
  windows lặp4 lần; nhóm32 có640 unique windows. First8 trajectories và window
  samples giống nhau trong mỗi cặp. Khác biệt gồm số data độc lập/độ đa dạng và
  mức lặp lại; không gọi đây là tác động thuần của geometric coverage.
- Validation8/test16 trajectories, RNG offsets10000/20000/30000+seed. Cùng
  initial model seed và shuffle700000+seed; sampler seed=seed.
- **B — equal tuning budget:** original G=e1,32 train trajectories. Mỗi SPD và
  general head được cùng2 LR {.001,.01} ×3 dev seeds42–44 ×3000updates. Loss
  giữ nguyên. Chọn riêng từng head bằng số dev không đạt, worst H15/ignore-input
  ratio, worst response, LR. Đạt=response<=.05 AND H15ratio<=.25.
- Khóa2 winners rồi xác nhận trên5 seeds MỚI82–86; cùng3000updates/head/run.
  B dùng32 train đã khai báo trước, không chọn lượng data bằng test A.
- Chỉ đọc test sau khi toàn bộ selection/validation decisions đã khóa. Không
  chọn lại config theo test, không dùng analytic oracle trong training loss.
- Đánh giá: input response và H15 có input; drift error trên predecessor states;
  zero-input rollout H15/H50 so exact linear flow; normalized nearest-train-state
  distance p95; tách drift/input/cross trên actual held-out actions bằng float64.
  MSE vector field không phải bound của rollout MSE; coverage correlation không
  tự chứng minh nguyên nhân. H50 là phép đo bổ sung, không dùng để chọn config.
- Tổng32 runs: A10, B-dev12, B-confirm10; tổng96000updates/6144000presentations.
  Raw checkpoints100/300, data và histories ignored. Curate final checkpoint của
  A10 và B-confirm10 để giữ cả paired data controls và các failure quan trọng.
- Nạp lại tất cả local checkpoints để đối chiếu validation, tái tạo input hashes;
  `verify-saved` chỉ kiểm tra20 curated finals và metadata selection, không tự
  chứng minh lại dev checkpoints đã bỏ khỏi Git.

Không mở nonlinear sweep hoặc train H2/H3 trong đợt này. Kết luận luồng tiếp theo
phải dựa trên các positive/negative controls trên, không đổi historical verdicts.

## Kết quả đã kiểm chứng

**32 finite runs / 96,000 updates / 6,144,000 sample presentations**. Đã nạp lại 64 checkpoint-validation evaluations và20 curated finals (train/validation/test). 84 tests PASS.

### A. Data diversity với ngân sách bằng nhau

| Train trajectories | Validation đạt | Test đạt | Test H15 MSE mean | Test drift MSE mean | Zero-input H50 MSE mean | Nearest train p95 mean |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3/5 | 1/5 | 0.000509121 | 0.0171776 | 0.00460011 | 1.327 |
| 32 | 5/5 | 5/5 | 9.73021e-06 | 0.000234165 | 8.70374e-05 | 0.4318 |

Tăng8→32 independent trajectories giảm ratio-of-means H15 error **52.3×**, drift error **73.4×** trên test, với cùng updates và presentations. Đây là tác động của intervention data-diversity/repetition đã khai báo; không suy ra geometric coverage là nguyên nhân duy nhất.

| Seed | Test H15: N8 | Test H15: N32 | Test drift: N8 | Test drift: N32 |
|---:|---:|---:|---:|---:|
| 72 | 0.000632571 | 2.07224e-06 | 0.0202213 | 0.000118132 |
| 73 | 0.000175624 | 2.99486e-05 | 0.00864488 | 0.000398513 |
| 74 | 0.000674115 | 2.59393e-06 | 0.0118578 | 0.000109981 |
| 75 | 0.000356699 | 5.32027e-06 | 0.0134163 | 0.000215862 |
| 76 | 0.000706595 | 8.71601e-06 | 0.0317478 | 0.000328339 |

Vì G và CB hằng trong positive controls, response validation/test trùng nhau; không coi đó là hai phép xác nhận input độc lập. Input response mean chỉ đổi từ.0003105 sang.0002983; cả hai nhóm đạt input-response5/5. Phần cải thiện lớn nằm ở prediction/drift. Zero-input H15/H50 so exact continuous flow nên vẫn gồm sai khác Euler–ZOH; cùng solver trong mỗi cặp giúp tránh thay solver thành confound.

### B. Tuning ngang ngân sách trên tọa độ gốc

| Head | LR | Dev đạt | Worst validation response | Worst H15/ignore-input ratio |
|---|---:|---:|---:|---:|
| SPD | 0.001 | 0/3 | 1.02609 | 1.04727 |
| SPD | 0.01 | 0/3 | 1.0117 | 1.00617 |
| general | 0.001 | 0/3 | 0.751336 | 0.855942 |
| general | 0.01 | 3/3 | 0.000459884 | 0.00465845 |

Ngân sách ngang nhau là updates, presented windows và số cấu hình; không khẳng định walltime/FLOP bằng nhau. Mỗi head có18000 tuning updates. Cả hai chọn LR=.01 theo cùng quy tắc; general đạt development3/3, SPD0/3. Việc hai winners trùng LR không thay đổi tính độc lập/ngang ngân sách của search.

| Head | New seeds82–86 validation đạt | Test đạt | Test response mean ± sample std | Test H15 MSE mean |
|---|---:|---:|---:|---:|
| SPD | 0/5 | 0/5 | 1.00811 ± 0.00319 | 0.0019133 |
| general | 5/5 | 5/5 | 0.000438467 ± 7.56e-05 | 1.95341e-06 |

General head học được hệ gốc sau tuning thích hợp; failure của LR=.001 ở đợt trước không phải bằng chứng mọi config general đều thất bại. SPD vẫn kém trong search/budget này. SPD và general khác cả expressive restriction lẫn optimization geometry; kết quả không tách riêng hai cơ chế đó và không chứng minh mọi optimizer SPD đều thất bại.

## Kết luận và luồng đã chốt

1. **Baseline positive controls đã đủ tin cậy trong protocol này:** N32 compatible với LR=.001, và original-general với LR=.01, đều đạt5/5 validation/test seeds mới. Không cần tiếp tục ép MSE về điểm paper.
2. **Không dùng upstream-SPD làm comparator duy nhất trên hệ gốc.** Giữ nó là diagnostic/faithful-code control; dùng general-head đã tune làm strong fixed-B comparator và ghi rõ deviation từ upstream. Không relabel thành exact paper reproduction.
3. **Bước tiếp theo là controlled nonlinear stress với baseline đã khóa**, cùng đủ data/coverage và equal tuning budget. Xác nhận a=0 vẫn là positive control trước khi tăng state-dependent input direction; đo fixed-map floor/excess fitting error, drift và rollout riêng.
4. Sau stress mới thiết kế matched-class one-branch/control-affine vs factorized learning comparisons. Chỉ thắng fixed-B không chứng minh novelty/ưu thế cơ chế factorization. H3 cần certificate/MPC evaluation riêng.

Đợt này kết thúc phần drift/coverage và tuning. Không chạy nonlinear sweep hoặc H2/H3 training; không sửa historical gates. Đây là sẵn sàng cho stress/architecture-experiment design, không phải H2 superiority hay H3 certificate. Kết quả thuộc synthetic positive controls32traj, không tự chuyển thành bảo đảm trên Pendulum/CartPole hoặc full paper replication.

## Tái lập / artifact

[TRACKING.md](../../TRACKING.md) chứa lệnh đầy đủ. `--stage all` tái tạo32 raw runs và kiểm tra local64 checkpoints; `--stage verify-saved` tái tạo input và kiểm tra20 final checkpoints được curate, không tự nạp lại12 development finals đã bỏ khỏi Git.
[Summary](../../outputs/diagnosis/p2_drift_coverage_2026-11-16/summary.json) ·
[Source/protocol hashes](../../outputs/diagnosis/p2_drift_coverage_2026-11-16/source_snapshot.json) ·
[Verification](../../outputs/diagnosis/p2_drift_coverage_2026-11-16/verification.json).

Giữ20 final A/B-confirmation checkpoints và concise metadata trong Git. 44 checkpoint versions khác, raw histories và generated data giữ local/ignored. Chọn config chỉ bằng validation; test được đọc sau selection, không dùng để chọn lại config.
