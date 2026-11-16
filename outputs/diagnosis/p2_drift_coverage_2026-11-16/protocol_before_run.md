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
