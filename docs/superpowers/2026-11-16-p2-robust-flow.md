# P2: baseline đa seed và luồng thí nghiệm tiếp theo

## Protocol khóa trước chạy

Mục tiêu là baseline đáng tin và đối chứng công bằng; không tối ưu để trùng điểm
paper. Diagnostic protocol này không thay kết quả/gates lịch sử.

- Tái dùng simulator, encoder/operator/loss và sampler đã kiểm chứng trong
  `p2_config_check.py`; không sửa code upstream hay các runner lịch sử.
- Hai ứng viên: LR=.001 hoặc .01, cố định 1000 epochs, AdamW WD=1e-5,
  cả hai encoder losses và Gramian penalty gốc. Latent6, batch64, window15,
  20 windows/traj, 8 train /8 validation /16 test trajectories, T50, dt=.02.
- Development seeds42/43/44 trên hệ compatible G=e2. Chạy đủ 1000 epochs,
  chọn theo số seeds không đạt, worst validation response, mean H15 ratio, LR.
  Đạt = response<=.05 AND H15 MSE<=.25×ignore-input H15 MSE.
- Khóa ứng viên rồi xác nhận seeds MỚI52–56: compatible, original SPD và
  original general head, tất cả cùng 1000 epochs và cùng observational data/actions
  trong từng cặp tọa độ. Không lựa chọn lại config bằng confirmation/test.
  Seed thay đồng thời model init, train data và sampler; chưa tách riêng các nguồn variance.
- Nếu ứng viên đạt development3/3 VÀ compatible VÀ original-general đều đạt
  validation5/5, chạy general head
  ở a=.5/1/2 trên52–56. Nếu không đạt, dừng nonlinear sweep theo protocol.
- Stress: dùng chung64 trạng thái từ original-a0 validation mỗi seed, symmetric
  actions ±1. Tách normalized response thành Var(G)+||CB-mean(G)||² bằng float64.
  Báo cáo nearest-train-state distance p95 chuẩn hóa std train (mốc coverage3),
  và rollout H15 riêng. Không dùng action oracle làm loss/gradient huấn luyện.
- Test52–56 chỉ đánh giá sau khi selection và các quyết định validation đã khóa.
  Train/validation/test dùng offset RNG10000/20000/30000+seed như runner gốc.
- Checkpoints100/500/1000, raw histories và synthetic inputs giữ local/ignored.
  Git chỉ giữ protocol, source/data hashes, summary/report và final checkpoints
  của15 confirmation controls để so sánh về sau. Recompute toàn bộ local trước
  báo cáo; verify-saved kiểm tra riêng các checkpoint được curate khi clone mới.

Kết quả được bổ sung sau chạy. PASS kiểm chứng không đồng nghĩa PASS khoa học;
không mở H3 hoặc tuyên bố H2 superiority từ một fixed-B floor.

## Kết quả thực chạy

**21 runs / 63,000 updates**, tất cả finite; tính lại 63 checkpoint-validation evaluations. Nạp lại15 final checkpoints để đối chiếu validation/test và source/data hashes. 82 tests PASS.

Hai ứng viên đều đạt development3/3. Quy tắc đã khóa chọn **LR=.001, constant,1000epochs** theo worst response. LR=.01 có H15 ratio trên development thấp hơn; không kết luận .001 tốt hơn mọi mặt và không đổi winner sau khi xem test.

| LR | Dev đạt | Worst response | Mean H15/ignore-input ratio |
|---:|---:|---:|---:|
| 0.001 | 3/3 | 0.000426717 | 0.0725343 |
| 0.01 | 3/3 | 0.000543527 | 0.00497067 |

Các seeds52–56 là xác nhận mới; mỗi seed đổi đồng thời model init, dữ liệu và sampler. Các bảng sau dùng config/budget đã khóa.

| Nhóm | Validation đạt cả hai tiêu chí | Test đạt cả hai tiêu chí | Test response mean ± sample std | Test H15 MSE mean |
|---|---:|---:|---:|---:|
| compatible | 5/5 | 4/5 | 0.000460154 ± 0.000213 | 0.000249296 |
| original-SPD | 0/5 | 0/5 | 1.01615 ± 0.00597 | 0.00404715 |
| original-general | 0/5 | 0/5 | 0.839762 ± 0.127 | 0.00310054 |

**Điểm cần đọc đúng:** compatible đạt response ở5/5 test seeds nhưng chỉ4/5 đạt thêm tiêu chí prediction. Seed56 có H15 ratio=.2840 > .25 (validation=.06681), dù response=.000545. Vì vậy chưa gọi toàn bộ baseline là ổn định trên test. Response ởa=0 giống nhau trên validation/test vì CB và G đều hằng; đó không phải hai xác nhận độc lập cho input map.

**Nonlinear sweep không chạy:** original-general validation0/5 nên prerequisite không đạt. Không lấy failure chuyển config làm bằng chứng mọi config của general head đều thất bại. Đây là bài kiểm tra transfer cùng config, chưa phải cuộc so sánh tuning công bằng giữa các tham số hóa.

## Phân tích sau test: tách drift và input

Không đổi selection/gates. Trên predecessor states và actions thật của compatible test, kiểm tra đẳng thức float64:

`MSE_vectorfield = MSE_drift + MSE_input + 2 mean(drift_error * input_error)`

| Seed | Drift MSE | Input MSE | Nearest-train distance p95 |
|---:|---:|---:|---:|
| 52 | 0.00498797 | 0.000303491 | 1.28 |
| 53 | 0.00792353 | 0.000488323 | 1.085 |
| 54 | 0.00854601 | 9.70781e-05 | 0.7978 |
| 55 | 0.00395958 | 0.000261802 | 0.7925 |
| 56 | 0.0143017 | 0.000360363 | 1.519 |

Seed56 có drift MSE=.01430, input MSE=.000360 (khoảng40×). Drift là thành phần lớn hơn trên trạng thái test, nhưng phép tách này không chứng minh nó là nguyên nhân duy nhất của rollout error. Seed56 cũng có nearest-train distance lớn nhất; đây là tương quan, chưa phải kiểm chứng nhân quả về coverage.

## Luồng tiếp theo rút ra từ bằng chứng

1. Giữ input-map positive control đã qua5 seeds làm sanity check; tách action-response khỏi state-prediction fidelity.
2. Kiểm tra drift/generalization trước: u=0 và lỗi theo vùng trạng thái; đối chứng tăng coverage với **cùng optimizer-update budget**. Đổi cả số trajectories lẫn epochs sẽ gây confounding.
3. Trước khi so kiến trúc, cho từng tham số hóa cùng **tuning budget** trên development đúng tọa độ sử dụng. Cùng LR không mặc nhiên là tuning công bằng. Trong các config đã đạt response constraint, cân nhắc criterion dự đoán/state generalization thay vì tối thiểu hóa tiếp một sai số response đã rất nhỏ; phải khóa protocol/test mới trước khi thử.
4. Stress mới phải xuất phát từ control mà baseline học được (G0=e2), thay vì mặc định transfer config sang G0=e1. Đối chứng general head vẫn cần để tách bias của tham số hóa. Chỉ mở kiến trúc khi đối chứng này đủ tin cậy cho câu hỏi cụ thể.

Đây là đề xuất bước tiếp theo, chưa thực hiện trong đợt này. Không tối ưu để trùng điểm paper; không đổi historical P1/P2 verdicts, không train H2/H3, không chuyển kết quả MPC/certificate cũ sang model mới.

## Tái lập và quản lý artifact

Lệnh đầy đủ (kể cả post-test decomposition) trong [TRACKING.md](../../TRACKING.md).
[Summary](../../outputs/diagnosis/p2_robust_flow_2026-11-16/summary.json) ·
[Source/protocol hashes](../../outputs/diagnosis/p2_robust_flow_2026-11-16/source_snapshot.json) ·
[Curated-checkpoint verification](../../outputs/diagnosis/p2_robust_flow_2026-11-16/verification.json) ·
[Posthoc diagnostic](../../outputs/diagnosis/p2_robust_flow_2026-11-16/posthoc_vector_field.json).

Git giữ15 final confirmation checkpoints cần cho các so sánh sau và concise metadata. 48 checkpoint versions còn lại, raw histories, data và dev runs giữ local/ignored. `--stage all` tái tạo bulk runs; `--stage verify-saved` chỉ xác minh15 curated finals, không tự xác minh lại checkpoint development/stress đã bỏ khỏi Git.
