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
