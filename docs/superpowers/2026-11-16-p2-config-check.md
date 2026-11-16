# Kiểm tra P2: config, code hay giới hạn mô hình?

## Protocol khóa trước khi chạy

Mục tiêu: xác nhận baseline học được một hệ nằm trong lớp mô hình trước khi
quy lỗi cho fixed-B. Đây là diagnosis có nhãn, không phải tái lập paper.

1. **Đáp án biết trước:** oscillator F=[[0,1],[-1,-.2]], G=[0,1]^T. Cài
   A/B và encoder đuôi bằng 0 để tái tạo chính xác hệ trong cấu trúc gốc.
   So sánh forward/loss/gradient với module chính thức; sai phân hữu hạn float64.
   Kiểm tra riêng Euler và RK4 để không nhầm sai số solver với lỗi cài đặt.
2. **Overfit nhỏ:** 2 quỹ đạo Euler, 30 bước, cửa sổ 15, tối đa 20 cửa sổ;
   LR .001 cố định, tối đa 1000 epochs. Nếu chưa đạt, thử riêng LR .01.
   Đo MSE train, đáp ứng điều khiển và gradient theo từng thành phần loss.
3. **Tìm config trên validation:** 8 quỹ đạo train, 8 validation, 16 test;
   50 bước, 20 cửa sổ/quỹ đạo, batch64, latent6, dt=.02. 6 cấu hình
   LR {1e-4,1e-3,1e-2} × {cố định, scheduler theo epoch mean}, kiểm tra
   100/500/1000 epochs. Giữ loss gốc; chỉ thử bỏ cả hai encoder loss nếu
   không cấu hình loss gốc nào đạt. AdamW weight decay1e-5; seed tune42.
   Một run dừng ở checkpoint đầu đạt BOTH validation response<=.05 và
   H15 MSE<=25% exact-drift/ignore-input H15. Chọn run đạt với ít updates
   nhất, rồi response, rồi thứ tự tên; tuyệt đối không chọn bằng test.
   Nếu không run nào đạt, báo thất bại và không mở stress/H2/H3.
4. **Xác nhận và đối chiếu:** config đã chọn được khóa (LR, scheduler,
   losses, epochs); chạy 5 seeds42–46 trên compatible G=e2, rồi G=e1 với
   SPD gốc và general P0 exp(M), cùng ngân sách. General giữ nguyên initial
   A/B. Positive control G=e2 phải đạt cả hai tiêu chí trên cả 5 seeds trên
   validation trước khi mở stress. Test chỉ đánh giá sau khi lựa chọn khóa.
5. **Stress có điều kiện:** nếu general trên G=e1 cũng đạt validation trên
   5 seeds, chạy general trên a=.5/1/2, 5 seeds, config/budget đã khóa;
   so sánh response với best-constant floor. Không thay historical gates.

Dữ liệu diagnosis mới: train/validation/test seeds khác nhau, state bounds
[-1.5,1.5], u uniform[-2,2]. Với a=0, hai cách tọa độ tương đương qua
x=P y, P=[[.2,1],[-1,0]]: cùng F, Gx=e1 và Gy=e2; initial x=P y,
cùng input và cùng trajectories được biến đổi. Với a>0 dùng hệ rotation
G(x)=[cos(a*x1),sin(a*x1)]^T, initial x=P y; công bố đây là protocol mới.
Train không dùng đạo hàm/oracle supervision. Analytic F/G chỉ sinh simulator,
đánh giá và kiểm tra đáp án. Các ví dụ validation/test không đưa vào train.
Nguồn, data, sampler, shuffle, checkpoint đều có hash/seed. Các bước dùng
cached Euler phải khớp forward và gradient chính thức trước khi train.

**Trạng thái trước chạy:** PLANNED. Kết quả được bổ sung bên dưới sau kiểm tra.

## Kết quả thực chạy

**Đã thực hiện 22 run, 18,100 optimizer updates.** Tất cả run hữu hạn;
đã nạp lại và đối chiếu 32 checkpoint evaluations và 15 test evaluations.
Inventory, quy tắc chọn config và các điều kiện mở/dừng từng bước được kiểm tra độc lập.

**1. Code/đáp án biết trước:** PASS. Forward khớp tuyệt đối; gradient cached so với
module chính thức lệch tối đa 3.73e-9. Sai phân hữu hạn loss theo tham số operator
float64 PASS. Đáp án cài sẵn cho response=0, H15 Euler MSE=2.44e-15;
trên RK4, H15 MSE=2.78e-6. Sai khác solver được ghi riêng. Kiểm tra float64
đặt default dtype trong phạm vi test vì builder upstream tạo zeros không chỉ rõ dtype;
không sửa code upstream và không suy ra toàn bộ hệ thống không còn bug.

**2. Overfit nhỏ:** loss gốc, LR=.001 cố định, 1.000 epochs đạt response
0.022359, H15 ratio so với bỏ input
0.068723. Mô hình học được tác dụng đầu vào trên bài tương thích.

**3. Tìm config:** kết quả validation seed42; chọn bằng quy tắc đã khóa, không dùng test.

| LR | Scheduler | Epoch dừng | Response | H15/bỏ input | Đạt |
|---:|---|---:|---:|---:|---|
| 0.0001 | constant | 1000 | 0.753452 | 2.28143 | False |
| 0.0001 | epochmean | 1000 | 0.753452 | 2.28143 | False |
| 0.001 | constant | 1000 | 0.000367538 | 0.133638 | True |
| 0.001 | epochmean | 1000 | 0.0125375 | 0.0653785 | True |
| 0.01 | constant | 100 | 0.000566277 | 0.0116091 | True |
| 0.01 | epochmean | 100 | 0.000566277 | 0.0116091 | True |

Cùng dữ liệu, khởi tạo, loss và **100 epochs** trên seed42: LR=.001 cho response 0.771278, trong khi LR=.01 cho 0.000566. Đây là đối chứng trực tiếp cho ảnh hưởng của LR.


Config được chọn: **LR=0.01, constant, 100 epochs, loss gốc**. Không cần mở nhánh bỏ encoder loss.
Quy tắc ưu tiên ít updates nhất chọn 100 epochs; việc này không chứng nhận ngân sách đó phù hợp mọi seed.

**4. Xác nhận trên 5 seeds:**

| Seed | Compatible validation response | Compatible test response | Validation đạt cả hai tiêu chí |
|---:|---:|---:|---|
| 42 | 0.000566277 | 0.000566277 | True |
| 43 | 0.000193423 | 0.000193423 | True |
| 44 | 0.000979011 | 0.000979011 | True |
| 45 | 0.60169 | 0.60169 | False |
| 46 | 0.0175705 | 0.0175705 | True |

**Đạt 4/5 seeds, chưa đạt điều kiện 5/5.** Seed42 đã dùng để tune; bốn seeds còn lại là xác nhận mới, không gọi cả năm là độc lập với tuning.
Bảng test được tạo sau khi config và quyết định các stage đã khóa; không dùng để chọn lại cấu hình.

| G=e1, config/ngân sách đã khóa | Validation response, mean ± sample std | Seeds đạt |
|---|---:|---:|
| SPD | 1.01086 ± 0.0127 | 0/5 |
| general | 0.863505 ± 0.0891 | 0/5 |

**5. Stress phi tuyến:** **không chạy**, vì prerequisite 5/5 chưa đạt. Đây là dừng có điều kiện đã khai báo trước, không phải bỏ sót kết quả.

## Điều có thể và chưa thể kết luận

- **Config/ngân sách có ảnh hưởng thực:** cùng baseline/loss gốc học được positive control;
  LR=.0001 chưa đạt sau 1000 epochs trên seed tune, trong khi .01 đạt sau100.
- **Chưa chứng nhận config ổn định:** lựa chọn từ một seed chưa vượt gate 5/5.
  Không kết luận rằng mọi config khác đều thất bại; các ứng viên .001/1000 chưa được xác nhận đa seed.
- **Chưa cô lập toàn bộ nguyên nhân trên hệ gốc:** cả SPD và general đều chưa đạt
  ở ngân sách được chọn. Vì vậy chưa thể quy toàn bộ sai số cho SPD hay fixed-B.
- H2/H3 tiếp tục NO-GO; không sửa historical gates và không tuyên bố tái lập paper.
- Bước tiếp theo có cơ sở: protocol mới xác nhận ngân sách dài hơn và chọn config
  bằng nhiều validation seeds; khóa test mới nếu tiếp tục chọn cấu hình. Không tự mở rộng
  tìm kiếm sau khi thấy test của đợt này.

## Chạy lại và bằng chứng

```bash
MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python -m research.diagnostics.p2_config_check --stage all
MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python -m research.diagnostics.p2_config_check --stage recompute
.venv/bin/python -m research.diagnostics.report_p2_config_check
```

Lệnh all tái sử dụng run đã hoàn tất; recompute nạp lại checkpoint. Muốn train lại từ đầu,
dùng bản sao script/output riêng để giữ bằng chứng gốc. Protocol trước khi chạy, source/data/
checkpoint hashes, history LR/loss và gradient từng thành phần được lưu tại
[thư mục bằng chứng](../../outputs/diagnosis/p2_config_check_2026-11-16/).

[Kiểm chứng độc lập](../../outputs/diagnosis/p2_config_check_2026-11-16/verification.json) ·
[Config đã chọn](../../outputs/diagnosis/p2_config_check_2026-11-16/selected_config.json) ·
[Gates](../../outputs/diagnosis/p2_config_check_2026-11-16/stage_gates.json).

Bộ kiểm thử repository: **80 tests PASS**.
