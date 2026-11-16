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
