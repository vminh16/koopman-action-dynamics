# RESEARCH REQUIREMENTS DOCUMENT (RRD)
## Controllability-Preserving State-Dependent Input Maps for Koopman Latent Dynamics

**Phiên bản:** 0.3 — 12/11/2026 (thay thế v0.2 "Dynamics-Factorized Learning for Controllability-Aware Latent World Models")  
**Trạng thái:** Research proposal; chưa xác nhận novelty; chưa triển khai phương pháp. Mệnh đề ở §5.3 mới có phác thảo chứng minh và kiểm tra số trên hệ đồ chơi.  
**Quyết định nghiên cứu:** Hướng trọng tâm là **H3 — giữ bảo đảm controllability khi input map phụ thuộc trạng thái**. **H2** (phân tách Autonomous / Action-Response) không còn là đóng góp dự kiến; nó là *lớp mô hình* mà H3 phải làm việc trên đó. H1 (Prediction / Control-specific branches) vẫn không triển khai.  
**Baseline duy nhất cần tái lập:** Miao et al., *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026 [P1].  
**Tài liệu đi kèm:** `EXPERIMENT.md` (giao thức đo đạc) và `ROADMAP.md` (thứ tự nghiên cứu, tiêu chí chuyển pha). Cả hai cần cập nhật theo §9–§10 của bản này.

> **Lập trường nghiên cứu:** Chúng ta không bắt đầu bằng một network hai nhánh rồi tìm lý do hợp thức hóa. Kiến trúc có input map phụ thuộc trạng thái **đã tồn tại** trong literature (§7), nên "S2 dự đoán tốt hơn S0" không phải là đóng góp. Câu hỏi của tài liệu này là: *khi cho phép hành động tác động khác nhau ở các trạng thái khác nhau, bảo đảm controllability của baseline còn giữ được đến đâu, và phải trả bao nhiêu độ trung thực để giữ nó?*

**Thay đổi so với v0.2 (tóm tắt):** tiêu đề và quyết định nghiên cứu; L1/L2 ở §2.3; §2.4; Tầng B ở §3; ghi chú ở §4.4; **§5.3 viết lại hoàn toàn và là phần lõi**; §5.4; bảng RQ và giả thuyết ở §6; bảng novelty ở §7; bảng lựa chọn ở §8; benchmark ở §9; gate ở §10; references. Số mục giữ nguyên để `EXPERIMENT.md` và `ROADMAP.md` không gãy tham chiếu.

---

## 1. Bài toán lớn và trực giác

### 1.1. Một world model phục vụ điều khiển cần biết gì?

Trong system identification / model-based control, hệ vật lý chịu tác động của cả trạng thái hiện tại lẫn lệnh điều khiển. Một mô hình được sử dụng để (i) dự đoán quỹ đạo từ chuỗi lệnh đã biết và (ii) giúp planner chọn chuỗi lệnh có thể đưa hệ đến mục tiêu.

Ba năng lực cần phân biệt:

1. **Dynamics fidelity**: biết thế giới chuyển động như thế nào với trạng thái và lệnh cho trước.
2. **Action-response fidelity**: biết **nếu thay lệnh mà giữ cùng trạng thái**, chuyển động sẽ khác thế nào.
3. **Control utility**: dùng mô hình để xác định trạng thái có thể đạt và chọn lệnh với chi phí hợp lý.

Prediction và controllability **không độc lập về mặt động lực học**: controllability là thuộc tính của hệ động lực học và lớp tín hiệu điều khiển. Nhưng **mục tiêu học** có thể khác: MSE trên các quỹ đạo thường gặp không nhất thiết làm chính xác *những khác biệt giữa các hành động*. Do vậy, một biểu diễn chung có thể phục vụ cả hai, nhưng cũng có nguy cơ ưu tiên loại thông tin dễ dự đoán hơn thông tin quan trọng cho control.

### 1.2. Ví dụ vật lý trực quan

Một xe chạy trên dốc chịu quán tính, ma sát và lực động cơ. Ở cùng vị trí/vận tốc, nếu thay lực ga, quỹ đạo tương lai thay đổi. Động cơ còn có thể suy giảm hiệu suất theo trạng thái hoặc theo thời gian. Mạng chỉ thấy chuỗi `(state, action, next-state)`; không được cung cấp nhãn "phần do ma sát" hay "phần do ga".

**Trực giác H2 (giữ làm lớp mô hình):** Cho mô hình học riêng (a) điều gì xảy ra **khi không có lệnh điều khiển** và (b) **mức thay đổi do lệnh điều khiển**, rồi hợp thành **một** hệ dynamics nhất quán cho dự đoán và phân tích controllability. Phân tách dựa trên *cơ chế động lực học*, không dựa trên hai nhiệm vụ đầu ra.

### 1.3. Trực giác sâu hơn: predictive vs control-relevant representation

Một representation có thể dự đoán rất tốt trên chính sách thu thập dữ liệu nhưng phản ứng sai với hành động chưa thử. Planner luôn hỏi *what if action khác đi?* nên lỗi ở đây có thể quan trọng hơn lỗi dự đoán trung bình. Cần **đo ngay từ baseline**: (i) trajectory MSE, (ii) counterfactual action-effect error, (iii) khác biệt thứ hạng hành động/chi phí, (iv) closed-loop MPC. Đây là **giả thuyết cần kiểm định**, không mặc định rằng tồn tại xung đột representation.

### 1.4. Vì sao controllability là trọng tâm (mới ở v0.3)

Baseline có một tài sản mà nhóm mô hình bilinear / input-affine không có: surrogate **controllable theo cấu trúc**, kèm Gramian để định lượng mức độ. Tài sản đó tồn tại nhờ `B` là hằng số. Mọi cách làm `B` phụ thuộc trạng thái để tăng độ trung thực đều biến surrogate thành hệ phi tuyến, và định lý của baseline không còn áp dụng nguyên trạng. Như vậy có một sự đánh đổi cụ thể, đo được:

- Giữ `B` cố định: có bảo đảm, sai về action response.
- Cho `B(z)` tự do: đúng hơn về action response, mất bảo đảm.

Trực giác trung tâm của v0.3: **Gramian của phần tuyến tính là một "ngân sách"**. Phần hiệu chỉnh phụ thuộc trạng thái được phép lớn đến đâu mà bảo đảm vẫn còn là do ngân sách đó quyết định. Gramian regularization của baseline vì thế có thêm một vai trò mới: mua ngân sách cho độ biểu cảm.

---

## 2. Baseline ICLR 2026: phương pháp gốc và ranh giới

### 2.1. Nhiệm vụ, dữ liệu, mô hình

Paper học động lực học liên tục chưa biết của hệ được quan sát bằng trạng thái vật lý đầy đủ, không học policy mạng neural, reward model hoặc visual encoder [P1]. Đầu vào là các trajectory trạng thái–hành động lấy từ simulator. Mô hình:

\[
 z=\phi_\theta(x)=\begin{bmatrix}x\\\psi_\theta(x)\end{bmatrix}\in\mathbb R^N,
 \qquad \dot z=A_\theta z+B_\theta u,
 \qquad \hat x=Cz,\; C=[I_n\;0].
\]

`z` là **lifted space** (không phải không gian nén thấp chiều); `ψ` là neural observable encoder; `A,B` là toán tử học qua tham số hóa canonical và phép biến đổi tọa độ khả nghịch; ODE solver truyền trạng thái, MPC tối ưu hành động. Paper chứng minh tính chất output/state-to-output controllability của **surrogate** theo các giả định đã nêu và điều chỉnh spectrum Gramian để hỗ trợ control [P1, §3–4].

### 2.2. Mạng học phần tử nào?

| Thành phần | Học trong training? | Chức năng |
|---|---|---|
| `ψθ(x)` | Có | Observable features bổ sung |
| Tham số canonical của `Aθ, Bθ` và phép biến đổi `Pθ` | Có | Dynamics với ràng buộc controllability |
| `C=[I,0]` | Không | Trích xuất physical state |
| ODE integration | Không (thuật toán số) | Tạo trajectory dự đoán |
| MPC | Không (bài toán tối ưu) | Chọn chuỗi action |

Mô hình tối ưu prediction loss và Gramian regularization trên cùng một động lực học; **không** chia prediction/controller thành hai neural world models.

### 2.3. Giới hạn có căn cứ trực tiếp trong baseline

**L1 — Fixed input map (đã là kết quả đã biết, không phải giả thuyết).** Vì `x=Cz` và `\dot z=Az+Bu`, ta có `\dot{\hat x}=CAz+CBu`, nên

\[
\boxed{\frac{\partial\dot{\hat x}}{\partial u}=CB,\quad\text{không phụ thuộc }z.}
\]

Phát biểu đầy đủ hơn phải tính cả các hàng lift. Với hệ thật control-affine `\dot x=f_0(x)+G_x(x)u`, đạo hàm của observable dọc quỹ đạo là `\tfrac{d}{dt}\psi(x)=J_\psi(x)[f_0(x)+G_x(x)u]`, nên input map **đúng** trong toạ độ lift là

\[
\boxed{B^\star(x)=J_\phi(x)\,G_x(x)=\begin{bmatrix}G_x(x)\\ J_\psi(x)G_x(x)\end{bmatrix}.}
\]

`B^\star` phụ thuộc trạng thái **ngay cả khi `G_x` là hằng số**, trừ khi `ψ` affine theo các hướng được tác động. Hệ quả: (i) mismatch của fixed-`B` không chỉ xuất hiện ở hệ có actuator phụ thuộc trạng thái; (ii) phép đo chỉ dựa trên `CB` bỏ sót sai lệch ở các hàng lift, tức sai lệch nhiều bước. Đây là kết quả đã có: dạng lift đúng có ma trận chuyển trạng thái hằng và ma trận input phụ thuộc trạng thái [P7]; mọi hệ control-affine có realization bilinear nhưng không nhất thiết có realization linear [P8].

Với open-loop rollout **không re-encode**, nghiệm của surrogate là

\[
\hat x(T)=Ce^{AT}\phi(x_0)+\int_0^T Ce^{A(T-\tau)}Bu(\tau)\,d\tau.
\]

Trong biểu thức này, **thành phần đáp ứng của chuỗi action không phụ thuộc `x₀`**, còn quỹ đạo tổng thể vẫn phụ thuộc `x₀` qua `φ(x₀)`. Nếu ứng dụng **re-encoding định kỳ** (repo chính thức có tuỳ chọn này [P6]) thì phải phân tích riêng. Đây là phát biểu về *lớp mô hình*, không khẳng định MSE của baseline luôn tệ hoặc MPC luôn thất bại.

**L2 — Expressiveness–tractability trade-off đã được tác giả nhận diện.** Appendix G.5 của [P1] thử `\hat u=g_\theta(\hat x)\odot u`. Trên Pendulum 5% data, prediction MSE giảm từ `0.0318` xuống `0.0067` (số do v0.2 ghi lại; **chưa đối chiếu được với PDF**), nhưng MPC không còn có cấu trúc tuyến tính chuẩn theo physical input `u`. **Đã có input encoder**; đề xuất đơn giản thêm `g(x)` không mới. Hai nhận xét mới: (i) Pendulum có `G_x` hằng, nên cải thiện này nếu đúng không đến từ "actuator phụ thuộc trạng thái" mà từ hàng lift hoặc từ capacity; (ii) biến thể này viết được thành `B\,\mathrm{diag}(g(x))\,u`, thuộc Lớp A ở §5.3.3, tức **vẫn giữ nguyên bảo đảm controllability nếu `g` không triệt tiêu**. Paper gốc có nêu điều này hay không cần kiểm tra khi đọc Appendix G.5.

**L3 — Không có giao thức online dynamics-shift adaptation trong paper gốc.** Toán tử được huấn luyện trên điều kiện danh định và dùng để dự đoán/MPC; paper không trình bày thuật toán học lại action-effect sau actuator drift ở deployment. Tuy vậy online Koopman identification đã tồn tại [P4, P15]: đây là **giới hạn của baseline**, chưa là **novelty gap toàn lĩnh vực**. Ở v0.3 đây là hướng phụ (§4.6).

**L4 — Ràng buộc prior có thể căng với fitting.** Appendix G.4 cho thấy Gramian penalty quá mạnh làm prediction/control kém, chứng tỏ trade-off về optimization/expressiveness. Trong khung v0.3, trade-off này trở thành một trục đo: Gramian lớn hơn nghĩa là ngân sách hiệu chỉnh lớn hơn (§5.3.4), nhưng phần tuyến tính có thể khớp dữ liệu kém hơn.

**L5 — Output controllability của surrogate không bảo đảm control thực.** Bảo đảm theo rank/Gramian của mô hình học không tự động chuyển sang hệ phi tuyến thực dưới sai số mô hình, input bounds, dynamics shift hoặc actuator failure. Mọi kết quả ở §5.3 cũng chịu giới hạn này.

### 2.4. Giả thuyết gap (không biến giả thuyết thành kết luận)

> Các mô hình Koopman có input map phụ thuộc trạng thái (bilinear, input-affine/LPV) đã biểu cảm hơn lifted LTI, nhưng theo rà soát hiện tại chưa có tham số hoá nào **bảo đảm controllability theo cấu trúc** như [P1] làm cho lớp LTI. Giả thuyết: tồn tại các lớp `B(z)` có cấu trúc mà (i) bảo đảm của baseline được giữ chính xác hoặc giữ với một biên định lượng tính từ Gramian, và (ii) vẫn bắt được phần lớn sai lệch action-response của fixed-`B` trên các hệ mục tiêu.

**Điều sẽ bác bỏ gap:** đã có công trình làm đúng việc này (rà soát §7 chưa đầy đủ); hoặc biên định lượng quá bảo thủ đến mức vô nghĩa trên mọi benchmark (tỉ số `χ` ở §5.3.6 lớn hơn 1 rất nhiều); hoặc lớp giữ được bảo đảm không tốt hơn S0/S1 về action response; hoặc fixed-`B` baseline đã đủ tốt trong closed loop có re-encode.

---

## 3. Formal problem statement

Xét hệ continuous-time:

\[
\dot x(t)=f_{\eta(t)}(x(t),u(t)),\quad x\in\mathcal X\subseteq\mathbb R^n,
\quad u\in\mathcal U\subseteq\mathbb R^m.
\]

`η` là chế độ vật lý chưa biết; ở giai đoạn chính của v0.3, `η` **cố định** (hệ danh định). Dữ liệu `D_train={(x(t_i),u(t_i),x(t_{i+1}),Δt_i)}`. Test time chỉ được sử dụng transition đã quan sát và lệnh đã thực hiện; **không biết** `η` và **không truy cập** outcome của hành động chưa làm.

Mục tiêu nghiên cứu có hai tầng, thứ tự bắt buộc:

- **Tầng A — Baseline diagnosis (trước phương pháp mới):** baseline sai chỗ nào về action response, và **phần hiệu chỉnh cần thiết lớn bao nhiêu so với ngân sách Gramian** (§5.3.6)?
- **Tầng B — H3 (chỉ khi Tầng A ủng hộ):** thiết kế lớp `B(z)` có cấu trúc, chứng minh bảo đảm controllability tương ứng, và đặc trưng hoá đường đánh đổi giữa bảo đảm và độ trung thực.

**Không nằm trong scope đầu:** raw pixels, partial observability, robust safe control guarantees, mọi loại nonstationarity, world-model video generation, neural policy, **bảo đảm cho hệ thật** (chỉ phát biểu cho surrogate, xem L5). Dynamics shift và online adaptation lùi xuống thành mở rộng sau khi H3 có kết quả.

---

## 4. Từ trực giác đến mô hình hóa toán học H2

### 4.1. Phân rã định nghĩa (luôn đúng khi `u=0` hợp lệ)

Với bất kỳ trường vector điều khiển nào `f(x,u)`:

\[
 f_0(x):=f(x,0),\qquad \Delta_f(x,u):=f(x,u)-f(x,0),
\]
\[
\boxed{f(x,u)=f_0(x)+\Delta_f(x,u),\qquad \Delta_f(x,0)=0.}
\]

Đây chỉ là **phân rã định nghĩa**, chưa tạo ra tính độc lập nhân quả hay điều kiện nhận dạng. `f_0` là *zero-input drift* (không đồng nghĩa với "mọi biến không thể điều khiển"); `Δf` là **increment do action tại cùng trạng thái**. Nếu `u=0` không thuộc tập lệnh hợp lệ, cần chọn **reference action** `u_ref` và ghi rõ phép phân rã phụ thuộc reference.

### 4.2. Giả định hẹp giúp toán học đơn giản

Với hệ **control-affine**:

\[
\boxed{\dot x=f_0(x)+G(x)u,\quad G(x)\in\mathbb R^{n\times m}.}
\]

`G(x)` là **ma trận độ nhạy tức thời với input**, khác với tính controllability của toàn hệ. Với nonlinear-in-action dynamics, mô hình này có model mismatch; khi đó cần nhánh `Δ_f(x,u)` tổng quát hơn. Không mặc định toàn bộ robot/contact dynamics đều control-affine trơn toàn miền.

### 4.3. Giả thuyết hai nhánh trong *một* lifted space

Giữ identity lifting của baseline để dự đoán physical state và giữ so sánh công bằng:

\[
 z=\phi_\omega(x)=\begin{bmatrix}x\\\psi_\omega(x)\end{bmatrix},\quad
 \dot z=F_\alpha(z)+G_\beta(z)u,\quad \hat x=Cz.
\]

- `Fα`: **Autonomous Dynamics Branch**, dự đoán zero-input drift trong lifted coordinates.
- `Gβ`: **Action-Response Branch**, mô tả độ nhạy với physical action.
- `φω`: **Shared Observable Encoder**; *không có hai latent spaces rời* trong mô hình chính.
- ODE Solver tính **một** trajectory dự đoán; controllability/reachability được **phân tích trên dynamics hợp nhất**.

Sơ đồ ý tưởng:

```text
Full-state x ── Shared identity-lift encoder ── z ──┬─ Autonomous F(z) ──┐
                                                    │                  ├─ Joint dz/dt ─ ODE ─ x̂(t)
Action u ────────────────────────────────────────────└─ G(z)u ──────────┘        │
                                                                     └─ Control analysis / MPC
```

**Không mặc định attention hoặc concat**: phép hợp nhất xuất phát từ định nghĩa dynamics. Trong v0.3, `F` **giữ tuyến tính** (`F(z)=A_cz`), vì toàn bộ lập luận ở §5.3 dựa vào phần tuyến tính `(A_c,B_c)`. Với `F` phi tuyến tổng quát, mô hình trở thành một control-affine neural ODE và không còn lý do để lift.

### 4.4. Bảo toàn tính liên tục với baseline: một họ mô hình lồng nhau

\[
\begin{aligned}
\text{S0 (paper):}\quad & \dot z=A_cz+B_cu.\\
\text{S1 (appendix):}\quad & \dot z=A_cz+B_c\big(g_\beta(Cz)\odot u\big).\\
\text{S2 (input-affine):}\quad & \dot z=A_cz+\big[B_c+\Delta B_\beta(z)\big]u.\\
\text{S3 (ứng viên H3):}\quad & \dot z=A_cz+\big[B_c+\Delta B_\beta(z)\big]M_\gamma(z)\,u,\quad \|\Delta B_\beta(z)\|\le\delta,\; M_\gamma(z)\ \text{khả nghịch}.
\end{aligned}
\]

`A_c,B_c` **ký hiệu toán tử đã qua canonical transformation**. Ghi chú:

- **S1 ⊂ S2**: `B_c(g\odot u)=B_c\,\mathrm{diag}(g)\,u`, tức `ΔB=B_c(\mathrm{diag}(g)-I)`. S1 chỉ đổi *độ lớn* theo từng kênh input; S2 đổi được cả *hướng* input đi vào không gian lift. Mọi ablation S1 vs S2 phải nhắm đúng khác biệt này.
- **S2 không mới**: đây là dạng Koopman input-affine/LPV, đã có cả lý thuyết [P7] lẫn cách học bằng deep encoder [P9, P10]. Dạng bilinear `\dot z=Az+B_0u+\sum_j u_jB_jz` là trường hợp `ΔB` tuyến tính theo `z` [P8, P11].
- **S3 là S2 có ràng buộc**, được chọn sao cho áp dụng được §5.3. Đây là ứng viên phương trình đầu tiên của v0.3.
- **Biến thể theo Jacobian**: `B(z)=J_\phi(Cz)\,\hat G(Cz)` chỉ học `\hat G` cỡ `n×m` và thoả điều kiện manifold ở §4.5 theo cấu trúc. Nó tối ưu về độ trung thực nhưng tự nó không có bảo đảm; dùng làm **mốc trên về fidelity** để đo S3 mất bao nhiêu.

Novelty (nếu có) phải đến từ bảo đảm và cơ chế học đi kèm, không phải từ bất kỳ dạng phương trình nào ở trên.

### 4.5. Tính nhất quán với latent manifold

Vì `z=φ(x)` với `x=Cz`, dynamics lý tưởng phải gần như tiếp tuyến với manifold `\mathcal M=\{\phi(x)\}`:

\[
\boxed{F(\phi(x))+G(\phi(x))u\ \approx\ J_\phi(x)[f_0(x)+G_x(x)u].}
\]

Đây là **điều kiện consistency**, không là equality có sẵn. Một mô hình có state reconstruction tốt vẫn có thể rollout latent lệch manifold. Có thể đo `e_\mathcal M(z)=\|z-\phi(Cz)\|` ở các time points như diagnostic. Lưu ý cho §5.3: các bảo đảm controllability phát biểu trên toàn `\mathbb R^N`, kể cả vùng ngoài manifold, nơi `z` không ứng với trạng thái vật lý nào; ý nghĩa vật lý chỉ có qua `\hat x=Cz`.

### 4.6. Online adaptation là mở rộng sau, không thuộc lõi v0.3

Nếu vật lý thay đổi:

\[
\dot x=f_{0,\eta_t}(x)+G_{\eta_t}(x)u,
\]

thì cả `f0` lẫn `G` **có thể** đổi. Hai lý do lùi phần này: (i) khi cố định encoder, mô hình tuyến tính theo `(A,B)` nên RLS đã là baseline mạnh, và meta-adaptation kèm bảo đảm ổn định đã có [P15]; (ii) nếu H3 thành công, nó cho một câu hỏi adaptation sắc hơn: *cập nhật online có giữ mô hình nằm trong lớp được chứng nhận không?* (ví dụ chỉ cập nhật `M_γ` và `ΔB_β` dưới ràng buộc `δ`). Việc khóa `F` và chỉ cập nhật `G` vẫn chỉ là thử nghiệm về **isolated actuation shift**.

---

## 5. Tính khả thi toán học và những gì chưa được bảo đảm

### 5.1. Fixed-B representation limitation: lập luận chính xác

Từ surrogate gốc: `\dot{\hat x}=CA\phi(x)+CBu` nên `∂\dot{\hat x}/∂u=CB`. Với hệ thực `\dot x=f_0(x)+G_x(x)u`, nếu `G_x(x_1)\ne G_x(x_2)` ở hai trạng thái hợp lệ, không có một `CB` **cố định** có thể khớp chính xác hai Jacobian đó. Ở các hàng lift, điều kiện khớp là `J_\psi(x)G_x(x)` hằng, chặt hơn nhiều (L1). Điều này chứng minh mismatch **về tác động tức thời**, **không** chứng minh baseline thất bại MPC trong mọi bối cảnh: với closed-loop replanning/re-encoding mỗi bước, sai lệch trong một horizon ngắn có thể nhỏ. Vì thế vẫn cần đo thực nghiệm, và kết quả "fixed-`B` đủ tốt trong closed loop" là một kết cục hợp lệ.

### 5.2. Tính nhận dạng: phân nhánh không tự động disentangle

Tại *một trạng thái x cố định*, nếu đo được `y_i=\dot x_i=f_0(x)+G_x(x)u_i` không nhiễu, có đủ hành động và ma trận thiết kế:

\[
 X=\begin{bmatrix}1&u_1^\top\\\vdots&\vdots\\1&u_K^\top\end{bmatrix},\qquad \operatorname{rank}(X)=m+1,
\]

thì `f0(x)` và từng cột của `G_x(x)` có thể nhận dạng bằng hồi quy tuyến tính **tại x ấy**. Khi dữ liệu quan sát chỉ một lệnh ứng với mỗi trạng thái, hoặc `X` thiếu hạng, việc phân rã không xác định duy nhất nếu không có ràng buộc bổ sung. Với dữ liệu trajectory liên tục, việc thu thập đủ lệnh *tại cùng x* thường không khả thi; cần giả định smoothness, coverage của lân cận, hoặc controlled resets trong simulator **chỉ cho diagnostics**. Kết quả này là bài toán regression cục bộ, không phải định lý nhận dạng toàn cục của neural latent model. Theory về action diversity/controlled world-model identifiability đã có tại [P5].

Lưu ý về dữ liệu của baseline: input ngẫu nhiên độc lập với trạng thái nên không có confounding, và MSE held-out đã đo action response *theo phân phối đó*. Khoảng cách giữa prediction và action-response fidelity chỉ lộ ra ở vùng trạng thái mà MPC ghé nhưng dữ liệu ngẫu nhiên hiếm tới, và ở chuỗi action có cấu trúc (hằng, bang-bang, kiểu MPC).

### 5.3. Controllability dưới input map phụ thuộc trạng thái — HƯỚNG TRỌNG TÂM (viết lại ở v0.3)

#### 5.3.1. Câu hỏi

Baseline bảo đảm controllability của surrogate nhờ cặp `(A_c,B_c)` ở dạng canonical. Với `\dot z=A_cz+B(z)u`, surrogate là hệ phi tuyến và tiêu chuẩn Kalman cho một cặp cố định không còn là bảo đảm. Tuyến tính hoá cục bộ chỉ cho kết luận cục bộ; điều kiện Lie bracket khó kiểm tra với mạng neural. Câu hỏi: **lớp `B(z)` nào cho phép giữ kết luận toàn cục của baseline, bằng một điều kiện ép được vào kiến trúc?**

Hai lớp dưới đây trả lời theo hai cách, và kéo về hai phía ngược nhau của đường đánh đổi.

#### 5.3.2. Ký hiệu

Với horizon `T>0`:

\[
W_T=\int_0^T e^{A_cs}B_cB_c^\top e^{A_c^\top s}\,ds,\qquad
W_T^{I}=\int_0^T e^{A_cs}e^{A_c^\top s}\,ds,
\]
\[
\lambda_T=\lambda_{\min}\!\big(CW_TC^\top\big),\qquad
\kappa_T^2=\lambda_{\max}\!\big(CW_T^{I}C^\top\big),\qquad
\boxed{\rho_T=\frac{\sqrt{\lambda_T}}{\kappa_T}.}
\]

`CW_TC^\top` là output-controllability Gramian của phần tuyến tính; `ρ_T` là **biên controllability** (controllability margin). Với bài toán state controllability trong lifted space, thay `C` bằng `I_N`. `W_T^I` là Gramian của cùng `A_c` khi "mọi hướng đều được tác động", nên `ρ_T ≤ \|B_c\|` và `ρ_T` đo mức `B_c` phủ các hướng so với trường hợp lý tưởng đó.

#### 5.3.3. Lớp A — giữ bảo đảm chính xác bằng đổi biến input

\[
B(z)=B_c\,M(z),\qquad M:\mathbb R^N\to GL(m)\ \text{liên tục}.
\]

Đặt `v=M(z)u`. Hệ trở thành `\dot z=A_cz+B_cv`, đúng là surrogate LTI canonical của baseline. Mọi tín hiệu `v(\cdot)` đều thực hiện được bằng `u=M(z)^{-1}v` dọc quỹ đạo, nên **với input không bị chặn, mọi kết luận controllability của baseline giữ nguyên**. Năng lượng: `\|u\|_{L^2}\le \sup_z\|M(z)^{-1}\|\cdot\|v\|_{L^2}`.

- **S1 thuộc Lớp A** khi mọi `g_i(x)\ne 0`; ép được bằng `g=\mathrm{softplus}(\cdot)+\epsilon` hoặc `\exp(\cdot)`.
- **Ép khả nghịch cho `m>1`**: `M=\exp(S(z))`, hoặc `M=L(z)L(z)^\top+\epsilon I`, hoặc tích của ma trận trực giao và đường chéo dương.
- **MPC**: bài toán tuyến tính theo `v`; ràng buộc `u\in\mathcal U` trở thành ràng buộc phụ thuộc trạng thái `M(z)^{-1}v\in\mathcal U`. Đây chính là chỗ cấu trúc lồi bị mất mà Appendix G.5 nhắc tới.
- **Giới hạn biểu cảm**: `\operatorname{Im}B(z)=\operatorname{Im}B_c` với mọi `z`. Input luôn đi vào qua cùng một không gian con `m` chiều. Để khớp chính xác cần `J_\phi(x)G_x(x)=B_cM(\phi(x))` với mọi `x`; riêng các hàng vật lý đòi `G_x(x)=(CB_c)M`, tức **không gian cột của `G_x` không đổi theo trạng thái**. Điều này sai với các hệ mà lực đi vào qua ma trận quay hay qua nghịch đảo ma trận quán tính (CartPole, Reacher, quadrotor phẳng).

#### 5.3.4. Lớp B — giữ bảo đảm bằng biên Gramian

\[
B(z)=B_c+\Delta B(z),\qquad \Delta B\ \text{Lipschitz},\qquad \sup_{z\in\mathbb R^N}\|\Delta B(z)\|_2\le\delta.
\]

> **Mệnh đề dự kiến (P-margin).** Nếu `δ<ρ_T` thì với mọi `z_0\in\mathbb R^N` và mọi đích `y^\star\in\mathbb R^n`, tồn tại `u\in L^2([0,T];\mathbb R^m)` đưa surrogate từ `z_0` đến `Cz(T)=y^\star`, với
> \[
> \|u\|_{L^2}\ \le\ \frac{\|y^\star-Ce^{A_cT}z_0\|}{\sqrt{\lambda_T}-\delta\,\kappa_T}.
> \]

**Phác thảo chứng minh.** Viết `Cz(T)=Ce^{A_cT}z_0+\mathcal Lu+\mathcal N(u)`, với `\mathcal Lu=\int_0^TCe^{A_c(T-\tau)}B_cu\,d\tau` và `\mathcal N(u)=\int_0^TCe^{A_c(T-\tau)}\Delta B(z_u(\tau))u(\tau)\,d\tau`. Vì `\|\Delta B(z)u\|\le\delta\|u\|` từng điểm, `\|\mathcal N(u)\|\le\delta\kappa_T\|u\|_{L^2}`. Nghịch đảo phải chuẩn nhỏ nhất `\mathcal L^\dagger y=B_c^\top e^{A_c^\top(T-t)}C^\top(CW_TC^\top)^{-1}y` có chuẩn `1/\sqrt{\lambda_T}`. Xét ánh xạ `\Phi(u)=\mathcal L^\dagger\big(y^\star-Ce^{A_cT}z_0-\mathcal N(u)\big)`. Nó liên tục (Gronwall, nhờ `ΔB` Lipschitz), có ảnh nằm trong một không gian con hữu hạn chiều nên compact, và ánh xạ quả cầu bán kính `r=\|y^\star-Ce^{A_cT}z_0\|/(\sqrt{\lambda_T}-\delta\kappa_T)` vào chính nó. Định lý điểm bất động Schauder cho `u=\Phi(u)`, và khi đó `Cz(T)=Ce^{A_cT}z_0+\mathcal L\mathcal L^\dagger(\cdot)+\mathcal N(u)=y^\star`. ∎

**Trạng thái của mệnh đề.** Phác thảo trên chưa được viết thành chứng minh đầy đủ và chưa ai phản biện. Kiểm tra số sơ bộ: trên 18 cấu hình ngẫu nhiên (`N=4`, `m=1`, cả state lẫn output controllability, `δ/ρ_T\in\{0.3,0.6,0.9\}`), phép lặp điểm bất động hội tụ sau 5–21 vòng, sai số đích ở mức sai số rời rạc hoá, và năng lượng luôn nằm dưới biên. Đây là kiểm tra hợp lý, không phải bằng chứng.

**Hệ quả dùng được ngay:**

- **Chứng nhận bằng kiến trúc.** Ép `\|\Delta B(z)\|\le\delta` bằng lớp đầu ra, ví dụ `\Delta B(z)=\delta\cdot H(z)/\max(1,\|H(z)\|_F)`. Sau huấn luyện tính `ρ_T` từ `(A_c,B_c)`; nếu `δ<ρ_T` thì mô hình được chứng nhận. Có thể đưa `\max(0,\delta-\rho_T)` vào loss.
- **Vai trò mới của Gramian regularization.** Tăng `λ_T` làm tăng ngân sách `ρ_T` cho phần hiệu chỉnh; L4 cho biết cái giá là phần tuyến tính khớp kém hơn. Đây là trục đánh đổi thứ hai.
- **Phiên bản có ngân sách năng lượng.** Với ngân sách `\|u\|_{L^2}\le E`, mọi đích thoả `\|y^\star-Ce^{A_cT}z_0\|\le E(\sqrt{\lambda_T}-\delta\kappa_T)` đều tới được. Đây là phát biểu thực tế hơn khi input bị chặn, vì controllability toàn cục không còn đúng ngay cả với hệ tuyến tính.

#### 5.3.5. Lớp hợp nhất (S3)

\[
B(z)=\big[B_c+\Delta B(z)\big]M(z),\qquad \|\Delta B(z)\|\le\delta<\rho_T,\quad M(z)\in GL(m).
\]

Đổi biến `v=M(z)u` rồi áp dụng P-margin. `M` gánh phần thay đổi *độ lớn và trộn kênh* (không tốn ngân sách); `ΔB` gánh phần thay đổi *hướng* (tốn ngân sách). Phân vai này chính là khác biệt S1/S2 ở §4.4, giờ có ý nghĩa định lượng.

#### 5.3.6. Đại lượng chẩn đoán: phần hiệu chỉnh cần thiết so với ngân sách

Trong simulator `G_x` biết được và `J_\phi` tính bằng autograd, nên với một baseline đã huấn luyện có thể đo trên miền dữ liệu `\mathcal X`:

\[
r_0=\sup_{x\in\mathcal X}\big\|J_\phi(x)G_x(x)-B_c\big\|,\qquad
r_A=\sup_{x\in\mathcal X}\big\|(I-\Pi_{B_c})\,J_\phi(x)G_x(x)\big\|,\qquad
\boxed{\chi=\frac{r_A}{\rho_T}.}
\]

`Π_{B_c}` là phép chiếu trực giao lên không gian cột của `B_c`. `r_0` là sai lệch input map của S0; `r_A` là phần sai lệch mà **Lớp A không thể sửa**, tức lượng tối thiểu `ΔB` phải gánh. `χ` so lượng đó với ngân sách:

- `χ<1`: tồn tại mô hình S3 vừa khớp input map trên `\mathcal X` vừa được chứng nhận.
- `χ` hơi lớn hơn 1: có đánh đổi thật; đường cong fidelity theo `δ` là kết quả chính.
- `χ≫1` trên mọi hệ: P-margin vô nghĩa về thực hành, cần công cụ khác (xem No-go ở §10).

Các đại lượng này tính được **chỉ với baseline đã tái lập**, không cần huấn luyện mô hình mới. Nên báo cáo cả phân vị (median, 95%) ngoài sup. Lưu ý encoder được học đồng thời nên `r_A` và `ρ_T` đều thay đổi khi huấn luyện S3; phép đo trên baseline chỉ là chỉ báo sớm.

#### 5.3.7. Những gì chưa được bảo đảm và rủi ro

1. **Kết quả định tính là cổ điển.** Controllability được bảo toàn dưới nhiễu nhỏ của `B` đã được chứng minh từ thập niên 1970 [P13], cùng dòng với các kết quả về hệ tuyến tính bị nhiễu phi tuyến bị chặn [P14]. Đóng góp tiềm năng **không phải định lý**, mà là: biên tường minh theo Gramian dùng được làm tiêu chí thiết kế, cách ép nó vào tham số hoá của mô hình học, và bằng chứng thực nghiệm về đường đánh đổi. Phải trích dẫn đúng và không trình bày như kết quả lý thuyết mới.
2. **Biên có thể rất bảo thủ.** Trong kiểm tra số, `ρ_T` chỉ ở mức `6\times10^{-3}` đến `4\times10^{-2}` với `\|B_c\|=1`, và biên năng lượng lỏng 2–60 lần. `ρ_T` có thể giảm nhanh theo `N` (latent dim của baseline là 16 trong ví dụ README). Đây là rủi ro lớn nhất của H3 và là lý do §5.3.6 phải đo trước.
3. **Chỉ là bảo đảm cho surrogate** (L5), và phát biểu trên toàn `\mathbb R^N` kể cả ngoài manifold (§4.5).
4. **Input không bị chặn.** Với `\mathcal U` bị chặn chỉ có phiên bản ngân sách năng lượng `L^2`; ràng buộc `L^\infty` chưa được xử lý.
5. **Tồn tại, không phải thuật toán.** Schauder không cho tính duy nhất hay tốc độ hội tụ. Phép lặp điểm bất động hội tụ trong thử nghiệm nhưng chưa có bảo đảm; bảo đảm co cần thêm hằng số Lipschitz của `ΔB`.
6. **Khớp định nghĩa với [P1].** P-margin phát biểu cho `Cz(T)=y^\star`. Cần đọc §3–4 và Appendix F của [P1] để xác nhận đây đúng là khái niệm output controllability mà paper dùng, và Gramian trong regularizer của paper là hữu hạn hay vô hạn horizon.
7. **Phụ thuộc `T`.** `ρ_T` thay đổi theo horizon; cần chọn `T` gắn với horizon MPC và báo cáo độ nhạy.

### 5.4. Expressiveness có giá điều khiển

Ở baseline, linear MPC trên `\dot z=Az+Bu` có cấu trúc thuận lợi. Với `B(z)u`, planning theo **physical action** nói chung là nonlinear MPC. Hai điểm cập nhật: (i) đây không phải rào cản mới, vì đã có MPC lặp dạng LPV cho mô hình Koopman có input phụ thuộc trạng thái [P16] và NMPC cho dạng bilinear [P11], nên cũng **không phải đóng góp**; (ii) với Lớp A, MPC tuyến tính theo `v` và chỉ ràng buộc input trở nên phụ thuộc trạng thái, là trường hợp nhẹ nhất. Vẫn phải đo latency/feasibility, không chỉ MSE.

### 5.5. Prediction không đủ cho planning: phép đo phải tách riêng

Giả sử tập candidate action sequences `\mathcal A` và true/estimated costs `J(A),\hat J(A)` trên cùng trạng thái. Có thể đo:

\[
\mathrm{CandidateRegret}=J(\arg\min_{A\in\mathcal A}\hat J(A))-
\min_{A\in\mathcal A}J(A).
\]

Chỉ dùng simulator làm oracle **khi đánh giá**. Công thức không là guaranteed global regret hoặc performance bound cho learned policy. Một representation hữu ích cho control phải tránh đánh giá sai hậu quả của các lệnh planner cân nhắc, chứ không chỉ tối thiểu hóa MSE trên lệnh đã thực hiện. Sai số action-response nên được chuẩn hoá theo độ lớn của chính đáp ứng và đo trên phân phối trạng thái–action do MPC sinh ra.

---

## 6. Research Questions — được ưu tiên theo thứ tự bằng chứng

| Mã | Câu hỏi | Tính chất cần đo ngay / sau này |
|---|---|---|
| **RQ0** | Dự đoán tốt trên hành động quan sát có đồng nghĩa với phản ứng đúng dưới hành động thay thế? | **Ngay khi reproduce**: on-policy MSE vs same-state counterfactual response error/ranking, trên cả phân phối do MPC sinh ra. |
| **RQ1** | Sai lệch input map của fixed-`B` lớn bao nhiêu **so với ngân sách Gramian**? | **Ngay khi reproduce**: `r_0`, `r_A`, `ρ_T`, `χ` (§5.3.6) trên từng hệ; không cần mô hình mới. |
| **RQ-C1 (H3 chính)** | Lớp `B(z)` nào giữ được bảo đảm controllability của baseline, chính xác hoặc với biên định lượng? | Chứng minh đầy đủ P-margin và Lớp A; khớp định nghĩa với [P1]; rà literature. |
| **RQ-C2** | Đường đánh đổi giữa bảo đảm và độ trung thực trông thế nào? | Action-response error và MPC theo `δ` và theo trọng số Gramian; so S0, S1 (Lớp A), S3, S2 tự do, và mốc Jacobian. |
| **RQ-C3** | Biên có đủ chặt để hữu ích không? | So biên năng lượng với năng lượng thật của MPC; `ρ_T` theo latent dim `N` và horizon `T`. |
| **RQ2 (hạ cấp)** | Input-affine có ưu thế về prediction so với LTI khi đã kiểm soát data/parameter/solver? | Đã được nghiên cứu [P8, P10]; chỉ tái xác nhận trên baseline này, không coi là đóng góp. |
| **RQ3** | Có phân biệt được drift, action effect và biến đổi chế độ từ trajectory không? | Hướng phụ; sau H3. |
| **RQ4** | Lợi ích có chuyển thành control utility, và adaptation có giữ mô hình trong lớp được chứng nhận? | Closed-loop tracking, candidate regret; adaptation là mở rộng. |

### Giả thuyết có thể bị bác bỏ

- **H0:** Fixed-`B` model vẫn đủ tốt trong closed loop có re-encode ở benchmark được chọn; không cần `B(z)`.
- **H3a:** Trên các hệ mà không gian cột của `G_x` đổi theo trạng thái, `r_A>0` rõ rệt và Lớp A (S1) không khử được sai lệch action-response; trên hệ có `G_x` hằng hướng, Lớp A là đủ.
- **H3b:** Tồn tại chế độ huấn luyện mà S3 được chứng nhận (`δ<ρ_T`) và vẫn giảm đáng kể sai số action-response so với S0/S1.
- **H3c:** Tăng trọng số Gramian làm tăng `ρ_T` và cho phép `δ` lớn hơn, nhưng có điểm tối ưu nội tại vì phần tuyến tính khớp kém đi (L4).
- **H3d:** Biên năng lượng của P-margin lỏng nhưng có cùng xu hướng với năng lượng điều khiển thật khi `δ` thay đổi.
- **H2c (giữ từ v0.2, hướng phụ):** Khi shift tác động chủ yếu lên actuator, adaptation có cấu trúc hồi phục nhanh hơn cập nhật tràn lan.
- **H2d:** Lợi ích về fidelity **không nhất thiết** dẫn tới control gain; đó là câu hỏi thực nghiệm riêng.

---

## 7. Phân biệt research gap với kết quả đã biết

| Điều đã có | Tác động đối với novelty |
|---|---|
| **P1 — ICLR 2026 Koopman controllability** | Canonical parameterization, identity lifting, Gramian loss, input encoder Appendix G.5 đã tồn tại. H3 phải là mở rộng rõ ràng, không lặp lại. |
| **P7 — Koopman form with inputs (Automatica 2024)** | Dạng lift đúng có input matrix phụ thuộc trạng thái (LPV), kèm chặn sai số khi xấp xỉ bằng LTI. **L1 là kết quả đã biết.** |
| **P8 — Bilinear Koopman realizations (RA-L 2021)** | Control-affine luôn có realization bilinear, không nhất thiết linear; bilinear thắng linear về prediction trong MPC. **"S2 > S0" không phải đóng góp.** |
| **P9, P10 — Deep Koopman input-affine (CDC 2021; 2025)** | Encoder sâu + input effect phụ thuộc lifted state + T-step loss; có so sánh các cấu trúc input linear/bilinear/input-affine/general. **Kiến trúc S2 và phép so sánh S0/S1/S2 đã có.** |
| **P11 — Koopman NMPC; P16 — LPV-MPC lặp** | Điều khiển trên mô hình bilinear/LPV đã giải được; không lấy MPC làm đóng góp. |
| **P12 — Bilinearization & controllability (CDC 2017, TAC 2022)** | Controllability của dạng Koopman bilinear đã được **phân tích** bằng phương pháp đại số Lie. Khác với H3 ở chỗ đây là phân tích trên hệ đã biết, không phải bảo đảm theo cấu trúc khi học. Phải đối chiếu kỹ. |
| **P17 — Neural Koopman Lyapunov / CBF; tube MPC với contraction** | Bảo đảm **stabilizability / an toàn / ISS** cho Koopman bilinear đã có. H3 phải phân biệt rõ controllability kiểu Kalman/Gramian với các chứng chỉ này. |
| **P13, P14 — Controllability dưới nhiễu (1972–1979)** | Kết quả định tính của P-margin là cổ điển. Không tuyên bố định lý mới. |
| **P18 — K-PWM (NeurIPS 2026); Bilinear World Models (arXiv 2026)** | World model với latent bilinear, planning hiệu quả, và ràng buộc cấu trúc trên tác động của action đã có; là đối thủ gần nhất về mặt mô hình. |
| **P2 — Iso-Dream; P3 — DWM** | Phân nhánh controllable/noncontrollable và tách world effect/action effect đã tồn tại. |
| **P4, P15 — Online / meta-adaptive Koopman** | Cập nhật operator online, kể cả kèm bảo đảm ổn định vòng kín, không mới nếu đứng riêng. |
| **P5 — Controlled-world-model identifiability (2026)** | Action excitation và counterfactual uncertainty đã được lý thuyết hóa. |

**Khoảng mở cần nghiên cứu (provisional):** một tham số hoá cho Koopman latent dynamics có input map phụ thuộc trạng thái mà **controllability của surrogate được bảo đảm theo cấu trúc**, kèm đặc trưng hoá định lượng sự đánh đổi với action-response fidelity. Vài lượt tìm kiếm chưa thấy công trình làm đúng việc này, nhưng đó **chưa phải xác nhận**. Rủi ro bị đi trước là thật: nhóm tác giả [P1] cũng đã công bố về Koopman bilinear [P19], nên mở rộng này là bước tự nhiên của chính họ. Cần rà lại literature trước mỗi gate.

---

## 8. Những lựa chọn kiến trúc không chốt sớm

| Giải pháp | Quan hệ với H3 | Quyết định |
|---|---|---|
| Hai nhánh **Prediction / Controllability** + attention | Phân chia theo task; không có đường tới bảo đảm | **Không làm.** |
| `Az + Bu` (S0) | Có bảo đảm, sai action response | **Mốc dưới về fidelity.** |
| `Az + B·M(z)u`, `M` khả nghịch (Lớp A; S1 là trường hợp đường chéo) | Giữ bảo đảm chính xác; hướng input cố định | **Ứng viên 1**; rẻ, nên thử trước. |
| `Az + (B + ΔB(z))M(z)u`, `‖ΔB‖≤δ` (S3) | Giữ bảo đảm khi `δ<ρ_T` | **Ứng viên chính của H3.** |
| `Az + (B + ΔB(z))u` tự do (S2) | Không bảo đảm | **Đối chứng**, không gọi là mới. |
| `Az + J_φ(Cz)Ĝ(Cz)u` | Khớp manifold theo cấu trúc; không bảo đảm | **Mốc trên về fidelity.** |
| `Az+B_0u+Σu_jB_jz` (bilinear) | `ΔB` tuyến tính theo `z`, không bị chặn toàn cục | **Đối chứng từ literature**; P-margin không áp dụng trực tiếp. |
| `F(z)` phi tuyến, hai encoder, attention fusion | Mất phần tuyến tính mà mọi lập luận dựa vào | **Không làm.** |

**Tiêu chí chọn phương pháp cuối:** có bảo đảm phát biểu được và đúng; biên không vô nghĩa trên benchmark; cải thiện action response so với S0/S1 sau matched-budget ablation; chi phí MPC trong giới hạn; không tuyên bố bảo đảm cho hệ thật. Không thêm gate/threshold/routing làm novelty chính.

---

## 9. Tập dữ liệu, benchmark và nguyên tắc tái lập

Paper [P1, Appendix G] sử dụng simulation trajectories (250 bước/trajectory, random admissible inputs; training lấy windows 15 bước), gồm Mountain Car, Pendulum, CartPole, Reacher, Franka, GRN (theo v0.2; chưa đối chiếu được với PDF). **Giai đoạn đầu chỉ tái lập paper gốc.** **Không yêu cầu chạy baseline bên ngoài ở giai đoạn này.**

Điều chỉnh ở v0.3:

- **Thứ tự hệ theo câu hỏi.** Pendulum và Mountain Car có `G_x` hằng: dùng để tái lập và làm trường hợp "Lớp A đủ". CartPole và Reacher có hướng của `G_x` đổi theo trạng thái (qua nghịch đảo ma trận quán tính): đây mới là nơi kiểm H3a. CartPole phải vào sớm, không chỉ "sau đó".
- **Một họ hệ đồ chơi có núm vặn**, 1–2 chiều, ví dụ `G_x(x)=R(a\,h(x))\,g_0`, với `a` điều khiển mức quay hướng input. Vẽ `r_A`, `χ`, sai số action-response và kết quả chứng nhận theo `a`. Đường cong dose-response này rẻ (vài ngày) và trả lời sớm liệu H3 có nội dung hay không.
- **Phép đo đầu tiên** sau khi tái lập là §5.3.6, không phải huấn luyện mô hình mới.

Một **khác biệt nguồn cần giải quyết**: paper nói training window **15** bước (theo v0.2); README repo chính thức cho ví dụ CLI `--window_len 50`. Ngoài ra lệnh đánh giá trong README trỏ tới `configs/...yaml` nhưng thư mục gốc của repo không hiển thị folder `configs`, và README không có lệnh chạy MPC; cần kiểm tra ngay khi clone và liên hệ tác giả sớm nếu thiếu [P6]. Chi tiết triển khai/đo kiểm ở `EXPERIMENT.md`, thứ tự thực hiện ở `ROADMAP.md`.

---

## 10. Tiêu chuẩn chứng cứ, điều kiện go/no-go

Các ngưỡng dưới đây là đề xuất khởi điểm; phải chốt số trước khi chạy, không sửa sau khi thấy kết quả.

**Go từ reproduce sang diagnosis:** code chính thức chạy đúng; biết rõ phiên bản, dữ liệu, hyperparameters và kiểu đánh giá; tái lập metric chính trong khoảng đã định trước (đề xuất: trong 1 độ lệch chuẩn qua ≥5 seed, hoặc ±20% nếu paper không báo phương sai), giải thích mọi chênh lệch.

**Go từ diagnosis sang H3 prototype** (cần cả ba): (i) `r_A` đáng kể trên ít nhất một hệ chuẩn và tăng đơn điệu theo `a` trên hệ đồ chơi; (ii) `χ` không vượt quá khoảng 10 trên ít nhất một hệ, hoặc có lộ trình rõ để tăng `ρ_T`; (iii) sai số action-response của S0 chuyển thành chênh lệch đo được trong MPC hoặc candidate regret, không chỉ do solver hay coverage.

**Go từ H3 prototype sang paper:** chứng minh đầy đủ P-margin và Lớp A, đã được một người có nền control đọc; có ít nhất một hệ mà S3 được chứng nhận và thắng S0/S1 về action response với khoảng tin cậy tách biệt; có đường đánh đổi theo `δ` và theo trọng số Gramian; có ablation model class/param budget/data/compute; novelty review cập nhật.

**No-go/pivot:**

- Tìm thấy công trình đã làm bảo đảm controllability theo cấu trúc cho input map phụ thuộc trạng thái.
- `χ≫1` trên mọi hệ và không cải thiện được: P-margin vô nghĩa. Pivot sang chứng chỉ cục bộ hoặc theo quỹ đạo, hoặc chỉ giữ Lớp A.
- Lớp A (S1) đã khử hết sai lệch trên mọi benchmark: kết quả thu gọn thành "G.5 vốn đã giữ bảo đảm", mức ghi chú ngắn.
- Fixed-`B` đủ tốt trong closed loop có re-encode: viết thành kết quả âm có đặc trưng hoá.
- Lợi ích chỉ do thêm tham số; hoặc chi phí MPC quá lớn; hoặc chỉ có bảo đảm của surrogate nhưng claim an toàn hệ thực.

Cần bổ sung vào `ROADMAP.md`: ngân sách thời gian/compute và hạn chót cho mỗi gate.

---

## 11. Thứ tự thực hiện và phạm vi tài liệu

- **RRD.md (tài liệu này):** Problem, trực giác, mô hình toán học, giả định, rủi ro, novelty boundaries, RQ/Hypotheses.
- **EXPERIMENT.md:** Protocol tái lập, tập dữ liệu, metrics, logging, phép đo ngay từ baseline (thêm §5.3.6), test matrices, quy tắc chống leakage.
- **ROADMAP.md:** Phân kỳ, thứ tự thực thi, deliverables, gate quyết định.

Thứ tự đề xuất: **P0** đọc [P1] §3–4, Appendix F, G.4–G.5 và [P7, P8, P12, P13]; clone repo, kiểm tra config/MPC → **P1** tái lập Pendulum rồi CartPole; song song chạy hệ đồ chơi dose-response → **P2** đo `r_0, r_A, ρ_T, χ` và RQ0 trên baseline → **gate** → **P3** viết chứng minh đầy đủ, hiện thực Lớp A rồi S3 → **P4** đường đánh đổi, ablation, MPC → **P5** mở rộng (adaptation trong lớp được chứng nhận) và tổng hợp paper. **Không tạo model mới trước P2.**

---

## 12. References (đã phân biệt paper chính thức và preprint)

- **[P1]** Miao, K., Wang, H., Ding, X., Gatsis, K., Krause, A., Papachristodoulou, A. (2026). *Learning Koopman Representations with Controllability Guarantees*. **ICLR 2026**. https://proceedings.iclr.cc/paper_files/paper/2026/file/e3781fb5d2c17ea7283cc2a543793bdb-Paper-Conference.pdf — đặc biệt §3–4, Appendix F, G.1–G.5.
- **[P2]** Pan, M. et al. (2022). *Iso-Dream*. **NeurIPS 2022**. https://proceedings.neurips.cc/paper_files/paper/2022/hash/9316769afaaeeaad42a9e3633b14e801-Abstract-Conference.html ; *Iso-Dream++*, **TPAMI 2024**: https://pubmed.ncbi.nlm.nih.gov/37999968/
- **[P3]** Zhang, Y.-G. et al. (2026). *DWM: Separating World Effects from Actions in Latent World Models*. **arXiv preprint**. https://arxiv.org/abs/2607.18715
- **[P4]** Gao, B. et al. (2025). *Conformal Online Learning of Deep Koopman Linear Embeddings*. **NeurIPS 2025**. https://proceedings.neurips.cc/paper_files/paper/2025/hash/784fd5a46dfe303e5b51c8621b84cf3f-Abstract-Conference.html
- **[P5]** Zhang, X. et al. (2026). *On the Identifiability of Controlled World Models*. **arXiv preprint**. https://arxiv.org/abs/2607.22430
- **[P6]** Official implementation of [P1]: https://github.com/KYMiao/Controllable-Koopman — README dùng `window_len=50`, `latent_dim=16`; có tuỳ chọn re-encode định kỳ.
- **[P7]** Iacob, L. C., Tóth, R., Schoukens, M. (2024). *Koopman Form of Nonlinear Systems with Inputs*. **Automatica**. https://arxiv.org/abs/2207.12132
- **[P8]** Bruder, D., Fu, X., Vasudevan, R. (2021). *Advantages of Bilinear Koopman Realizations for the Modeling and Control of Systems with Unknown Dynamics*. **IEEE RA-L 6(3)**. https://arxiv.org/abs/2010.09961
- **[P9]** Iacob, L. C., Beintema, G. I., Schoukens, M., Tóth, R. (2021). *Deep Identification of Nonlinear Systems in Koopman Form*. **CDC 2021**. https://arxiv.org/abs/2110.02583
- **[P10]** Iacob, L. C. et al. (2025). *Learning Koopman Models From Data Under General Noise Conditions*. **arXiv preprint** (đã nộp SIADS). https://arxiv.org/abs/2507.09646
- **[P11]** Folkestad, C., Burdick, J. W. (2021). *Koopman NMPC: Koopman-based Learning and Nonlinear Model Predictive Control of Control-affine Systems*. https://arxiv.org/abs/2105.08036 *(tên tác giả ghi theo trí nhớ, cần xác nhận)*
- **[P12]** Goswami, D., Paley, D. A. (2022). *Bilinearization, Reachability, and Optimal Control of Control-Affine Nonlinear Systems: A Koopman Spectral Approach*. **IEEE TAC 67**, 2715–2728. https://doi.org/10.1109/TAC.2021.3088802 ; bản CDC 2017: *Global bilinearization and controllability of control-affine nonlinear systems*.
- **[P13]** Davison, E. J., Ball, D. (1972). *On the global controllability of perturbed controllable systems*. **IEEE TAC 17(6)**, 825–826.
- **[P14]** Lukes, D. L. (1972). *Global controllability of nonlinear systems*. **SIAM J. Control 10**, 112–126. Xem thêm memo Berkeley ERL M79/74 về robustness của controllability dưới nhiễu phi tuyến: https://www2.eecs.berkeley.edu/Pubs/TechRpts/1979/ERL-m-79-74.pdf
- **[P15]** Han, M. et al. *MAKO: Meta-Adaptive Koopman Operators for Learning-based MPC of Parametrically Uncertain Nonlinear Systems*. https://arxiv.org/abs/2510.09042 *(bản Automatica chưa xác nhận được)*
- **[P16]** Kernel-based Koopman control operator learning với iterated LPV-MPC: https://arxiv.org/abs/2405.07312 ; Li, Z. et al. (2024), input-augmented Koopman với MPC lặp lồi: https://arxiv.org/abs/2408.02315
- **[P17]** Zinage, V., Bakolas, E. *Neural Koopman Lyapunov Control*: https://arxiv.org/abs/2201.05098 ; *Neural Koopman Control Barrier Functions*: https://arxiv.org/abs/2209.07685 ; *Tube MPC for Bilinear Koopman Models using RCCM*: https://arxiv.org/abs/2607.29538
- **[P18]** Rajkumar, S. M. et al. (2026). *K-PWM*. **NeurIPS 2026**. https://arxiv.org/abs/2611.11521 ; *Bilinear World Models*. **arXiv preprint**. https://arxiv.org/abs/2609.36305
- **[P19]** Xiong, Z., Yuan, Z., Miao, K., Wang, H., Cortés, J., Papachristodoulou, A. (2025). *Data-enabled predictive control for nonlinear systems based on a Koopman bilinear realization*. **CDC 2025**.

---

**Decision log — 12/11/2026 (v0.2).** Chốt ưu tiên H2 như *research hypothesis*, không chốt architecture hay claim novelty. Bắt đầu từ **official baseline reproduction**.

**Decision log — 12/11/2026 (v0.3).** Sau khi rà literature: kiến trúc input-affine và phép so sánh S0/S1/S2 đã có tiền lệ, nên H2 không còn là đóng góp dự kiến. Chuyển trọng tâm sang **H3 — giữ bảo đảm controllability dưới input map phụ thuộc trạng thái** (§5.3). Giữ nguyên nguyên tắc: tái lập trước, đo trên baseline trước (thêm `r_A`, `ρ_T`, `χ`), không huấn luyện mô hình mới trước gate. Online adaptation lùi thành mở rộng. Chưa claim novelty; mệnh đề P-margin mới ở mức phác thảo.
