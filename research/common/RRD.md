# RESEARCH REQUIREMENTS DOCUMENT (RRD)
## Dynamics-Factorized Learning for Controllability-Aware Latent World Models

**Phiên bản:** 0.2 — 12/11/2026  
**Trạng thái:** Research proposal; chưa xác nhận novelty, chưa triển khai phương pháp H2  
**Quyết định nghiên cứu:** Ưu tiên **H2 — phân tách Autonomous Dynamics / Action-Response Dynamics**. H1 (Prediction / Control-specific branches) giữ làm đối chứng khái niệm, **không** triển khai song song.  
**Baseline duy nhất cần tái lập:** Miao et al., *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026 [P1].  
**Tài liệu đi kèm:** `EXPERIMENT.md` (giao thức đo đạc) và `ROADMAP.md` (thứ tự nghiên cứu, tiêu chí chuyển pha).

> **Lập trường nghiên cứu:** Chúng ta không bắt đầu bằng một network hai nhánh rồi tìm lý do hợp thức hóa. Bắt đầu bằng câu hỏi *mô hình dự đoán tốt nhưng có hiểu đúng tác động của hành động không?*; chứng minh giới hạn của baseline; chỉ khi có bằng chứng mới thiết kế H2. Hai nhánh là **giả thuyết về inductive bias**, không phải đóng góp khoa học mặc nhiên.

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

**Trực giác H2:** Cho mô hình học riêng (a) điều gì xảy ra **khi không có lệnh điều khiển** và (b) **mức thay đổi do lệnh điều khiển**, rồi hợp thành **một** hệ dynamics nhất quán cho dự đoán và phân tích controllability. Phân tách dựa trên *cơ chế động lực học*, không dựa trên hai nhiệm vụ đầu ra.

### 1.3. Trực giác sâu hơn: predictive vs control-relevant representation

Một representation có thể dự đoán rất tốt trên chính sách thu thập dữ liệu nhưng phản ứng sai với hành động chưa thử. Planner luôn hỏi *what if action khác đi?* nên lỗi ở đây có thể quan trọng hơn lỗi dự đoán trung bình. Cần **đo ngay từ baseline**: (i) trajectory MSE, (ii) counterfactual action-effect error, (iii) khác biệt thứ hạng hành động/chi phí, (iv) closed-loop MPC. Đây là **giả thuyết cần kiểm định**, không mặc định rằng tồn tại xung đột representation.

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

**L1 — Fixed physical-input map.** Vì `x=Cz` và `\dot z=Az+Bu`, ta có `\dot{\hat x}=CAz+CBu`. Giữ `z` cố định, đạo hàm tức thời theo action là:

\[
\boxed{\frac{\partial\dot{\hat x}}{\partial u}=CB,\quad\text{không phụ thuộc }z.}
\]

Do đó, **trên miền trạng thái mà identity lifting đúng**, surrogate LTI không biểu diễn chính xác mọi hệ có độ nhạy tức thời đối với hành động phụ thuộc trạng thái. Thậm chí, với open-loop rollout **không re-encode**, nghiệm của chính surrogate là

\[
\hat x(T)=Ce^{AT}\phi(x_0)+\int_0^T Ce^{A(T-\tau)}Bu(\tau)\,d\tau.
\]

Trong biểu thức này, **thành phần đáp ứng của chuỗi action không phụ thuộc `x₀`** (nếu cùng `A,B,C`, cùng time horizon và action units), còn quỹ đạo tổng thể vẫn phụ thuộc `x₀` qua `φ(x₀)`. Đây là giới hạn mạnh hơn ở cấp open-loop action response; nếu ứng dụng **re-encoding định kỳ** theo quan sát mới thì phải phân tích riêng. Đây là phát biểu về *lớp mô hình*, không khẳng định MSE của baseline luôn tệ hoặc MPC luôn thất bại.

**L2 — Expressiveness–tractability trade-off đã được tác giả nhận diện.** Appendix G.5 của [P1] thử `\hat u=g_\theta(\hat x)\odot u`. Trên Pendulum 5% data, prediction MSE giảm từ `0.0318` xuống `0.0067` (báo cáo tác giả), nhưng MPC không còn có cấu trúc tuyến tính chuẩn theo physical input `u`. **Đã có input encoder**; đề xuất đơn giản thêm `g(x)` không mới.

**L3 — Không có giao thức online dynamics-shift adaptation trong paper gốc.** Toán tử được huấn luyện trên điều kiện danh định và dùng để dự đoán/MPC; paper không trình bày thuật toán học lại action-effect sau actuator drift ở deployment. Tuy vậy online Koopman identification đã tồn tại: đây là **giới hạn của baseline**, chưa là **novelty gap toàn lĩnh vực**.

**L4 — Ràng buộc prior có thể căng với fitting.** Appendix G.4 cho thấy Gramian penalty quá mạnh làm prediction/control kém, chứng tỏ trade-off về optimization/expressiveness; **chưa chứng minh** là xung đột hai representation hoặc chứng cứ bắt buộc phân nhánh.

**L5 — Output controllability của surrogate không bảo đảm control thực.** Bảo đảm theo rank/Gramian của mô hình học không tự động chuyển sang hệ phi tuyến thực dưới sai số mô hình, input bounds, dynamics shift hoặc actuator failure.

### 2.4. Giả thuyết gap (không biến giả thuyết thành kết luận)

> Trong các hệ có **state-dependent action response** hoặc **action-dependent dynamics shifts**, một lifted LTI với fixed `B` có thể suy giảm **counterfactual action-response fidelity** dù on-distribution trajectory MSE vẫn tốt. Một representation học có cấu trúc phân tách có thể giảm sai lệch này và hỗ trợ online identification; tuy nhiên cần phân biệt lợi ích do **model class**, **supervision**, **parameter count**, **action coverage** và **controller**.

**Điều sẽ bác bỏ gap:** fixed-`B` baseline vẫn dự đoán đúng action effects và điều khiển tốt trong các setting mục tiêu; hoặc chỉ sửa optimization/data coverage đã giải quyết; hoặc kiến trúc Appendix G.5 là đủ.

---

## 3. Formal problem statement

Xét hệ continuous-time:

\[
\dot x(t)=f_{\eta(t)}(x(t),u(t)),\quad x\in\mathcal X\subseteq\mathbb R^n,
\quad u\in\mathcal U\subseteq\mathbb R^m.
\]

`η` là chế độ vật lý chưa biết; trong trường hợp nghiên cứu chính, một phần dynamics thay đổi qua *action-response law*. Dữ liệu `D_train={(x(t_i),u(t_i),x(t_{i+1}),Δt_i)}`. Test time chỉ được sử dụng transition đã quan sát và lệnh đã thực hiện; **không biết** `η` và **không truy cập** outcome của hành động chưa làm.

Mục tiêu nghiên cứu có hai tầng, thứ tự bắt buộc:

- **Tầng A — Baseline diagnosis (trước phương pháp mới):** hệ hiện tại học được gì và sai chỗ nào về action response, khả năng điều khiển và thứ hạng hành động?
- **Tầng B — H2 (chỉ khi Tầng A ủng hộ):** thiết kế một structured latent dynamics có phân tách nguồn thay đổi, học action effects đúng hơn và thích ứng với dynamics shifts từ ít transition mà không hủy control utility.

**Không nằm trong scope đầu:** raw pixels, partial observability, robust safe control guarantees, mọi loại nonstationarity, world-model video generation, neural policy. Không đánh đồng state observation với image observation; không đánh đồng state-dependent action effects với actuator shifts — một cái là **model-class expressiveness**, một cái là **deployment adaptation**.

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

**Không mặc định attention hoặc concat**: phép hợp nhất xuất phát từ định nghĩa dynamics. Attention chỉ là phương án thay thế nếu có chứng cứ rằng coupling vật lý không thể mô hình hóa bằng factorization hiện tại.

### 4.4. Bảo toàn tính liên tục với baseline: một họ mô hình lồng nhau

Để sửa **ít nhất có thể** và không lẫn hiệu quả của hai thay đổi, nghiên cứu ba mức phương trình (đây là *các ứng viên lý thuyết*, không phải chỉ thị chạy nhiều mô hình ngay):

\[
\begin{aligned}
\text{S0 (paper):}\quad & \dot z=A_cz+B_cu.\\
\text{S1 (appendix):}\quad & \dot z=A_cz+B_c\big(g_\beta(Cz)\odot u\big).\\
\text{S2 (H2 candidate):}\quad & \dot z=A_cz+\big[B_c+\Delta B_\beta(z)\big]u.
\end{aligned}
\]

`A_c,B_c` ở đây **ký hiệu toán tử đã qua canonical transformation**, không khẳng định mọi phép cập nhật/`ΔB` vẫn giữ nguyên theorem của paper. S2 phân tách được **base operator** và **state-dependent action-response correction**; có thể bắt đầu với S2 có tham số nhỏ, còn nhánh autonomous vẫn tuyến tính theo `z`. Một lựa chọn structured khác:

\[
\dot z=Az+B_0u+\sum_{j=1}^m u_j B_jz
\]

(dạng bilinear Koopman). Các dạng này đều **đã có tiền lệ học thuật**. Novelty (nếu có) phải đến từ learning/identification/control mechanism có kiểm chứng, không phải phép cộng hai nhánh.

### 4.5. Tính nhất quán với latent manifold

Vì `z=φ(x)` với `x=Cz`, dynamics lý tưởng phải gần như tiếp tuyến với manifold `\mathcal M=\{\phi(x)\}`:

\[
\boxed{F(\phi(x))+G(\phi(x))u\ \approx\ J_\phi(x)[f_0(x)+G_x(x)u].}
\]

Đây là **điều kiện consistency**, không là equality có sẵn. Một mô hình có state reconstruction tốt vẫn có thể rollout latent lệch manifold. Có thể đo `e_\mathcal M(z)=\|z-\phi(Cz)\|` ở các time points như diagnostic; chỉ sau khi thấy vấn đề mới cân nhắc regularization tương ứng.

### 4.6. Online adaptation là bước sau, không phải định nghĩa H2

Nếu vật lý thay đổi:

\[
\dot x=f_{0,\eta_t}(x)+G_{\eta_t}(x)u,
\]

thì cả `f0` lẫn `G` **có thể** đổi. H2 gợi ý cập nhật *đúng thành phần chịu shift* khi nhận dạng được, nhưng không được mặc định mọi shift nằm ở action branch. Việc khóa `F` và chỉ cập nhật `G` là một thử nghiệm về **isolated actuation shift**, không là công thức giải bài toán general dynamics shift.

---

## 5. Tính khả thi toán học và những gì chưa được bảo đảm

### 5.1. Fixed-B representation limitation: lập luận chính xác

Từ surrogate gốc: `\dot{\hat x}=CA\phi(x)+CBu` nên `∂\dot{\hat x}/∂u=CB`. Với hệ thực `\dot x=f_0(x)+G_x(x)u`, nếu `G_x(x_1)\ne G_x(x_2)` ở hai trạng thái hợp lệ, không có một `CB` **cố định** có thể khớp chính xác hai Jacobian đó. Điều này chứng minh mismatch **về tác động tức thời** dưới các giả định, **không** chứng minh baseline thất bại MPC trong mọi bối cảnh. Ở rollout open-loop, *quỹ đạo nền* phụ thuộc trạng thái ban đầu qua `φ`, nhưng *độ thay đổi do một chuỗi action cố định gây ra* không phụ thuộc trạng thái ban đầu trong lifted LTI. Với closed-loop replanning/re-encoding, quan hệ kết quả–hành động có thể phụ thuộc trạng thái; vì thế vẫn cần đo thực nghiệm.

### 5.2. Tính nhận dạng: phân nhánh không tự động disentangle

Tại *một trạng thái x cố định*, nếu đo được `y_i=\dot x_i=f_0(x)+G_x(x)u_i` không nhiễu, có đủ hành động và ma trận thiết kế:

\[
 X=\begin{bmatrix}1&u_1^\top\\\vdots&\vdots\\1&u_K^\top\end{bmatrix},\qquad \operatorname{rank}(X)=m+1,
\]

thì `f0(x)` và từng cột của `G_x(x)` có thể nhận dạng bằng hồi quy tuyến tính **tại x ấy**. Khi dữ liệu quan sát chỉ một lệnh ứng với mỗi trạng thái, hoặc `X` thiếu hạng, việc phân rã không xác định duy nhất nếu không có ràng buộc bổ sung. Với dữ liệu trajectory liên tục, việc thu thập đủ lệnh *tại cùng x* thường không khả thi; cần giả định smoothness, coverage của lân cận, hoặc controlled resets trong simulator **chỉ cho diagnostics**. Kết quả này là bài toán regression cục bộ, không phải định lý nhận dạng toàn cục của neural latent model. Theory về action diversity/controlled world-model identifiability đã có tại [P5].

### 5.3. Controllability sau phân tách: không được kế thừa bảo đảm máy móc

Ở surrogate tuyến tính, cặp `(A,B)` và `C` có tiêu chuẩn rank thích hợp. Với `\dot z=F(z)+G(z)u`, hệ **phi tuyến**; Kalman rank của một cặp cố định không còn là bảo đảm tổng quát. Nếu tuyến tính hóa quanh `(z_*,u_*)`:

\[
 A_{\mathrm{lin}}=\left.\frac{\partial(F(z)+G(z)u)}{\partial z}\right|_*,\qquad
 B_{\mathrm{lin}}=G(z_*),
\]

có thể khảo sát rank / Gramian của hệ **tuyến tính hóa cục bộ**; không được diễn giải là global nonlinear controllability. Các điều kiện liên quan Lie brackets, local accessibility, actuator bounds và miền trạng thái cần được phát biểu nếu về sau muốn chứng minh bảo đảm mới.

### 5.4. Expressiveness có giá điều khiển

Ở baseline, linear MPC trên `\dot z=Az+Bu` có cấu trúc thuận lợi. Với `G(z)u`, planning tối ưu theo **physical action** nói chung trở thành nonlinear MPC; việc "toán tử vẫn tuyến tính theo effective input" không bảo đảm tập lệnh admissible/chi phí giữ lồi sau phép biến đổi phụ thuộc trạng thái. Phải đo latency/feasibility, không chỉ MSE.

### 5.5. Prediction không đủ cho planning: phép đo phải tách riêng

Giả sử tập candidate action sequences `\mathcal A` và true/estimated costs `J(A),\hat J(A)` trên cùng trạng thái. Có thể đo:

\[
\mathrm{CandidateRegret}=J(\arg\min_{A\in\mathcal A}\hat J(A))-
\min_{A\in\mathcal A}J(A).
\]

Chỉ dùng simulator làm oracle **khi đánh giá**. Công thức không là guaranteed global regret hoặc performance bound cho learned policy. Một representation hữu ích cho control phải tránh đánh giá sai hậu quả của các lệnh planner cân nhắc, chứ không chỉ tối thiểu hóa MSE trên lệnh đã thực hiện.

---

## 6. Research Questions — được ưu tiên theo thứ tự bằng chứng

| Mã | Câu hỏi | Tính chất cần đo ngay / sau này |
|---|---|---|
| **RQ0** | Dự đoán tốt trên hành động quan sát có đồng nghĩa với phản ứng đúng dưới hành động thay thế? | **Ngay khi reproduce**: on-policy MSE vs same-state counterfactual response error/ranking; kiểm tra xung đột thật hay ảo. |
| **RQ1** | Fixed-input Koopman bị hạn chế đến đâu trên state-dependent action effect? | Baseline frozen dưới controlled system variants; kiểm tra lỗi tức thời, rollout và MPC. |
| **RQ2 (H2 chính)** | Factorized autonomous/action-response learning có ưu thế so với unified operator nếu đã kiểm soát data/parameter/solver? | Chỉ sau khi baseline diagnosis xác nhận failure mode. |
| **RQ3** | Có phân biệt được drift, action effect và biến đổi chế độ từ trajectory không? | Action-coverage / excitation; oracle diagnostics tách khỏi training. |
| **RQ4** | Cải thiện action-response có chuyển hóa thành controllability/control utility và adaptation sample efficiency? | Closed-loop tracking, candidate regret, recovery; giai đoạn thử H2. |

### Giả thuyết có thể bị bác bỏ

- **H0:** Fixed-`B` model vẫn đủ tốt ở benchmark được chọn; two-branch không cần thiết.
- **H2a:** Khi `G_x(x)` biến thiên, sai số action-response tăng nhanh hơn on-policy MSE của baseline.
- **H2b:** Nếu có action variation đủ, factorized dynamics cải thiện response fidelity/transfer; **không** chỉ nhờ tăng capacity.
- **H2c:** Khi shift tác động chủ yếu lên actuator, adaptation có cấu trúc có thể hồi phục nhanh hơn cập nhật tràn lan; không dự đoán tương tự cho drift-only shifts.
- **H2d:** Lợi ích về fidelity **không nhất thiết** dẫn tới control gain; đó là câu hỏi thực nghiệm riêng.

---

## 7. Phân biệt research gap với kết quả đã biết

| Điều đã có | Tác động đối với novelty |
|---|---|
| **P1 — ICLR 2026 Koopman controllability** | Canonical OOC, identity lifting, Gramian loss, input encoder Appendix G.5 đã tồn tại. |
| **P2 — Iso-Dream (NeurIPS 2022), Iso-Dream++ (TPAMI 2024)** | Phân nhánh controllable/noncontrollable dynamics đã tồn tại; không lấy dual branch làm đóng góp. |
| **P3 — DWM (arXiv 2026)** | Tách zero-input/world effects với action-driven signal ở mức supervision; phải đối chứng lý thuyết trước khi tuyên bố factorization mới. |
| **P4 — Online Koopman research** | Cập nhật operator bằng online learning / adaptive MPC không mới nếu đứng riêng. |
| **P5 — Controlled-world-model identifiability (2026)** | Action excitation và counterfactual uncertainty đã được lý thuyết hóa; không xem đơn thuần yêu cầu action diversity là novelty. |

**Khoảng mở cần nghiên cứu (provisional):** Một phương pháp *có thể nhận dạng, có cấu trúc và tương thích control* để học **state-dependent action response** và kiểm tra tính khả dụng của nó trong dynamics shifts với ngân sách ít dữ liệu. Chưa xác nhận rằng sự kết hợp này chưa được giải quyết trong literature; các bài về bilinear Koopman / adaptive control-affine phải tiếp tục được đối chiếu trước submission.

---

## 8. Những lựa chọn kiến trúc không chốt sớm

| Giải pháp | Quan hệ với H2 | Quyết định |
|---|---|---|
| Hai nhánh **Prediction / Controllability** + attention | Phân chia theo task; cần chứng minh fusion giữ dynamics consistent | **Không ưu tiên**, chỉ đối chứng H1 sau. |
| Shared encoder + **Autonomous / Action Response** | Phân chia theo nguồn biến thiên dynamics | **Ưu tiên H2** về trực giác. |
| `Az + B(g(x)⊙u)` | Biến thể Appendix G.5 của paper | **Không gọi là mới**. |
| `Az + (B + ΔB(z))u` | Sửa tối thiểu phần action map của baseline | **Ứng viên phương trình đầu tiên** nếu cần thiết kế. |
| `Az+B_0u+Σu_jB_jz` | Bilinear Koopman | **Ứng viên có cấu trúc**, đã có tiền lệ. |
| Hai encoder độc lập, attention fusion | Tăng độ khó nhận dạng và phân tích control | **Chưa có bằng chứng để làm**. |

**Tiêu chí chọn phương pháp cuối:** biểu diễn đúng action effects; có giao thức identification hợp lệ; không tuyên bố bảo đảm sai; giữ chi phí MPC trong giới hạn; cải thiện sau matched-budget ablation. Không thêm gate/threshold/routing làm novelty chính.

---

## 9. Tập dữ liệu, benchmark và nguyên tắc tái lập

Paper [P1, Appendix G] sử dụng simulation trajectories (250 bước/trajectory, random admissible inputs; training lấy windows 15 bước), gồm Mountain Car, Pendulum, CartPole, Reacher, Franka, GRN. **Giai đoạn đầu chỉ tái lập paper gốc**, ưu tiên Pendulum, sau đó CartPole; Reacher là mở rộng nếu cần. **Không yêu cầu chạy DKO, MLP, MAKO, Iso-Dream, DWM hay bất cứ baseline bên ngoài nào ở giai đoạn này.** Chúng chỉ dùng làm literature/novelty boundary.

Một **khác biệt nguồn cần giải quyết**: paper nói training window **15** bước; README repo chính thức cho ví dụ CLI `--window_len 50`. Không áp dụng máy móc lệnh README để báo cáo "reproduced paper"; phải xác định đúng config, ghi cả giá trị và nguồn [P1, P6]. Chi tiết triển khai/đo kiểm ở `EXPERIMENT.md`, thứ tự thực hiện ở `ROADMAP.md`.

---

## 10. Tiêu chuẩn chứng cứ, điều kiện go/no-go

**Go từ reproduce sang diagnosis:** code chính thức chạy đúng; biết rõ phiên bản, dữ liệu, hyperparameters và kiểu đánh giá; tái lập được metric chính trong sai số hợp lý với báo cáo gốc, giải thích mọi chênh lệch.

**Go từ diagnosis sang H2 prototype:** xuất hiện failure mode có thể tái lập liên quan trực tiếp action-response; tín hiệu counterfactual có ý nghĩa vượt prediction MSE; kiểm tra không chỉ do solver, số liệu train hoặc action coverage.

**Go từ H2 prototype sang paper:** thắng một kiểm chứng *không tầm thường*, có ablation model class/param budget/data/compute, có improvement về control hoặc characterization chặt chẽ về khi nào không cải thiện, có novelty review cập nhật.

**No-go/pivot:** fixed-`B` đủ tốt; simple Appendix G.5 giải quyết hoàn toàn; two-branch lợi ích chỉ do thêm tham số; action effect không nhận dạng được trên protocol; lợi ích MSE phải trả giá MPC quá lớn; chỉ có bảo đảm rank của surrogate nhưng claim an toàn hệ thực.

---

## 11. Thứ tự thực hiện và phạm vi tài liệu

- **RRD.md (tài liệu này):** Problem, trực giác, mô hình toán học, giả định, rủi ro, novelty boundaries, RQ/Hypotheses.
- **EXPERIMENT.md:** Protocol tái lập, tập dữ liệu, metrics, logging, phép đo ngay từ baseline, test matrices, quy tắc chống leakage.
- **ROADMAP.md:** Phân kỳ, thứ tự thực thi, deliverables, gate quyết định. **P0 là tái lập paper; không tạo model mới trước.**

---

## 12. References (đã phân biệt paper chính thức và preprint)

- **[P1]** Miao, K. et al. (2026). *Learning Koopman Representations with Controllability Guarantees*. **ICLR 2026**. Paper: https://proceedings.iclr.cc/paper_files/paper/2026/file/e3781fb5d2c17ea7283cc2a543793bdb-Paper-Conference.pdf — đặc biệt §3–4, Appendix F, G.1–G.5.
- **[P2]** Pan, M. et al. (2022). *Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models*. **NeurIPS 2022**. https://proceedings.neurips.cc/paper_files/paper/2022/hash/9316769afaaeeaad42a9e3633b14e801-Abstract-Conference.html ; *Iso-Dream++*, **TPAMI 2024**: https://pubmed.ncbi.nlm.nih.gov/37999968/
- **[P3]** Zhang, Y.-G. et al. (2026). *DWM: Separating World Effects from Actions in Latent World Models*. **arXiv preprint**. https://arxiv.org/abs/2607.18715
- **[P4]** Online Koopman research: *Conformal Online Learning of Deep Koopman Linear Embeddings*, **NeurIPS 2025**: https://proceedings.neurips.cc/paper_files/paper/2025/hash/784fd5a46dfe303e5b51c8621b84cf3f-Abstract-Conference.html
- **[P5]** Zhang, X. et al. (2026). *On the Identifiability of Controlled World Models*. **arXiv preprint**. https://arxiv.org/abs/2607.22430
- **[P6]** Official implementation of [P1]: https://github.com/KYMiao/Controllable-Koopman — README currently illustrates `window_len=50`; check against paper before reproduction.

---

**Decision log — 12/11/2026.** Chốt ưu tiên H2 như *research hypothesis*, không chốt architecture hay claim novelty. Bắt đầu từ **official baseline reproduction**; đo **control-relevant vs predictive fidelity** ngay từ đầu, bằng phép đo bổ sung trên cùng baseline, không cần huấn luyện model khác.
