# RESEARCH REQUIREMENTS DOCUMENT (RRD)
## Expressive and Controllability-Aware Koopman Latent Dynamics
### Hai hướng nghiên cứu độc lập: H2 — Dynamics-Factorized Learning; H3 — Controllability-Preserving State-Dependent Input Maps

**Phiên bản:** 0.5 — 12/11/2026 (sửa có kiểm soát từ v0.4)  
**Trạng thái:** Research proposal; chưa xác nhận novelty, chưa triển khai H2/H3.  
**Baseline duy nhất cần tái lập ban đầu:** Miao et al., *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026 [P1].  
**Phạm vi giai đoạn đầu:** continuous-time, full-state observation, nominal dynamics, simulation trajectories. **Reproduce**: Pendulum rồi CartPole theo paper; **stress-test giả thuyết**: họ hệ có input-direction rotation, Unicycle và Planar Quadrotor. Hai mục đích không được đánh đồng.  
**Quyết định:** **Giữ H2 và H3 là hai research tracks độc lập**, cùng sử dụng dữ liệu, baseline và các phép đo chẩn đoán. Không lấy H3 thay thế H2; không ép H2 thành module con của H3.  
**Tài liệu thực thi:** `EXPERIMENT.md` và `ROADMAP.md` (chưa được cập nhật theo v0.5; mục 11 và Appendix C là yêu cầu đồng bộ, **không** mặc nhiên coi các file cũ là execution protocol hợp lệ).

> **Research position.** Chúng ta không lấy “thêm hai nhánh”, “thêm input encoder”, “cho B phụ thuộc trạng thái”, hoặc “thêm một regularizer” làm novelty mặc định. Trước hết phải đo được *failure mode* của baseline, hiểu điều kiện nhận dạng và controllability, rồi mới xác định phương pháp nào xứng đáng triển khai. Prediction fidelity, action-response fidelity, controllability **và tính khả thi tính toán của MPC** là những yêu cầu liên quan nhưng không đồng nhất. Một surrogate có bảo đảm reachability nhưng đánh mất lợi thế giải QP của Koopman không đương nhiên là cải tiến.

**Nhãn mức độ tin cậy trong tài liệu:** **[B]** kết quả đã kiểm tra trên paper nguồn; **[M]** suy ra từ mô hình/toán học với giả định nêu rõ; **[H]** giả thuyết chưa kiểm chứng; **[E]** thiết kế thực nghiệm. Số liệu chưa chạy trên mã nguồn riêng không được trình bày là kết quả của nhóm.

---

## 0. Executive research decision

| Chủ đề | Quyết định v0.4 |
|---|---|
| Bài toán lớn | Học một latent dynamics model có ích đồng thời cho dự đoán phản thực theo hành động và thiết kế điều khiển. |
| Lõi chung | Phân tích điểm yếu fixed-input Koopman qua action-response và closed-loop MPC, không chỉ MSE. |
| **H2 — học biểu diễn** | Liệu phân rã *zero-input drift* và *action response* bằng inductive bias/learning protocol riêng có cải thiện identification, prediction và control utility so với mô hình thống nhất **cùng năng lực biểu diễn**? |
| **H3 — bảo đảm mô hình** | Liệu có lớp input map phụ thuộc trạng thái đổi hướng tác động của input, có điều kiện **controllability của surrogate** khả dụng, và vẫn giữ **MPC tractability** hoặc một Pareto trade-off thuyết phục? |
| Phụ thuộc H2/H3 | **Không phụ thuộc logic.** H2 có thể thành công mà không có chứng nhận H3; H3 có thể thành công mà không có hai nhánh H2. Chỉ nghiên cứu tổ hợp H2+H3 sau hai thí nghiệm độc lập. |
| Online adaptation | Giữ là động cơ ứng dụng dài hạn và hướng mở rộng sau, không hứa chứng minh cùng lúc hai mục tiêu chính. |
| **Hard requirement — control** | Đo closed-loop performance **và** latency/feasibility của MPC với đúng physical input bounds; phân biệt convex QP chính xác với một QP xấp xỉ do frozen scheduling. |
| Bước tiếp theo | P0 audit paper/repo → P1 reproduce Pendulum/CartPole → P2 đo baseline ở benchmark gốc **và** stress-tests có cơ chế thay đổi hướng input → quyết định H2/H3 **riêng biệt**. |

---

## 1. Bối cảnh, trực giác và bài toán lớn

### 1.1. Tại sao cần một *world model* cho điều khiển?

Trong system identification / model-based control, ta quan sát trạng thái vật lý `x` và lệnh `u`, rồi học hệ chuyển trạng thái để (i) dự đoán quỹ đạo dưới một chuỗi lệnh cụ thể, (ii) dự đoán sự khác nhau khi thay chuỗi lệnh, (iii) lập kế hoạch đưa hệ đến mục tiêu dưới chi phí/giới hạn actuator. Đây là **control-oriented world modeling**, chưa đồng nghĩa với huấn luyện một policy RL hoặc mô hình video từ pixels.

Ba năng lực phân biệt:

- **Dynamics fidelity:** dự đoán `x(t)` đúng khi biết `x(0),u(·)`.
- **Counterfactual action-response fidelity:** tại *cùng trạng thái*, thay `u` có tạo ra chênh lệch biến đổi đúng không? Planner phụ thuộc trực tiếp vào điều này.
- **Control utility:** hệ học được có giúp đánh giá reachability, lập kế hoạch và điều khiển tốt, kể cả dưới giới hạn thời gian/biên độ lệnh?

Một model có thể MSE thấp trên những trajectory phổ biến mà vẫn sai khi MPC thử lệnh hiếm gặp. Điều ngược lại cũng có thể: một model không hoàn hảo về MSE nhưng xếp hạng candidate action tốt. **Đây là giả thuyết cần đo**, không là kết quả mặc nhiên. [H]

### 1.2. Trực giác về hai hướng nghiên cứu

Xe trên dốc chịu quán tính, trọng lực, ma sát và lực ga. Để dự đoán, ta cần mô tả tổng chuyển động. Để điều khiển, ta còn cần biết cùng lực ga sẽ đổi chuyển động ra sao ở mỗi trạng thái và các thay đổi đó lan truyền nhiều bước thế nào.

- **Trực giác H2:** nên cho mô hình học *phần diễn biến ở lệnh tham chiếu* và *phần thay đổi bởi action* với cơ chế học phân biệt, nhưng vẫn hợp thành **một** vector field. Không nên nhầm phần không chịu tác động tức thời với phần “không thể điều khiển”.
- **Trực giác H3:** càng linh hoạt hóa phần tác động của action, ta càng rời cấu trúc LTI mà ICLR 2026 dùng để chứng nhận controllability. Liệu có cách tăng độ biểu cảm mà vẫn giữ **một bảo đảm đúng và hữu ích** không?

Đây là **hai câu hỏi khác nhau**: H2 nói về việc học *đúng thông tin*; H3 nói về *lớp mô hình có chứng nhận*.

### 1.3. Điều không được suy diễn

1. Prediction và controllability không phải hai luật vật lý độc lập. Controllability là thuộc tính của **toàn bộ dynamics và tập lệnh điều khiển**.
2. Tách hai neural branches không tự động làm các thành phần có thể nhận dạng (*identifiable*).
3. `∂ẋ/∂u` là độ nhạy **tức thời**, không phải controllability nhiều bước; hệ `ṗ=v, ṽ=u` cho phản ví dụ rõ ràng.
4. Controllability của surrogate, dù chính xác, **không** bảo đảm plant thật controllable, MPC ổn định hoặc bảo đảm an toàn khi có model mismatch.
5. **State-dependent** input maps trong H2/H3 nói chung làm MPC gốc trở thành **nonlinear, nonconvex program** theo physical inputs. Một QP theo input hiệu dụng hoặc QP với frozen schedule không chứng minh tính lồi của bài toán gốc.

---

## 2. Baseline ICLR 2026 và giới hạn đã xác minh

### 2.1. Paper thực sự học gì? [B]

Paper [P1] học system dynamics từ trajectory full-state và input, không học neural policy, reward model hay visual encoder. Dạng mô hình:

\[
 z=\phi_\omega(x)=\begin{bmatrix}x\\\psi_\omega(x)\end{bmatrix}\in\mathbb R^N,\qquad
 \dot z=A_\theta z+B_\theta u,\qquad \hat x=Cz,\quad C=[I_n\;0].
\]

`z` là **lifted space** (`N≥n`), không phải mặc nhiên latent nén. `ψ` học neural observables; `A,B` học qua canonical parameterization và similarity transform có ràng buộc; ODE solver suy diễn trajectory, MPC tính chuỗi action. Paper bảo đảm **state-to-output controllability (SOC)** của surrogate theo phạm vi lý thuyết, qua rank criterion và cấu trúc toán tử, đồng thời dùng Gramian-based loss hỗ trợ chất lượng điều khiển. [P1, §3, Appendix B–F]

| Thành phần | Trainable? | Chức năng |
|---|---|---|
| `ψω(x)` | Có | Tạo observables bổ sung |
| Tham số hóa `Aθ,Bθ,Pθ` | Có | Learned lifted dynamics có cấu trúc |
| `C=[I,0]` | Không | Lấy physical-state output |
| ODE integrator | Không là neural parameter | Giải phương trình |
| MPC | Không là policy network | Tối ưu chuỗi lệnh |

**Ký hiệu thống nhất:** từ đây viết `A,B` là **toán tử thực tế ở lifted coordinates sau các phép biến đổi của paper**, tránh đồng thời dùng chỉ số `c` vừa cho canonical coordinates vừa cho mô hình đã biến đổi.

### 2.2. Fixed-input map: giới hạn biểu diễn được chứng minh [M]

Tại một điểm lifted hiện tại `z=φ(x)`:

\[
 \dot{\hat x}=CA\phi(x)+CBu\quad\Longrightarrow\quad
 \boxed{\frac{\partial\dot{\hat x}}{\partial u}=CB.}
\]

Độ nhạy tức thời đối với physical input là **hằng**, không phụ thuộc trạng thái. Với open-loop LTI rollout không re-encode,

\[
 \hat x(T)=Ce^{AT}\phi(x_0)+\int_0^T Ce^{A(T-s)}Bu(s)\,ds.
\]

Do đó, *phần tăng thêm do chuỗi input* không phụ thuộc `x₀` đối với cùng `A,B,C,T`. Cần tách trường hợp **closed-loop re-encoding**: solver được reset bằng quan sát mới sau mỗi bước có thể cải thiện MPC, không bác bỏ giới hạn biểu diễn cục bộ.

### 2.3. Giới hạn của phép lifting khi có input [M; nền tảng P7]

Xét plant control-affine với `u=0` hợp lệ:

\[
 \dot x=f_0(x)+G_x(x)u.
\]

Theo chain rule, khi `φ` khả vi:

\[
 \dot z=J_\phi(x)f_0(x)+\underbrace{J_\phi(x)G_x(x)}_{B^\star(x)}u,
 \qquad B^\star(x)=\begin{bmatrix}G_x(x)\\J_\psi(x)G_x(x)\end{bmatrix}.
\]

**Ngay cả khi `G_x` cố định, `B★(x)` có thể phụ thuộc trạng thái do `Jψ(x)`.** Đây không phải khám phá mới của nhóm: bài Automatica 2024 [P7] đã xác lập vai trò state-dependent input maps trong Koopman systems with inputs. Vì thế Pendulum/Mountain Car chỉ là hệ có **physical** input map tương đối đơn giản; không mặc định lifted input map có hướng không đổi.

Đồng thời, để lifted model nhất quán hoàn toàn với plant cần cả điều kiện drift:

\[
 A\phi(x)=J_\phi(x)f_0(x),\qquad B(\phi(x))=J_\phi(x)G_x(x).
\]

Đây là **điều kiện cần trên manifold cho mô hình chính xác**, không mặc nhiên thỏa được với một finite-dimensional `A`.

### 2.4. Appendix G.5: việc input encoder giữ controllability **đã có trong baseline** [B]

Paper đã xét

\[
 \hat u=g_\beta(Cz)\odot u,\quad \dot z=Az+B\hat u,
\]

và **đã nêu** rằng reachability/controllability của cặp `(A,B)` có thể được bảo toàn khi xét effective input `û`, đồng thời linear MPC trên physical input trở nên khó hơn. Table 10 xác nhận Pendulum 5%: MSE **0.0318** (không encoder) và **0.0067** (có encoder). [P1, Appendix G.5, Table 10]

**Điều chỉnh toán học khi áp dụng kết luận này:** Đối với `m>1`, câu “input encoder không triệt tiêu tất cả inputs” là **quá yếu nếu hiểu là điều kiện đủ tổng quát**. Nếu `g_1(z)=0` trên một miền mà kênh 1 thiết yếu, effective input mất một chiều, dù các kênh khác không bị triệt tiêu. Điều kiện thuận tiện, đủ mạnh cho đổi biến input là `M(z)∈GL(m)` ở mọi trạng thái cùng giới hạn inverse thích hợp; không suy ra bảo đảm cho physical inputs bị chặn.

**Hệ quả:** H3 **không** được tuyên bố mới ở lớp `B M(z)` khả nghịch. Cơ hội hẹp hơn nằm ở việc cho **không gian ảnh hưởng của input đổi hướng** trong lifted state, mà vẫn có chứng nhận có ý nghĩa.

**Giá trị baseline về control [B/M]:** Sau zero-order hold, `z_{k+1}=A_dz_k+B_du_k`; với quadratic cost lồi và ràng buộc affine, MPC theo `u` là **convex QP**. Đây là ưu thế cốt lõi cần giữ làm đối chứng. Lưu ý: *linear lifted dynamics* tự nó **chưa đủ** để MPC thành QP nếu cost/ràng buộc khác không lồi.

### 2.5. Những giới hạn khác và mức chứng cứ

- Paper không xây test-time dynamics shift adaptation protocol [B]; nhưng online Koopman learning đã có [P9]. Không lấy “online update A/B” làm novelty.
- Gramian regularization quá mạnh có thể giảm prediction/control fidelity [B, Appendix G.4]; chưa chứng minh biểu diễn hai nhiệm vụ xung đột về bản chất.
- SOC của surrogate dưới lệnh không bị chặn không tương đương control với giới hạn actuator [M].

---

## 3. Formal research problem và giả định

Xét plant liên tục:

\[
 \dot x(t)=f_{\eta(t)}(x(t),u(t)),\quad x\in\mathcal X\subseteq\mathbb R^n,\quad u\in\mathcal U\subseteq\mathbb R^m.
\]

Dữ liệu chính `D={(x(t_i),u(t_i),x(t_{i+1}),Δt_i)}` từ các trajectory; không có nhãn tách sẵn drift/action. Ở giai đoạn **chính**, `η` giữ cố định trong mỗi plant; dynamics shifts dùng về sau khi H2 hoặc H3 đã có bằng chứng. Dữ liệu *ground-truth vector field/true G_x* của simulator **chỉ được dùng làm evaluation oracle**, không được cung cấp ngầm cho training method mới.

**Giả định nền:**

- Full-state `x` được quan sát; sampling/ODE solver biết `Δt`; input history khớp với state transitions.
- Phạm vi phân tích chính: hệ control-affine với `u=0∈U`. Nếu không, dùng input tham chiếu `u_ref` và phải chỉnh mô hình drift tương ứng.
- `φ` khả vi khi dùng Jacobian-based diagnostic. Encoder ReLU được hiểu có Jacobian gần như khắp nơi; các mệnh đề yêu cầu smoothness mạnh hơn cần quy định architecture tương ứng hoặc làm việc trong miền smooth.
- Chứng nhận controllability chỉ phát biểu cho **learned surrogate** trừ khi có một định lý riêng nối sai số model với plant.
- `U=R^m` chỉ trong những định lý reachability không ràng buộc; thí nghiệm MPC dùng **input bounds thực tế**.

**Tiêu chí control bắt buộc:** Bên cạnh closed-loop task cost, dùng thời gian giải thực tế, tỉ lệ solver thành công, độ lệch so với deadline và input-constraint violations. Kiểm soát cùng `Δt`, horizon, hardware, tolerance, điều kiện khởi tạo, và cùng physical cost. **Không** đánh giá phương pháp chỉ bằng MSE hoặc chứng chỉ toán học.

**Không nằm trong scope đầu:** raw pixels, POMDP, video generation, safety guarantee cho robot thật, học policy RL, global controllability của mọi hệ phi tuyến và mọi loại nonstationarity.

---

## 4. Nền tảng toán học chung H2–H3

### 4.1. Phân rã theo action: định nghĩa chứ chưa là causal disentanglement

Với `0∈U`, định nghĩa:

\[
 f_0(x)=f(x,0),\qquad \Delta_f(x,u)=f(x,u)-f(x,0),
\]
\[
 \boxed{f(x,u)=f_0(x)+\Delta_f(x,u),\quad\Delta_f(x,0)=0.}
\]

Với hệ control-affine:

\[
 \Delta_f(x,u)=G_x(x)u,\quad G_x(x)=\partial f(x,u)/\partial u.
\]

`f0` là **zero-input drift**, không đồng nghĩa *non-controllable dynamics*. Một chiều state không chịu input tức thời vẫn có thể điều khiển sau nhiều bước thông qua drift coupling.

### 4.2. Identifiability: điều kiện đủ cục bộ, định danh ở mức phân phối và khả năng học hữu hạn mẫu

**Mục đích:** không nhầm "phải lặp đúng cùng một trạng thái" thành điều kiện cần cho mọi thuật toán, nhưng cũng không nhầm *identifiability lý thuyết* với việc hai MLP chắc chắn phân tách đúng từ dữ liệu thưa. [M]

**A. Hồi quy tại một trạng thái cố định (điều kiện đủ tường minh).** Khi đo được đạo hàm không nhiễu `y_i=f₀(x)+G_x(x)u_i` tại **cùng x** và

\[
 X_x=\begin{bmatrix}1&u_1^\top\\\vdots&\vdots\\1&u_K^\top\end{bmatrix},\qquad \operatorname{rank}(X_x)=m+1,
\]

thì các hệ số `f₀(x),G_x(x)` là duy nhất trong mô hình affine theo `u` tại **điểm x ấy**. Nếu rank thiếu, không thể định danh duy nhất tại điểm đó mà không thêm giả định. Điều kiện này **không yêu cầu** phải được thỏa bằng phép reset vật lý trong mọi bộ dữ liệu; đây là minh họa cơ sở đại số.

**B. Identifiability ở mức phân phối (không cần dữ liệu lặp đúng một giá trị x).** Giả sử

\[
 y=f_0(x)+G_x(x)u+\varepsilon,\quad
 \mathbb E[\varepsilon\mid x,u]=0,\quad
 \Sigma_u(x)=\operatorname{Cov}(u,u\mid x)\succ0,
\]

với các conditional moments tồn tại và có thể xác định. Khi đó:

\[
 \operatorname{Cov}(y,u\mid x)=G_x(x)\Sigma_u(x),\quad
 \boxed{G_x(x)=\operatorname{Cov}(y,u\mid x)\Sigma_u(x)^{-1}},
\]
\[
 \boxed{f_0(x)=\mathbb E[y\mid x]-G_x(x)\mathbb E[u\mid x]}.
\]

Đây là kết quả *population identification* tại những `x` mà conditional distribution có nghĩa, **không** cho sample-complexity bound, không áp dụng nguyên xi nếu nhiễu phụ thuộc `u` hoặc khi `y` là finite-difference derivative có sai lệch rời rạc hóa. Dữ liệu trajectory chỉ cho phép ước lượng các conditional moments thông qua tính trơn, inductive bias và coverage; chiều trạng thái lớn làm bài toán hữu hạn mẫu khó hơn nhiều.

**C. Phản ví dụ khi thiếu action excitation.** Nếu dữ liệu chỉ thực thi `u=\bar u(x)` tại mỗi `x`, với bất kỳ ma trận hàm `H(x)∈R^{n×m}`, đặt

\[
 f'_0(x)=f_0(x)+H(x)\bar u(x),\qquad G'_x(x)=G_x(x)-H(x).
\]

Ta vẫn có `f'₀(x)+G'ₓ(x)\bar u(x)=f₀(x)+Gₓ(x)\bar u(x)` trên dữ liệu, nhưng dự đoán counterfactual khác khi `u≠\bar u(x)`. Do đó trajectory fit không đủ để tuyên bố factorized dynamics đã học đúng cơ chế vật lý.

**D. Lifted identifiability có thêm giới hạn.** Khi `z=φ(x)=[x;ψ(x)]`, điều kiện vật lý ở trên chưa tự bảo đảm các nhánh `F(φ(x)),G(φ(x))` được nhận dạng ở *ngoài* manifold hoặc đạt đúng `Jφ f₀, JφGₓ`. Nếu không quan sát `\dot x` trực tiếp, training sử dụng rollout ODE; phải đo approximation/solver error riêng.

**Kết luận dành cho H2:** *same-state multi-action* là một **công cụ chẩn đoán sạch trong simulator**, không phải điều kiện bắt buộc của learning algorithm; còn `Σ_u(x)\succ0`, coverage, regularity và số lượng dữ liệu mới là các yếu tố cần thử nghiệm. Random-action rollouts giảm bớt confounding do chính sách có điều kiện, nhưng không đảm bảo kích thích đủ và coverage ở vùng mà MPC sẽ tới.

### 4.3. Physical vs lifted consistency

Với `z=φ(x)` và `x=Cz`, định nghĩa manifold `M_φ={φ(x):x∈X}`. Một learned model `ẋz=F(z)+G(z)u` có trajectory đúng trên manifold nếu:

\[
 \left. F(z)\right|_{z=\phi(x)}=J_\phi(x)f_0(x),\qquad
 \left. G(z)\right|_{z=\phi(x)}=J_\phi(x)G_x(x).
\]

Diagnostics:

\[
 e_{\mathrm{drift}}(x)=\|F(\phi(x))-J_\phi(x)f_0(x)\|,
\]
\[
 e_{\mathrm{input}}(x)=\|G(\phi(x))-J_\phi(x)G_x(x)\|,
\]
\[
 e_{\mathcal M}(z)=\|z-\phi(Cz)\|.
\]

Các phép đo dùng oracle vector field nếu simulator cung cấp; không thay thế cho MPC evaluation. `e_M` phụ thuộc tỉ lệ latent coordinates, cần report normalized variant.

### 4.4. Controllability không phải action sensitivity

Với `ṗ=v, ṽ=u`, `∂ṗ/∂u=0` nhưng hệ linear controllable vì `rank[B,AB]=2`. Bất kỳ metric H2 nào chỉ đo `G` tức thời đều phải đi kèm **multi-step response** và **closed-loop MPC**; đối với H3, phải kiểm tra **SOC của toàn hệ**, không dùng chỉ số local Jacobian để suy ra global nonlinear controllability.

---

## 5. Track H2 — Dynamics-Factorized Representation Learning (độc lập)

### 5.1. Research question và giả thuyết

**RQ-H2:** Nếu giữ chung một latent state, liệu chuyên biệt hóa việc học **drift** và **action response** có cải thiện counterfactual dynamics identification và downstream control so với monolithic predictor sau khi matched-budget, dữ liệu, solver và controller được kiểm soát?

**H2a [H]:** Factorized learning giúp giảm sai số action response ngoài phân phối, không chỉ trajectory MSE trên dữ liệu thu thập.  
**H2b [H]:** Cải thiện đến từ inductive bias/supervision, **không chỉ** từ việc cho mô hình input-affine nhiều biểu cảm hơn fixed-`B`.  
**H2c [H]:** Khi dữ liệu không đủ action excitation, factorization có thể thất bại; hiệu quả phụ thuộc coverage.  
**H2d [H]:** Cải thiện dynamics fidelity không chắc cải thiện MPC; cần đo riêng.  
**H2e (future) [H]:** Cập nhật action branch dưới isolated actuator shift có thể ít dữ liệu hơn full retraining; chưa là đóng góp chính giai đoạn đầu.

### 5.2. Ba kiến trúc để tách hiệu ứng model class khỏi learning mechanism

**H2-0 (paper, làm mốc):**

\[\dot z=Az+Bu.\]

**H2-1 (cùng model class input-affine, KHÔNG dùng two-branch làm novelty):**

\[\dot z=Az+[B+\Delta B_\beta(z)]u.\]

Đây là control-affine Koopman/LPV model đã có tiền lệ; H2-1 trả lời: *chỉ tăng expressiveness có giải quyết failure mode không?*

**H2-2 (hai dynamics branches trong một state chung):**

\[
 z=\phi_\omega(x),\qquad
 \boxed{\dot z=F_\alpha(z)+G_\beta(z)u},\qquad \hat x=Cz.
\]

`Fα` học zero-input drift; `Gβ` học action response. **Không** độc lập về semantics: cả hai phải cùng mô tả **một** vector field và tôn trọng lifted-manifold consistency. Có thể thiết kế `Fα(z)=Az+r_α(z)` như một ablation; **nếu `rα≠0` thì không được kế thừa bảo đảm SOC của baseline**.

**H2-3 (matched-class monolithic control-affine):** Sử dụng một **shared-trunk network** sinh đồng thời `a_ξ(z)∈R^N` và `D_ξ(z)∈R^{N×m}`, sau đó ghép thành `h_ξ(z,u)=a_ξ(z)+D_ξ(z)u`. So sánh với H2-2 có hai parameter groups riêng, nhưng **cùng control-affine function class**, số tham số/FLOPs và training data tương đương. Nếu khác biệt chỉ do H2-2 có thêm capacity hoặc H2-3 dùng `h(z,u)` với lớp hàm không tương đương, không thể quy kết gain cho cơ chế phân nhánh. Một generic `h_ξ(z,u)` không affine theo `u` có thể dùng làm **expressive comparator phụ**, không phải matched-class control.

**Fusion:** không mặc định attention. Phép hợp nhất mặc định xuất phát từ `F+Gu`. Khi muốn dùng attention, trước hết chứng minh ablation rằng coupling learned attention tốt hơn structured composition với cùng số tham số, không mất consistency.

### 5.3. Learning signals, nhận dạng và hai giao thức dữ liệu không được trộn

**Protocol A — Observational (đề xuất là task chính).** Chỉ dùng `(x_t,u_t,x_{t+1},Δt)` do exploration policy tạo ra; không được lấy ground-truth `f₀`, `Gₓ` hay outcome của hành động chưa thực hiện từ simulator để làm nhãn. Các tín hiệu có thể nghiên cứu sau baseline diagnosis:

- Multi-step trajectory loss dưới chuỗi action **đã thực thi**; theo dõi rollout drift và off-manifold consistency.
- Structural identity `G(z)u=0` khi `u=0` (thuộc **lớp mô hình**, tự nó không là dữ liệu chứng minh drift thật đúng).
- Action-coverage-aware loss/regularization *chỉ khi* có lập luận rõ ràng về điều kiện thông tin, không dùng pseudo-counterfactual labels chưa quan sát.
- Nếu trong dữ liệu có các state rất gần nhau với nhiều action, dùng local regression/conditioning diagnostics; không gọi "đủ gần" là bằng chứng đã khử bias khi chưa lượng hóa smoothness và sai số xấp xỉ.

**Protocol B — Controlled intervention (tách thành thí nghiệm phụ, không lẫn Protocol A).** Cho phép simulator reset về một trạng thái rồi thi hành nhiều hành động khác nhau *để thu thập dữ liệu train*. Đây là **giao thức thông tin mạnh hơn**, không phải data leakage nếu tách đúng train/test, công khai số lượt tương tác và so sánh với baseline có **cùng quyền truy cập và ngân sách dữ liệu**. Không lấy cải thiện B so với A làm đóng góp riêng của architecture H2.

**Evaluation-only oracle:** vector field thật, `Gₓ`, cùng-state action differences và mọi counterfactual rollout từ simulator chỉ sử dụng ở held-out evaluation của Protocol A. Tách quyền truy cập oracle bằng code/config và log để tránh leakage.

**Matched comparisons:** **H2-2 vs H2-3** là so sánh chính về *factorization/parameter sharing* vì cùng control-affine function class; **H2-0 vs H2-1** là so sánh phụ về *expressiveness*. Phải matched số tham số/FLOPs, dữ liệu, optimizer, horizon và solver nơi có thể; kiểm tra có khác biệt kết quả từ inductive bias/learning protocol hay chỉ từ capacity. Việc đặt hai nhãn branch khác nhau không tự tạo expressive advantage. Đánh giá practical identifiability theo `m`, latent dimension `N`, action covariance eigenvalues, trajectory budget và vùng `x` do MPC ghé. [E]

### 5.4. Điều kiện đánh giá độc lập H2

H2 **GO** khi baseline có meaningful action-response failure trên ít nhất một stress-test **được đăng ký trước**, action excitation/coverage đủ đo được, và H2-2 tốt hơn **H2-3 matched-class** (đồng thời đánh giá H2-1 để loại tác động của việc tăng expressiveness) **sau khi matched train data, solver và compute** trên counterfactual response **và/hoặc** MPC với kết quả có độ bất định rõ ràng.  
H2 **NO-GO** nếu H2-3 matched-class đã đạt hiệu quả tương đương, hoặc H2-1 / Appendix G.5 đã khắc phục failure mode với chi phí thấp hơn, nếu gains do nhiều tham số, hoặc nếu việc tách drift/action không nhận dạng được với data thực tế.

**Đóng góp kỳ vọng (chưa xác nhận):** một learning mechanism biểu diễn/cập nhật dynamics có phân tách nhưng duy trì consistency; không tuyên bố *dual branch* là mới.

---

## 6. Track H3 — Controllability-Preserving State-Dependent Input Maps (độc lập)

### 6.1. RQ-H3 và điều đã có trong literature

**RQ-H3:** Có lớp state-dependent input maps đổi được **hướng** action effect trong lifted state mà vẫn có điều kiện **SOC của surrogate** đủ khả dụng, đồng thời duy trì **convex/efficient MPC** trong một giao thức được nêu rõ hoặc cải thiện Pareto frontier chất lượng điều khiển–độ trễ so với baseline QP không?

H3 không cần phải có neural two-branch theo H2: giữ `F(z)=Az` và tập trung vào `B(z)`. Paper đã xét `B M(z)` ở Appendix G.5; controllability under bounded perturbation là một hướng lý thuyết cổ điển. **Novelty (nếu có):** thiết kế lớp học + chứng nhận hữu ích + **planning tractability được đánh giá nghiêm ngặt**. Đây không phải phép biến đổi input hay riêng mệnh đề perturbation; literature CDC 2024 về controllability preservation cho Koopman bilinear và SIAM 2025 về nonlinear input transformation giữ MPC convexity là đối chứng trực tiếp [P10, P11].

### 6.2. Họ mô hình và ký hiệu

\[
\begin{aligned}
\text{S0 (paper):}\quad &\dot z=Az+Bu,\\
\text{S1 (Appendix G.5 / class A):}\quad &\dot z=Az+B M(z)u,\\
\text{S2 (unconstrained input-affine):}\quad &\dot z=Az+[B+\Delta B(z)]u,\\
\text{S3 (certified candidate):}\quad &\dot z=Az+[B+\Delta B(z)]M(z)u.
\end{aligned}
\]

`A∈R^{N×N}`, `B∈R^{N×m}`, `C∈R^{n×N}`; `ΔB(z)∈R^{N×m}`, `M(z)∈R^{m×m}`. Trong S3 yêu cầu `M(z)` khả nghịch và `ΔB` bounded/Lipschitz theo điều kiện sau. Đây là **candidate class**, không phải thuật toán đã chứng minh ưu thế. **S0** giữ dynamics tuyến tính để MPC là QP dưới cost/constraints lồi; **S1–S3** nói chung có dynamics phi tuyến theo chuỗi `u` và **không tự giữ tính lồi** của physical-input MPC. Phải báo cáo rõ solver/QP approximation dùng cho từng mô hình.

### 6.3. Lớp A — đổi biến đầu vào, giữ reachability nhưng không mặc nhiên giữ convex QP [M]

Giả sử `M:R^N→GL(m)` liên tục. Với physical input `u`, đặt `v(t)=M(z(t))u(t)`; hệ đổi thành:

\[\dot z=Az+Bv.\]

Với bất kỳ quỹ đạo LTI khả dĩ sinh bởi **input liên tục không bị chặn** `v(·)` trên `[0,T]`, đặt `u(t)=M(z_v(t))^{-1}v(t)`. Tính liên tục và khả nghịch trên quỹ đạo compact bảo đảm `u(·)` liên tục; vậy reachable outputs theo unrestricted physical input trùng với LTI. **Đây không phải kết quả mới** (baseline Appendix G.5 đã lập luận trên effective input; ở đây điều kiện được nói chặt hơn).

Nếu `sup_z ||M(z)^{-1}||₂≤K_M`, ta có `||u||_{L²}≤K_M||v||_{L²}`. Nếu `u∈U` có biên độ, ràng buộc thực biến thành:

\[\boxed{M(z_k)^{-1}v_k\in\mathcal U}\]

phụ thuộc biến quyết định trạng thái `z_k`, **nói chung không lồi**. Ngoài ra cost vật lý `\sum_k\|u_k\|_R^2=\sum_k\|M(z_k)^{-1}v_k\|_R^2` không phải một quadratic cost hằng trong `v`.

**Một QP có thể giữ lại bằng xấp xỉ/chọn tập bảo thủ, nhưng không phải QP tương đương đầy đủ:** lấy tập virtual input cố định `V` sao cho `V⊆\bigcap_{z∈Z}M(z)U`, và dùng **quadratic cost trên v** hoặc upper bound convex được chỉ rõ. Nếu `V` có biểu diễn affine/polyhedral, tối ưu `z_{k+1}=A_dz_k+B_dv_k` là QP trên **tập con chính sách** và có thể có cost khác physical objective. Phải đo mất mát reachable set và control performance. Một tập `V` rỗng hoặc quá nhỏ không bảo toàn hiệu quả ứng dụng.

**Hạn chế expressiveness:** `Im(BM(z))=Im(B)` ở mọi `z`; chỉ thay tỷ lệ/trộn input channels, không thay input subspace trong lifted state. Đổi hướng sang một subspace khác buộc có correction ngoài span(B) hoặc thay đổi lớp embedding.

### 6.4. Controllability margin — phiên bản đủ giả định để kiểm định [M]

Với `T>0` cố định và `C` lấy physical output, đặt:

\[
W_T=\int_0^T e^{As}BB^\top e^{A^\top s}\,ds,\qquad
Q_T=\int_0^T e^{As}e^{A^\top s}\,ds,
\]
\[
W_y=CW_TC^\top,\quad Q_y=CQ_TC^\top,\quad
\lambda_T=\lambda_{\min}(W_y),\quad
\kappa_T=\sqrt{\lambda_{\max}(Q_y)},
\]
\[
\boxed{\rho_T=\frac{\sqrt{\lambda_T}}{\kappa_T}.}
\]

`λ_T>0` khi `(A,B,C)` là SOC ở horizon `T`; `κ_T>0` nếu `C` không triệt tiêu. `W_y` là output Gramian. `Q_T` dùng để khống chế ảnh hưởng của perturbation từ `R^N` sang output; **đây không phải Gramian loss trong paper gốc**. `ρ_T` là margin của chứng nhận **đề xuất**, không phải universal controllability margin.

**Mệnh đề P-margin (đưa ra với các giả định tường minh; cần phản biện độc lập trước khi công bố).** Giả sử:

1. `A,B,C` cố định, `(A,B,C)` SOC, do đó `W_y≻0` cho `T>0`.
2. `ΔB:R^N→R^{N×m}` globally Lipschitz và `sup_z||ΔB(z)||₂≤δ`.
3. Inputs được phép liên tục và **không bị chặn biên độ**.
4. `δ<ρ_T`.

Khi đó với mọi `z₀∈R^N` và `y★∈R^n`, hệ

\[\dot z=Az+[B+\Delta B(z)]u,\qquad y=Cz\]

có **ít nhất một** input liên tục chuyển output từ `Cz₀` tới `y★` tại thời gian `T`, và có thể chọn sao cho:

\[
\boxed{\|u\|_{L^2(0,T)}\le
\frac{\|y^\star-Ce^{AT}z_0\|_2}{\sqrt{\lambda_T}-\delta\kappa_T}.}
\]

**Chứng minh đầy đủ trong phạm vi giả định (xem chi tiết ngay dưới):** Định nghĩa toán tử LTI `\mathcal L:L²([0,T];R^m)→R^n` và perturbation:

\[
\mathcal Lu=\int_0^T Ce^{A(T-s)}Bu(s)\,ds,
\quad
\mathcal N(u)=\int_0^T Ce^{A(T-s)}\Delta B(z_u(s))u(s)\,ds.
\]

Từ Cauchy–Schwarz, `||\mathcal N(u)||≤δκ_T||u||₂`. Vì `W_y=\mathcal L\mathcal L^* ≻0`, nghịch đảo phải chuẩn nhỏ nhất

\[
\mathcal L^\dagger q(t)=B^\top e^{A^\top(T-t)}C^\top W_y^{-1}q,
\quad \mathcal L\mathcal L^\dagger=I_n,
\quad\|\mathcal L^\dagger\|=1/\sqrt{\lambda_T}.
\]

Đặt `d=y★−Ce^{AT}z₀`, `α=δκ_T/√λ_T<1` và xây dựng ánh xạ **hữu hạn chiều**

\[
\Psi(q)=d-\mathcal N(\mathcal L^\dagger q),\qquad q\in\mathbb R^n.
\]

Với `R=||d||/(1−α)`, ta có `||Ψ(q)||≤||d||+α||q||≤R` nếu `||q||≤R`. `Ψ` liên tục: `q↦\mathcal L^\dagger q` là tuyến tính hữu hạn chiều vào tập điều khiển liên tục bị chặn theo chuẩn khi `q` bị chặn; nghiệm ODE phụ thuộc liên tục vào `q` nhờ Lipschitz của `ΔB` và giới hạn toàn cục; tích phân định nghĩa `\mathcal N` cũng liên tục. Định lý **Brouwer** trên quả cầu đóng `B_R⊂R^n` cho `q★=Ψ(q★)`.

Chọn `u★=\mathcal L^\dagger q★`; đây là tín hiệu **liên tục theo t**. Khi đó

\[
\mathcal L u^\star+\mathcal N(u^\star)=q^\star+\mathcal N(u^\star)=d,
\]

nên `Cz(T)=y★`. Hơn nữa

\[
\|u^\star\|_{L²}\le \frac{\|q^\star\|}{\sqrt{\lambda_T}}
\le \frac{\|d\|}{\sqrt{\lambda_T}-\delta\kappa_T}.
\]

**Phạm vi kết luận:** lập luận tồn tại trên `R^N` cho learned surrogate dưới input không bị chặn, không phát biểu input-constrained MPC, uniqueness của điều khiển, stability hoặc controllability của plant. **Trạng thái novelty:** đây là lập luận perturbation/fixed point cần đối chiếu với lý thuyết cổ điển và được chuyên gia control phản biện trước khi xem là đóng góp mới; các bước trên không tạo bảo đảm nếu những giả định bị bỏ.

**Đánh giá phản biện "unstable A làm ρ_T tất yếu tiến về 0" [M].** Kết luận ấy **không đúng tổng quát** vì tử và mẫu cùng chứa `e^{As}`. Với hệ scalar `\dot z=az+bu`, `a>0`, `C=1`:

\[
 W_y=b^2\frac{e^{2aT}-1}{2a},\quad Q_y=\frac{e^{2aT}-1}{2a},\quad \boxed{\rho_T=|b|\quad\forall T>0}.
\]

Tuy nhiên, **margin có thể co rất nhanh** trong hệ nhiều mode. Ví dụ:

\[
 A=\operatorname{diag}(1,-1),\quad B=(1,1)^\top,\quad C=I_2,
\]

\[
 Q_y=\operatorname{diag}\left(\frac{e^{2T}-1}{2},\frac{1-e^{-2T}}{2}\right),\quad
 W_y=\begin{bmatrix}(e^{2T}-1)/2&T\\T&(1-e^{-2T})/2\end{bmatrix}.
\]

Suy ra `ρ_T` với `T=0.5,1,2,4,8` lần lượt xấp xỉ `0.1472, 0.1841, 0.1126, 0.0181, 0.000335`. Tức **không đơn điệu theo T**, nhưng có thể suy biến mạnh với horizon dài. Rủi ro thực chất là *anisotropic modal controllability, coordinate conditioning, horizon* chứ không phải chỉ `Re λ(A)>0`.

**Kiểm tra giá trị chứng nhận:** đo `ρ_T` theo `N,T`, seeds, Gramian regularization, gauge/normalization và `||B||`; so correction thực cần với `δ<ρ_T`. Nếu certified correction quá nhỏ đến mức không giảm action-response error hoặc không giúp MPC, H3 theo chứng chỉ global này là **NO-GO**, không cứu bằng cách chỉ tăng prediction capacity.


### 6.5. S3 kết hợp M và ΔB: chứng nhận, tham số hóa M và chi phí physical control

\[\boxed{\dot z=Az+[B+\Delta B(z)]M(z)u.}\]

Khi `M(z)∈GL(m)` liên tục, đổi `v=M(z)u` đưa S3 về dạng P-margin. Nếu `sup_z||M(z)^{-1}||≤K_M`, bound **trên physical control energy** trở thành:

\[\boxed{\|u\|_{L^2}\le K_M\frac{\|y^\star-Ce^{AT}z_0\|}{\sqrt{\lambda_T}-\delta\kappa_T}.}\]

**Không dùng penalty mềm để "chứng nhận"** `M` khả nghịch ở mọi `z`; cần hard parameterization và kiểm tra số. Một lựa chọn đơn giản, không phải SVD/Cayley bắt buộc:

\[
 M_\gamma(z)=I+E_\gamma(z),\qquad
 E_\gamma(z)=\gamma\frac{H_\theta(z)}{\max(1,\|H_\theta(z)\|_F)},\quad 0<\gamma<1.
\]

Vì `||E_γ||₂≤||E_γ||_F≤γ`, theo bounds singular values:

\[
 \boxed{1-\gamma\le\sigma_{\min}(M_\gamma(z))\le \sigma_{\max}(M_\gamma(z))\le1+\gamma},\qquad
 \|M_\gamma(z)^{-1}\|_2\le\frac1{1-\gamma}.
\]

Đây là **đủ** nhưng giới hạn biểu diễn ở lân cận `I`; có thể thử positive-definite/orthogonal-factor parameterization cho `m≥2` sau khi chứng minh cần thiết, không mặc định làm SVD network. Chuẩn hóa bằng `max` là liên tục và Lipschitz nếu `Hθ` Lipschitz hữu hạn; nếu định lý/ODE solver cần smoothness cao hơn, sử dụng phép chuẩn hóa smooth và chứng minh vẫn bounded.

**Giới hạn của chứng nhận:** `u` bounded, state-dependent cost/constraints và MPC tractability không suy ra từ energy bound. Với physical cost và physical constraints, phải dùng NMPC hoặc QP xấp xỉ đã công bố rõ, không được gọi bài toán gốc là convex QP.

### 6.6. Sửa lỗi chỉ số χ của v0.3 — KHÔNG dùng làm feasibility gate

Với target lifted input map từ oracle `G★(x)=Jφ(x)G_x(x)`, v0.3 định nghĩa:

\[
r_A=\sup_{x\in X}\|(I-\Pi_B)G^\star(x)\|_2,
\qquad \chi=r_A/\rho_T.
\]

`r_A` là **độ lệch của G★ khỏi không gian cột của B**; nó cho biết Lớp A `BM(x)` không thể triệt tiêu phần nào. Nhưng **χ<1 không đủ** và **χ≫1 không loại trừ** khả năng S3 khớp `G★`, vì S3 còn phép đổi biến `M(x)` có thể khuếch đại/thu nhỏ hiệu chỉnh `ΔB`. Vì vậy phải xóa hai mệnh đề “χ<1 ⇒ tồn tại S3 vừa khớp vừa certified” và “χ≫1 ⇒ H3 vô dụng” khỏi v0.3.

**Phản ví dụ một trạng thái:** Lấy `A=[[0,1],[0,0]]`, `B=[0,1]ᵀ`, `C=I₂`, `T=1`. Khi đó `ρ_T≈0.197`. Với target `G★=[0.01, 0.0001]ᵀ`, `r_A=0.01` ⇒ `χ≈0.051<1`, nhưng bất kỳ factorization `G★=(B+Δ)M` (`M` scalar khác 0) đòi khoảng cách từ `B` tới `span(G★)` gần `0.99995>ρ_T`, nên không thể vừa **khớp đúng** vừa certified. Ngược lại, với `G★=[10,1000]ᵀ`, `χ≈50.8`, nhưng chọn `M=1000`, `Δ=[0.01,0]ᵀ` thì khớp chính xác và `||Δ||=0.01<ρ_T`.

**Diagnostics được sửa:**

1. Giữ `r_A` làm **Lớp A approximation gap**, không gọi là feasibility certificate S3.
2. Đo **orientation mismatch**:

   \[
   d_{\mathrm{orient}}(x)=\|(I-\Pi_{G^\star(x)})B\|_2,
   \]

   với `Π_G` là projector lên không gian cột của `G` (dùng SVD/pseudoinverse ổn định). Nếu có **exact** S3 `G★=(B+Δ)M`, với `M` invertible, thì `Im(G★)=Im(B+Δ)`; do đó `d_orient(x)≤||Δ(x)||≤δ`. Suy ra điều kiện **cần** `sup_X d_orient≤δ<ρ_T`, **không đủ** cho existence của một `M(z),Δ(z)` trơn/toàn cục.
3. Để xử lý scaling/conditioning, định nghĩa **pointwise constrained factorization distance**:

   \[
   d_\star(x;\underline\sigma,\overline\sigma)
   =\inf_{M:\,\underline\sigma\le\sigma_{\min}(M),\,\sigma_{\max}(M)\le\overline\sigma}
   \|G^\star(x)M^{-1}-B\|_2.
   \]

   Nếu `sup_X d★≥ρ_T`, không thể exact-match trên `X` bằng S3 thỏa cả margin và singular-value bounds. Nếu `sup_X d★<ρ_T`, đó mới là **tín hiệu khả thi từng điểm**, vẫn chưa chứng minh tồn tại global Lipschitz network `M,Δ`, hay chất lượng control. **Không gọi phép tối ưu pointwise là certificate đầy đủ.**
4. Report thêm residual physical-input map `||CB(z)−G_x(x)||`, latent lifted residual `||B(z)−G★(x)||`, và các phân vị P50/P95 trên tập trạng thái test. `sup` theo sample chỉ là sample maximum, không phải mathematical supremum trên continuum.

### 6.7. Đo ngân sách Gramian đúng cách [M/E]

`λ_min(W_y)` tăng **không tự động** khiến `ρ_T` tăng vì `κ_T` cũng có thể thay đổi. Đo **trực tiếp** `ρ_T` theo latent size `N`, horizon `T`, learning seed và Gramian-loss weight. `ρ_T` không phụ thuộc đơn lẻ vào stability của `A`: các mode không được kích thích đồng đều và norm/gauge của lifted coordinates quyết định conditioning.

**Yêu cầu thực nghiệm:** chuẩn hóa physical state/action units; giữ thống nhất `C` và latent coordinate convention khi so sánh; báo cáo cả spectral quantities `λ_min(W_y)`, `κ_T`, `||B||`, `cond(W_y)`; thử `T` từ MPC horizon và thêm sweep ngắn/dài. Một mô hình được chứng nhận ở horizon `T₀` không tự động được chứng nhận với cùng `δ` ở mọi horizon.

**Mốc quyết định H3:** So `d_orient`, `d★` với `ρ_T` trên ít nhất một stress-test đổi hướng input. Không dùng scalar unstable toy làm lý do loại margin; không dùng kết quả trên Pendulum dễ để loại cả track. Chỉ dùng `sup` nếu có chứng minh/interval bound trên compact domain; sample maxima là diagnostic.

### 6.8. Điều không được tuyên bố

- P-margin là **bảo đảm của surrogate** với `ΔB` bounded globally, không là bảo đảm của physical plant.
- `SOC` với unrestricted input không đồng nghĩa MPC tối ưu/lồi, input-bounded reachability, stability, safety.
- `A` có mode unstable không tự chứng minh `ρ_T→0`; cũng không có bảo đảm `ρ_T` không suy giảm khi đổi `N,T`.
- QP với *virtual input* hoặc *frozen scheduling* không chứng minh MPC phi tuyến theo **physical input** là convex.
- `d_orient<ρ_T` chỉ **cần** cho exact S3; điều kiện không xác nhận model tồn tại.
- Không gọi class A, generic LPV/bilinear Koopman, bounded-perturbation robustness hay state-dependent encoder là novelty độc lập.

### 6.9. Ranh giới convex MPC: hard design constraint chung cho H2 và H3

**LTI baseline QP.** Dưới zero-order hold, `z_{k+1}=A_dz_k+B_du_k`. Nếu stage/terminal costs quadratic với ma trận trọng số PSD và constraints tuyến tính/polyhedral, bài toán finite-horizon theo toàn bộ chuỗi `u₀,…,u_{H−1}` là convex QP. Khi QP khả thi và solver hội tụ, nghiệm là global optimum **của mô hình và objective đã nêu**, không phải bảo đảm tối ưu cho plant thật.

**State-dependent input models.** Với `\dot z=Az+B(z)u`, nghiệm liên tục và discrete transition nói chung phi tuyến theo các input quá khứ. Nếu ràng buộc physical action và/hoặc true cost phụ thuộc `z,u`, MPC toàn horizon nói chung thành **nonconvex NLP**. Mất convexity là một **rủi ro thiết kế chính**, không phải chi tiết implementation; một certified controllability class **không** khắc phục điều này.

**Ba chế độ control bắt buộc phân biệt trong experiments:**

1. **QP-exact baseline:** S0 (hoặc S1 trong điều kiện rất đặc biệt làm tương đương một QP trên `v`, với tập/cost được mô tả chính xác).
2. **Sequential/frozen-schedule QP:** lấy `\bar z_k` từ trajectory trước, cố định `\bar B_k=B(\bar z_k)` và giải QP cho hệ tuyến tính biến thiên theo thời gian, **lặp nếu cần**. Đây là local/iterative approximation, không được gọi global convexity của nonlinear MPC; phải báo cáo số QP iterations và mismatch so với full dynamics.
3. **Nonlinear MPC:** multiple shooting/SQP hoặc method khác trực tiếp tối ưu trên state-dependent dynamics; có nguy cơ local minima và latency cao. Phải báo cáo khởi tạo, tolerances, maximum iterations, timeout và tỉ lệ converged/feasible.

**Ví dụ cách frozen schedule rời rạc hóa nhất quán:** với `u` giữ hằng trên bước `Δt`, cố định lịch `\bar z_k(s)` và đặt `\bar B_{d,k}=\int_0^{\Delta t}e^{A(\Delta t-s)}B(\bar z_k(s))\,ds`; khi đó `z_{k+1}\approx A_dz_k+\bar B_{d,k}u_k` tạo LTV-QP. Nếu dùng đơn giản `B(\bar z_k(0))`, ghi rõ thêm sai số rời rạc hóa.

**H3 hard Go/No-Go:** Ít nhất một cấu hình target phải cho **thực chất** trade-off giữa (a) action-response/candidate-regret/closed-loop improvement, (b) usable controllability certificate và (c) solver latency/feasibility trong ngân sách. Nếu H3 chỉ tốt dưới NMPC có computation vượt xa baseline, claim phải thu hẹp thành *certified modeling*, không gọi là nâng cấp hiệu quả MPC. Không có định lý hiện tại nào bảo đảm đồng thời S3 **bất kỳ** + physical-input bounded **và** convex QP.

**Đối chứng phải đọc:** Cibulka–Korda–Haniš (SIAM JADS 2025) có nonlinear input transformation nhưng giữ **convexity của một online MPC formulation cụ thể** [P11]. Điều đó không đồng nghĩa mọi `M(z)` đều cho MPC lồi. Kết quả về bilinear controllability preservation từ CDC 2024 [P10] cũng hạn chế novelty của việc nói "H3 là lần đầu bảo toàn controllability trên Koopman phi tuyến".

---

## 7. Quan hệ H2–H3: độc lập trước, kết hợp sau

| Kết quả | H2 | H3 | Kết luận nghiên cứu |
|---|---|---|---|
| H2 tốt, H3 không hữu ích | Action-response learning cải thiện prediction/control | Certificate quá bảo thủ | Giữ paper track H2 nếu có novelty đủ sâu; không ép chứng nhận. |
| H2 không tốt, H3 tốt | Factorization không chứng minh lợi ích | Certified model class cho gain rõ | H3 có thể thành paper độc lập. |
| Cả hai tốt | Learning specialization có ích | Certified input maps có ích | Thử **H2+H3** dưới matched-budget; kiểm tra mutual interference. |
| Cả hai không tốt | Không gain | Không usable margin | Viết negative characterization hoặc pivot, không hợp thức hóa kiến trúc bằng lời. |

**Không được coi `H2 → H3` là chuỗi bắt buộc.** H3 có thể huấn luyện trực tiếp một `ΔB(z),M(z)` với shared encoder mà không triển khai drift branch; H2 có thể học `F,G` phi tuyến mà không yêu cầu class H3.

Nếu kết hợp về sau, phải giải quyết lại bản chất chứng nhận: `F(z)=Az+r(z)` thay vì `Az` làm mệnh đề P-margin không còn áp dụng nguyên trạng. Chỉ kết hợp khi drift vẫn `Az` hoặc phát triển một định lý mới có ràng buộc riêng cho `r`.

---

## 8. Research questions, falsifiable hypotheses, evidence levels

| Mã | Câu hỏi | Khi nào đo? | Kết quả bác bỏ |
|---|---|---|---|
| **RQ0** | MSE tốt có che action-response/ranking error không? | Sau reproduce; dùng **một** baseline frozen | Không thấy sai lệch có ý nghĩa trên target tasks. |
| **RQ1** | Fixed-`B` sai bao nhiêu ở physical/lifted input và drift? | Sau reproduce, oracle diagnostics | Sai số rất nhỏ trên các vùng planner đi tới. |
| **RQ-H2-1** | Hai nhánh có tốt hơn monolithic **cùng model class** không? | Prototype H2 | Gain biến mất khi matched-capacity/supervision. |
| **RQ-H2-2** | Identifiability lý thuyết có chuyển thành practical estimation với finite random rollouts và coverage thực tế không? | H2; đo conditional action covariance/coverage trước | Không có khả năng recovery ổn định của drift/action effect dù đủ dữ liệu để dự đoán. |
| **RQ-H3-1** | S1 bảo toàn SOC đến đâu; cần `M` có điều kiện gì? | Theory, trước S3 | Chỉ có effective-input claim nhưng physical bounds vô dụng. |
| **RQ-H3-2** | P-margin có chứng minh đầy đủ và hữu ích không? | Theory + baseline diagnostics | Bound quá nhỏ; d★ luôn vượt bound. |
| **RQ-H3-3** | S3 có cải thiện error và **control–compute Pareto frontier** so với S0/S1/S2 không? | Prototype H3, same physical constraints/horizon | S3 chỉ thêm tham số/solver cost hoặc chỉ giữ chứng nhận hình thức. |
| **RQ-Bench** | Failure mode có tồn tại trên benchmark đổi hướng input mà không chỉ do coverage/solver? | Ngay P2: toy-rotation + Unicycle/Planar Quadrotor | Chỉ xuất hiện khi simulator thiếu nhất quán hoặc out-of-distribution không kiểm soát. |
| **RQ-Future** | Adaptation có giữ được chứng nhận khi dynamics shift? | Sau hai tracks | Certificate không cập nhật được khi model học online. |

**Hypotheses trước khi chạy:**

- **H0:** Fixed-`B` + closed-loop re-encoding đã đủ tốt trên **stress-tests đã pre-register**; không cần H2/H3 trong phạm vi đã xét. Kết quả âm trên riêng Pendulum không đủ để chốt H0.
- **H2:** Factorized learning tăng out-of-distribution action-response fidelity sau matched-budget mà không mất control utility.
- **H3-A:** Lớp A đủ trong các trường hợp lifted input **không đổi hướng** (không suy từ physical `G_x` hằng).
- **H3-B:** Có ít nhất một target task mà `d★` dưới usable `ρ_T` trên phần miền test liên quan, cho phép S3 vừa certified vừa cải thiện response fidelity.
- **H3-C:** Gramian-weight/δ có đường đánh đổi hữu ích; tăng Gramian weight không bảo đảm tăng margin.
- **H-control:** Những improvement đáng kể về dynamics phải được kiểm chứng bằng candidate regret **và/hoặc** MPC trong cùng physical constraints, đồng thời **đo solver latency/feasibility**; không tự động coi prediction improvement là control gain.

---

## 9. Novelty boundaries và đối chứng học thuật

| Nguồn | Điều đã có / ranh giới novelty |
|---|---|
| **[P1] ICLR 2026** | Koopman SOC-preserving canonical parameterization, Gramian-based learning; **input encoder Appendix G.5 và thảo luận preservability đã có**. |
| **[P2] Iso-Dream / Iso-Dream++** | Disentangled controllable/noncontrollable world representations đã có; hai nhánh không tự tạo novelty. |
| **[P3] DWM (preprint 2026)** | Separate world-effect/action-effect learning đã có; H2 phải khác learning objective/identification evidence. |
| **[P4] Automatica 2024 Koopman with Inputs** | State-dependent lifted input maps/LPV và approximation gap của LTI đã có; `JφG` là known result. |
| **[P5] Bilinear Koopman (RA-L 2021; TAC 2022)** | Bilinear representation/control và liên hệ controllability đã có; S2/bilinear không tự mới. |
| **[P6] Online Koopman learning (NeurIPS 2025)** | Adaptation của Koopman embeddings/operators đã có; online learning đơn thuần không mới. |
| **[P7] Controlled world-model identifiability (preprint 2026)** | Action diversity/identification limits đã được nghiên cứu; không gọi rank của local regression là discovery. |
| **[P8] Classical robustness of controllability** | Các kết quả perturbation controllability đã có; H3 phải có **useful learnable certificate** và thực nghiệm, không nhận bounded-perturbation tồn tại là đóng góp lý thuyết mới. |
| **[P10] Choi et al., CDC 2024** | *On the Controllability Preservation of Koopman Bilinear Surrogate Model* đã phân tích output controllability và output ε-controllability (với giả định cụ thể) của bilinear surrogate. H3 phải so đúng loại guarantee và giả định; **không thể claim đầu tiên** bảo toàn controllability cho nonlinear-input Koopman. |
| **[P11] Cibulka et al., SIAM JADS 2025** | *Dictionary-Free Koopman MPC with Nonlinear Input Transformation* có một formulation **giữ convexity của online MPC**. H3 không thể claim input transform + convex MPC là chưa có; phải chỉ rõ khác nhau ở model class, state-dependence, constraints và guarantees. |
| **[P12] Control-Coherent Koopman Modeling, CDC 2024** | Xây input matrix chính xác trong một lifted modeling framework nhờ chọn state variables, dùng MPC trên robot. H2/H3 cần đối chiếu trước khi claim `learn exact action effect` là mới. |
| **[P13] Koopman-based control with closed-loop guarantees (2025)** | Bilinear surrogate + approximation error bounds + controller guarantees là một hướng khác với surrogate SOC; cần phân biệt controllability guarantee, stability và robust closed-loop claims. |

**Novelty check còn mở:** phải rà thêm certified LPV/bilinear control-affine Koopman, neural parametrizations đảm bảo output reachability, input-constrained certificates, và các công trình của chính nhóm [P1] sau ICLR 2026. “Chưa tìm thấy” ≠ “chưa ai làm”.

---

## 10. Dataset, benchmark và phân tầng thực nghiệm

### 10.1. Reproduction: chỉ chạy lại paper ICLR 2026

Paper [P1, Appendix G.1–G.2] sinh full-state trajectories dưới random controls, **250 time steps/trajectory**, training window **15** bước. Repo README có ví dụ `--window_len 50`, không được coi tương đương trước khi audit. Trong P0–P2 **chỉ huấn luyện baseline [P1]**, không cần reproduce DKO, MAKO, DWM, Iso-Dream hoặc các baseline nghiên cứu khác.

**Hai tập benchmark có nhiệm vụ khác nhau:**

| Tập | Hệ | Lý do chọn | Được dùng làm Go/No-Go H2/H3? |
|---|---|---|---|
| **Reproduction** | Pendulum | Reproduce paper, check solver/Jacobian, input encoder Appendix G.5; `Gₓ` vật lý hằng. | **Không, nếu đứng riêng**. |
| **Reproduction** | CartPole | Reproduce paper, phân tích khác nhau theo góc / vùng trạng thái. | **Không, nếu chỉ miền gần cân bằng**. |
| Reproduction phụ | Mountain Car, Reacher | Theo paper; physical action map đơn giản/multi-joint coupling. | Không dùng kết quả âm tính ở môi trường dễ để đóng track. |
| **Synthetic stress-test** | `G_a(x)=R(a h(x))g₀` (2D hoặc có drift) | `a` kiểm soát chính xác mức quay hướng input; đo response–orientation–certificate dose-response. | **Có**, nếu protocol đã đăng ký trước. |
| **Stress-test** | **Unicycle** | `G(θ)=[[cos θ,0],[sin θ,0],[0,1]]` đổi hướng input rõ; Lie-bracket control khác direct rank. | Có, với domain/angle và input set được mô tả. |
| **Stress-test** | **Planar Quadrotor** | Hướng thrust tác động lên gia tốc phụ thuộc attitude `θ`; có coupling và physical bounds. | Có, với rõ ràng thrust/torque action. |
| Mở rộng | Acrobot, Reacher | Underactuation/coupled mass matrix; nhiều trạng thái hoặc actuators. | Chỉ sau khi protocol trên ổn định. |

**Không được dùng con số “CartPole chỉ đổi 10–15%” làm kết luận chung** nếu chưa cố định model parameters và miền `θ`: cần tính trực tiếp `Gₓ(x)` từ phương trình thật trong setup. Pendulum/Mountain Car có physical `Gₓ` hằng **không** suy ra lifted `JφGₓ` hằng.

**Cẩn trọng cơ chế benchmark:** Unicycle được lấy với `u=(v,ω)` và dynamics `\dot x=v\cos θ, \dot y=v\sin θ, \dot θ=ω`; nếu thay `u` thành acceleration/steering thì đó là plant khác. Planar Quadrotor có `\ddot x=-(T/m)\sin θ`, `\ddot z=(T/m)\cos θ-g` và angular torque input theo mô hình được ghi rõ; physical thrust bounds phải được áp dụng như nhau. Synthetic rotation dùng orthogonal matrix `R(·)` thích hợp cho **chiều input effect đã định**, không tự suy ra global controllability của plant.

**Cách đo hard-stress:** giữ fixed `data budget`, train/test state region (kể cả MPC visitation), action bounds, solver, control targets; thay một tham số `a` của state dependence, rồi vẽ action-response/candidate regret/closed-loop cost theo `a`. Khi stress tăng, model error có thể tăng do OOD; phải kiểm soát coverage và báo cáo tách phần này.

### 10.2. Shared diagnostics P2 — không train model mới

1. `MSE_1`, `MSE_H` trên held-out trajectories, open-loop và periodic re-encode, ghi rõ re-encode frequency.
2. Same-state physical action-effect error trên simulator **evaluation only**: `||(f_model(x,u)−f_model(x,u_ref))−(f_true(x,u)−f_true(x,u_ref))||`; có cả normalized variant và paired action differences.
3. Lifted input-map mismatch `||B−JφGₓ||` và drift mismatch `||Aφ−Jφf₀||` (nếu oracle field có sẵn), và `e_manifold`. Không giả định H2 components đã identifiable chỉ từ MSE.
4. Gramian `λ_min(W_y)`, `cond(W_y)`, `κ_T`, `ρ_T` theo `N,T`, loss weight, seeds và **stable/unstable modes**; không mặc định unstable `A⇒ρ_T→0`.
5. `r_A`, `d_orient`, `d★` với bounds trên `M` định trước; **không sử dụng χ làm feasibility gate**, và không coi sample `max` là supremum trên toàn domain.
6. MPC tracking/success, physical-control effort, candidate regret và **solver regime**: exact QP vs frozen-schedule QP vs NMPC; latency median/P95, iterations, timeouts, convergence/feasibility, physical input violations, model mismatch của QP xấp xỉ.
7. Coverage map: train-like states vs MPC-visited states; action `Σ_u(x)`/local design conditioning, nearest-neighbor distances theo chuẩn state scale, và số lần action thực sự thăm dò từng vùng; đo phân phối riêng.
8. Positive/negative control cases: scalar unstable LTI có `ρ_T` không co theo `T`; multi-mode unstable toy có `ρ_T` co; constant physical-input Pendulum; rotation-input stress.

### 10.3. Track-specific experiments — chỉ sau gates riêng

**H2 (observational primary):** fixed-input baseline → input-affine one-branch → factorized two-branch → matched-model-class/capacity monolithic control; action-excitation sweep, train budget, state dimension, noisy vs clean observations và ODE solver sensitivity. Chỉ nếu cần, dùng **Protocol B controlled interventions** như thí nghiệm phụ có `training privilege` được báo cáo và ngân sách cùng mức cho đối chứng.

**H3:** S0 → S1 (Appendix G.5) → S2 unconstrained → S3 certified; vary `δ`, `M` singular-value bounds, Gramian weight, `N,T`; measure **actual certificate**, action-response error, MPC latency, physical input bounds và closed-loop success. So solver costs bằng hardware + tolerance giống nhau. Nếu dùng virtual-input QP hoặc frozen scheduling phải báo **độ mất control utility/feasibility so với full NMPC**, không so QP surrogate như thể nó tối ưu đúng nonlinear program.

### 10.4. Nguyên tắc tính khoa học

- Khóa seeds/hyperparameters/stop criteria *trước* held-out; report mean/uncertainty và positive/negative tests qua nhiều runs.
- Không trộn trajectory vào train/test; nếu generalization qua dynamics regime, split theo regime.
- Counterfactual reset, true `f,Gₓ` và `∂f/∂u` từ simulator chỉ dùng diagnosis/evaluation của Protocol A; training đặc quyền phải có nhãn Protocol B.
- Matched physical `Δt`, action bounds, control cost, MPC horizon, solver tolerances, compute ceilings, deadline và QP/NMPC regime khi so final methods.
- Reproduction failure trên benchmark gốc là blocker để diễn giải H2/H3 gain, nhưng **không** mặc nhiên bác bỏ H2/H3; cần sửa pipeline trước.

## 11. Roadmap và Go/No-Go độc lập

| Phase | Việc chính | Deliverable | Điều kiện chuyển pha |
|---|---|---|---|
| **P0 — Audit** | Lock official commit; paper-vs-README config; dataset/ODE/MPC scripts | Reproduction manifest | Source, solver, evaluation, seed và physical units được chốt hoặc sai khác được log. |
| **P1 — Reproduce** | **Chỉ** train/eval [P1] trên Pendulum và CartPole | Checkpoints, logs, metric table | Sai khác trong tolerance pre-register; nếu không, audit pipeline trước. |
| **P2a — Synthetic stress diagnosis** | Tạo `G_a` với controlled rotation; đo baseline theo `a`; Unicycle/Planar Quadrotor theo dynamics rõ ràng | Response/ranking/control curves | Xác định được model-class vs coverage vs solver error. **Không cần train model khác**. |
| **P2b — Margin & MPC diagnosis** | `d_orient`, `d★`, `ρ_T` theo `N,T`, stable/unstable; QP vs model response | Certificate-feasibility report | Có khoảng correction có ý nghĩa, và hiểu giá điều khiển của input state-dependence. |
| **GATE H2** | Check action excitation/conditional covariance, mismatch và same-class fair design | H2 GO/NO-GO | Có failure mode ở stress-case và thiết kế matched-budget hợp lệ; Pendulum âm không loại H2. |
| **GATE H3** | Proof/literature audit; usable margin; MPC convexity/latency assessment | H3 GO/NO-GO | Không chỉ có certificate hình thức: có correction window **và** tractable planning path hoặc Pareto target thực tế. |
| **P3-H2** | Prototype observational H2; optional Protocol B tách biệt | Independent H2 report | Cải thiện response/utility so **H2-3 cùng class**, không chỉ so H2-1. |
| **P3-H3** | Theorem review, hard-M parametrization; S1/S2/S3; QP approximation và NMPC | Independent H3 report | Chứng nhận đúng + có ích + solver trade-off rõ. |
| **P4** | Robustness, significance, latent scaling, out-of-distribution control | Paper-evidence report | Claims khớp experimental evidence; không lấy 1 task thắng làm đủ. |
| **P5 optional** | H2+H3 hoặc online adaptation | New scoped RRD | Chỉ sau independent track evidence. |

**Hard Go/No-Go logic:**

- **H2:** kết luận thất bại chỉ sau khi đánh giá ít nhất một stress-test đổi hướng tác động input, coverage được đo, và so cùng lớp hàm/năng lực. Nếu dữ liệu observational không đủ kích thích, phân biệt **data-limited** với **architecture-limited** thay vì đổi mô hình để trốn vấn đề.
- **H3:** kiểm chứng đồng thời (i) model-input map cần correction đủ lớn, (ii) có certificate hữu ích dưới giới hạn `M,δ`, (iii) control-performance/latency trade-off có ý nghĩa. Nếu S3 chỉ dùng được NMPC rất chậm, **không** coi là giữ ưu thế Koopman Convex QP; pivot sang track certified modeling thuần túy hoặc dừng.
- **P-margin:** phải được control expert phản biện độc lập; một mệnh đề đúng nhưng cực bảo thủ không đủ cho phương pháp thực nghiệm.
- **Không xóa kết quả âm tính:** báo cáo đầy đủ margin không đạt, solver timeouts, identifiability thất bại và những trường hợp S0 closed-loop đã đủ tốt.

**Pre-registration:** ≥5 seeds cho báo cáo học mô hình khi khả thi; khóa data budget, solver tolerance, peak memory, per-step compute ceiling, horizon, input bounds, evaluation regimes và milestone deadline trong `ROADMAP.md` **trước khi xem kết quả**. Không đặt ngưỡng % improvement tùy ý dựa trên một lần chạy.

## 12. Đóng góp khoa học kỳ vọng và tiêu chuẩn thành paper

**Track H2 — Có thể viết paper khi** có một learning mechanism phân rã dynamics có thể nhận dạng/kiểm chứng, vượt các alternative cùng model class/parameter budget trên counterfactual action fidelity và có ý nghĩa trong closed-loop MPC; không chỉ hai heads hoặc add-loss.

**Track H3 — Có thể viết paper khi** tồn tại certified model class/training scheme có chứng nhận đúng, hữu ích định lượng (không chỉ tồn tại hình thức), thể hiện **Pareto trade-off giữa action response, controllability và MPC solver efficiency** dưới physical-input bounds. Nếu không giữ được convex MPC, phải chứng minh rõ lợi ích vượt chi phí và không tuyên bố thừa hưởng global-QP optimality của [P1]. Nếu P-margin quá bảo thủ, có thể chuyển sang trajectory/local certificate nhưng phải **đổi lại scope/định lý**, không giữ global claim.

**Paper tổng hợp H2+H3** chỉ có ý nghĩa nếu kiểm chứng được synergy rõ rệt: representation được học tốt hơn mà chứng nhận vẫn hữu ích. Việc kết hợp đơn thuần hai modules không cấu thành đóng góp.

**Chưa xác nhận:** novelty so với toàn literature, khả năng tái lập reproduction, chất lượng định lý mới sau phản biện, H2/H3 có cải thiện hay không. Không dự đoán venue acceptance trước khi có kết quả.

---

## 13. References và provenance

**Nguồn đã đối chiếu với bài gốc và trang công bố trong lần cập nhật:**

- **[P1]** Miao, K., Wang, H., Ding, X., Gatsis, K., Krause, A., Papachristodoulou, A. (2026). *Learning Koopman Representations with Controllability Guarantees*. **ICLR 2026**, §3–4, Appendix B–G, Table 10. https://proceedings.iclr.cc/paper_files/paper/2026/file/e3781fb5d2c17ea7283cc2a543793bdb-Paper-Conference.pdf
- **[P4]** Iacob, L. C., Tóth, R., Schoukens, M. (2024). *Koopman form of nonlinear systems with inputs*. **Automatica 162**, 111525. https://doi.org/10.1016/j.automatica.2024.111525
- **[P6]** *Conformal Online Learning of Deep Koopman Linear Embeddings*. **NeurIPS 2025**. https://proceedings.neurips.cc/paper_files/paper/2025/hash/784fd5a46dfe303e5b51c8621b84cf3f-Abstract-Conference.html

**Nguồn đối chứng bổ sung đã kiểm tra ở mức abstract/full-text metadata (chưa đọc lại toàn bộ proofs):**

- **[P10]** Choi, J., Cho, M., Park, H., Vijay, V., Hwang, I. (2024). *On the Controllability Preservation of Koopman Bilinear Surrogate Model*. **IEEE CDC 2024**, IEEE CDC 2024 (page range cần đối chiếu trước manuscript). https://css.paperplaza.net/images/temp/CDC/files/1350.pdf — output controllability và output ε-controllability dưới giả định của nghiên cứu.
- **[P11]** Cibulka, V., Korda, M., Haniš, T. (2025). *Dictionary-Free Koopman Model Predictive Control with Nonlinear Input Transformation*. **SIAM Journal on Applied Dynamical Systems 24(1)**, 94–130. https://doi.org/10.1137/23M1597174 — một formulation giữ convexity của online MPC.
- **[P12]** *Control-Coherent Koopman Modeling: A Physical Modeling Approach*. **IEEE CDC 2024**, DOI: 10.1109/CDC56724.2024.10886771. https://ieeexplore.ieee.org/document/10886771/ — input matrix được thiết kế phù hợp với actuator dynamics.
- **[P13]** Strässer, R., Berberich, J., Schaller, M., Worthmann, K., Allgöwer, F. (2025). *Koopman-based control of nonlinear systems with closed-loop guarantees*. https://www.degruyterbrill.com/de/document/doi/10.1515/auto-2024-0162/html — bilinear surrogate, error bound và closed-loop controller; khác SOC.

**Nguồn tham khảo cho novelty review, chưa coi là đã xác minh mọi chi tiết theorem:**

- **[P2]** *Iso-Dream* (NeurIPS 2022): https://proceedings.neurips.cc/paper_files/paper/2022/hash/9316769afaaeeaad42a9e3633b14e801-Abstract-Conference.html ; *Iso-Dream++* (TPAMI 2024): https://pubmed.ncbi.nlm.nih.gov/37999968/
- **[P3]** *DWM: Separating World Effects from Actions in Latent World Models* (2026, preprint): https://arxiv.org/abs/2607.18715
- **[P5]** Bruder, Fu, Vasudevan (2021), *Advantages of Bilinear Koopman Realizations for the Modeling and Control of Systems with Unknown Dynamics*, **IEEE RA-L**: https://arxiv.org/abs/2010.09961 ; Goswami & Paley (2022), bilinearization and control, **IEEE TAC**: https://doi.org/10.1109/TAC.2021.3088802
- **[P7]** *On the Identifiability of Controlled World Models* (2026, preprint): https://arxiv.org/abs/2607.22430
- **[P8]** Davison & Ball (1972), *On the global controllability of perturbed controllable systems*, **IEEE TAC**; cần so sánh trực tiếp với P-margin trước novelty claim.
- **[P9]** MAKO và recursive Koopman adaptation (2025–2026): cần xác minh chính xác tình trạng xuất bản trước khi trích như journal publication.
- **Code P1:** https://github.com/KYMiao/Controllable-Koopman

---

## Appendix A. Bảng sửa toán v0.3 → v0.4

| Điểm trong v0.3 | Vấn đề | Cách sửa ở v0.4 |
|---|---|---|
| `χ=r_A/ρ_T<1 ⇒ S3 exact-match & certified` | **Sai** khi có `M(z)` khả nghịch tự do | Loại bỏ implication. Dùng `d_orient` (necessary) và `d★` với bounds singular values (pointwise feasibility diagnostic). |
| `χ≫1 ⇒ H3 vô dụng` | **Sai** do `M` có thể scale | Không dùng χ làm Go/No-Go. |
| Physical constant `G_x` ⇒ class A đủ | Không đúng ở lifted input do `JφG_x` | Phân biệt physical/lifted; đo cả hai. |
| “Input encoder giữ controllability” như phát hiện mới | Appendix G.5 **đã nói rõ** | Gán là baseline-known; thêm giả định `M∈GL(m)`, đặc biệt multi-input. |
| Gramian λ_min tăng ⇒ correction budget tăng | Thiếu mẫu số `κ_T` | Đo `ρ_T=√λ/κ` trực tiếp. |
| P-margin không rõ proof status | Chưa peer-reviewed, thiếu chi tiết compactness/continuity | Viết giả định + fixed-point proof sketch + giới hạn rõ. |
| S3 certificate ⇒ physical MPC ổn định | Không có suy luận | Tách surrogate SOC / bounded-input reachability / closed-loop performance. |
| H3 thay thế H2 | Không đúng ý định nghiên cứu | Tách H2–H3 độc lập, dùng chung P0–P2, gates riêng. |
| Online adaptation được giả định là đóng góp tức thì | Literature đã có online Koopman | Chuyển thành future extension; không claim novelty. |

### A.1. Bảng sửa từ v0.4 → v0.5 (giữ các chỉnh sửa v0.3 → v0.4 ở trên)

| Vấn đề v0.4 | Kết luận phản biện / sửa ở v0.5 |
|---|---|
| H3 thiếu tính lồi của MPC làm hard requirement | Tách QP-exact / virtual-input restricted QP / frozen-scheduling sequential QP / nonlinear MPC; đo latency, feasibility, convergence và physical-input violations. |
| “A unstable ⇒ ρ_T→0” | Chứng minh **sai tổng quát** bằng hệ scalar, đưa hệ hai mode cho counterexample suy giảm; đo horizon/conditioning thay vì suy diễn từ phổ A. |
| `M(z)∈GL(m)` không ép bằng loss | Dùng hard parameterization `M=I+E`, `||E||₂≤γ<1`; từ đó suy singular-value và inverse bounds. |
| H2 đòi nhiều action tại cùng một `x` | Bổ sung population identification bằng conditional covariance; nhấn mạnh finite-sample coverage và deterministic-action counterexample. |
| Counterfactual oracle chỉ cho evaluation | Giữ cho **observational Protocol A**; mở **Protocol B intervention training** riêng nếu thay đổi quyền truy cập, với matched-data controls. |
| Pendulum/CartPole dùng làm gate | Tách reproduction và stress-test; Pendulum không được bác H2/H3; thêm controlled input rotation, Unicycle, Planar Quadrotor, sau đó Acrobot. |
| Novelty H3 cần rà thêm | Đưa CDC 2024 Koopman bilinear controllability, SIAM 2025 convex nonlinear input MPC, Control-Coherent Koopman CDC 2024 vào đối chứng trực tiếp. |
| Các phần không bị phản biện | **Giữ nguyên** formal problem, S0–S3, derivation lifted `JφGₓ`, P-margin fixed-point proof, sửa `χ` và `d★`, độc lập H2/H3, chính sách train oracle. |

## Appendix B. Checklists trước khi viết code phương pháp mới

**Baseline reproduction:** đã lock commit, seeds, sampling, train window, solver, evaluation; so sánh theo paper Appendix G; đủ checkpoint để đo `A,B,ψ,C`.

**H2:** đã có same-state action diagnostics, action-coverage test và matched model-class ablations; không lẫn supervision privileged. Ghi rõ Protocol A observational (chính) và Protocol B controlled-intervention training (phụ); đo conditional action covariance/coverage và effect of finite data.

**H3:** đã fix lỗi χ, kiểm tra margin trên đúng output Gramian, chọn hard parameterization và bounds `M`, có proof review, phân biệt pointwise fit vs global smooth functions; **có protocol QP-exact / approximate QP / NMPC** và báo cáo solve-time/physical-input constraints.

**Novelty:** đọc CDC 2024 Koopman bilinear controllability, SIAM 2025 nonlinear input convex MPC và các certified LPV/bilinear Koopman khác; cập nhật thư mục reference trước paper submission.

## Appendix C. Hướng đồng bộ `EXPERIMENT.md` và `ROADMAP.md`

1. Đổi nhãn H2-only thành **shared P0–P2 (reproduction vs stress-test tách biệt) + H2 gate + H3 gate**.
2. Thêm metrics `d_orient`, `d★(σmin,σmax)`, `ρ_T`, lifted Jacobian và drift mismatch.
3. Xóa mọi điều kiện Go/No-Go dựa trên `χ<1` / `χ≫1`.
4. Giữ yêu cầu **chỉ chạy lại [P1] trong P0–P2**; dùng cùng checkpoint baseline đánh giá thêm toy rotation, Unicycle, Planar Quadrotor (cần train chính mô hình [P1] trên dữ liệu các hệ đó, **không** huấn luyện model nghiên cứu mới). Không reproduce baseline ngoài trước.
5. Bổ sung theory audit H3 (horizon/gauge margin, hard M), matched-capacity/data/compute ablation H2; tách Protocol A/B sau gate.
6. Không báo “paper-ready” nếu mới có supervised prediction MSE hoặc surrogate rank certificate. Bổ sung metrics solver type, actual MPC feasibility/timeouts/latency, QP-approximation gap và same physical-input bounds.

---

**Decision log:**
- **v0.2:** ưu tiên H2; hai branches là giả thuyết, không phải claim novelty.
- **v0.3:** thêm H3 cùng perturbation margin; vô tình ghi H3 thay H2; metric χ có suy luận feasibility sai.
- **v0.4:** khôi phục **H2 và H3 độc lập**, chỉnh chặt định nghĩa P-margin và cơ chế đầu vào, thay χ bằng orientation/conditioned factorization diagnostics; baseline reproduction vẫn là bước đầu tiên.
- **v0.5:** không đổi hai-track decision; bổ sung control-tractability là hard requirement; sửa identifiability H2 theo population vs finite-sample; giới hạn phạm vi oracle training rõ; phân biệt reproduction benchmarks và hypothesis stress-tests; làm rõ unstability không suy ra margin co; đưa hard-parameterized `M` và literature CDC 2024/SIAM 2025 thành đối chứng.
