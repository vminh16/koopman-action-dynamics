# RESEARCH REQUIREMENTS DOCUMENT (RRD)
## Expressive and Controllability-Aware Koopman Latent Dynamics
### Hai hướng nghiên cứu độc lập: H2 — Dynamics-Factorized Learning; H3 — Controllability-Preserving State-Dependent Input Maps

**Phiên bản:** 0.4 — 12/11/2026  
**Trạng thái:** Research proposal; chưa xác nhận novelty, chưa triển khai H2/H3.  
**Baseline duy nhất cần tái lập ban đầu:** Miao et al., *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026 [P1].  
**Phạm vi giai đoạn đầu:** continuous-time, full-state observation, nominal dynamics, simulation trajectories; ưu tiên Pendulum, CartPole.  
**Quyết định:** **Giữ H2 và H3 là hai research tracks độc lập**, cùng sử dụng dữ liệu, baseline và các phép đo chẩn đoán. Không lấy H3 thay thế H2; không ép H2 thành module con của H3.  
**Tài liệu thực thi:** `EXPERIMENT.md` và `ROADMAP.md` (v0.2 là bản cũ; cần đồng bộ theo mục 11 và phụ lục C trước khi dùng làm execution protocol).

> **Research position.** Chúng ta không lấy “thêm hai nhánh”, “thêm input encoder”, “cho B phụ thuộc trạng thái”, hoặc “thêm một regularizer” làm novelty mặc định. Trước hết phải đo được *failure mode* của baseline, hiểu điều kiện nhận dạng và controllability, rồi mới xác định phương pháp nào xứng đáng triển khai. Prediction fidelity, action-response fidelity và controllability là **ba yêu cầu liên hệ nhưng không đồng nhất**.

**Nhãn mức độ tin cậy trong tài liệu:** **[B]** kết quả đã kiểm tra trên paper nguồn; **[M]** suy ra từ mô hình/toán học với giả định nêu rõ; **[H]** giả thuyết chưa kiểm chứng; **[E]** thiết kế thực nghiệm. Số liệu chưa chạy trên mã nguồn riêng không được trình bày là kết quả của nhóm.

---

## 0. Executive research decision

| Chủ đề | Quyết định v0.4 |
|---|---|
| Bài toán lớn | Học một latent dynamics model có ích đồng thời cho dự đoán phản thực theo hành động và thiết kế điều khiển. |
| Lõi chung | Phân tích điểm yếu fixed-input Koopman qua action-response và closed-loop MPC, không chỉ MSE. |
| **H2 — học biểu diễn** | Liệu phân rã *zero-input drift* và *action response* bằng inductive bias/learning protocol riêng có cải thiện identification, prediction và control utility so với mô hình thống nhất **cùng năng lực biểu diễn**? |
| **H3 — bảo đảm mô hình** | Liệu có lớp input map phụ thuộc trạng thái đủ linh hoạt và một điều kiện **controllability của surrogate** đủ chặt, áp đặt được khi học, không trả giá quá lớn về fidelity và MPC? |
| Phụ thuộc H2/H3 | **Không phụ thuộc logic.** H2 có thể thành công mà không có chứng nhận H3; H3 có thể thành công mà không có hai nhánh H2. Chỉ nghiên cứu tổ hợp H2+H3 sau hai thí nghiệm độc lập. |
| Online adaptation | Giữ là động cơ ứng dụng dài hạn và hướng mở rộng sau, không hứa chứng minh cùng lúc hai mục tiêu chính. |
| Bước tiếp theo | P0 audit paper/repo → P1 reproduce paper → P2 đo action-response/latent/controllability/candidate regret → quyết định H2 và H3 **riêng biệt**. |

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

### 4.2. Identifiability và giới hạn do dữ liệu

Tại **cùng một** `x`, từ quan sát đạo hàm không nhiễu `y_i=f_0(x)+G_x(x)u_i`, lập:

\[
 X=\begin{bmatrix}1&u_1^\top\\\vdots&\vdots\\1&u_K^\top\end{bmatrix}.
\]

Nếu `rank(X)=m+1`, các hệ số `f0(x)` và `G_x(x)` được nhận dạng duy nhất tại **trạng thái ấy** bằng hồi quy tuyến tính. Nếu thiếu rank, tồn tại nhiều phân rã tương thích dữ liệu. Với trajectory thực, thường không thể thực hiện nhiều input tại đúng một state; cần assumptions về smoothness, coverage hoặc controlled resets **chỉ làm evaluation**. Đây **không** là chứng minh nhận dạng toàn cục của neural representation.

Dữ liệu random input của paper giúp giảm confounding do policy tại thời điểm thu thập, nhưng không bảo đảm coverage ở trạng thái mà MPC đến hoặc ở các chuỗi lệnh dài có cấu trúc.

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

**H2-3 (capacity-matched monolithic control-affine):** Sử dụng shared trunk và hai output heads chỉ như cách biểu diễn `F,G`, hoặc một `h_ξ(z,u)` tổng quát với ràng buộc `h(z,0)`; phải xác định so sánh *cùng model class* và *cùng tổng số tham số*. Nếu H2-2 chỉ thắng một monolithic MLP yếu hơn, chưa đủ ủng hộ factorization.

**Fusion:** không mặc định attention. Phép hợp nhất mặc định xuất phát từ `F+Gu`. Khi muốn dùng attention, trước hết chứng minh ablation rằng coupling learned attention tốt hơn structured composition với cùng số tham số, không mất consistency.

### 5.3. Learning signals có nghĩa vật lý (candidates, chưa chốt objective)

- Multi-step trajectory loss dưới chuỗi action đã thực hiện.
- Zero/reference-action consistency khi **dữ liệu hợp lệ** có reference input; không giả lập nhãn không quan sát được như ground truth.
- Action-pair consistency: nếu dataset/experimental protocol có repeated/sufficiently close states với varying actions, mô hình nên phù hợp với sai khác của transitions. Không dùng simulator counterfactual outcome để train trừ khi rõ ràng đổi problem setup thành privileged training.
- Optional manifold loss để hạn chế off-manifold rollout, sau khi đo `e_M`.

Objective mới chỉ được chọn **sau baseline diagnosis**; H2 không có một “loss tùy ý cộng thêm” được xác nhận trước.

### 5.4. Điều kiện đánh giá độc lập H2

H2 **GO** khi baseline có meaningful action-response failure và H2-2 tốt hơn H2-1 **sau khi matched expressiveness và compute** trên counterfactual response hoặc MPC, qua nhiều seeds.  
H2 **NO-GO** nếu H2-1 / Appendix G.5 đã khắc phục hoàn toàn, nếu gains do nhiều tham số, hoặc nếu việc tách drift/action không nhận dạng được với data thực tế.

**Đóng góp kỳ vọng (chưa xác nhận):** một learning mechanism biểu diễn/cập nhật dynamics có phân tách nhưng duy trì consistency; không tuyên bố *dual branch* là mới.

---

## 6. Track H3 — Controllability-Preserving State-Dependent Input Maps (độc lập)

### 6.1. RQ-H3 và điều đã có trong literature

**RQ-H3:** Có lớp state-dependent input maps đổi được **hướng** action effect trong lifted state mà vẫn có điều kiện **SOC của surrogate**, chứng nhận được khi học và đủ ít bảo thủ để cải thiện action-response, MPC không?

H3 không cần phải có neural two-branch theo H2: giữ `F(z)=Az` và tập trung vào `B(z)`. Paper đã xét `B M(z)` ở Appendix G.5; controllability under bounded perturbation là một hướng lý thuyết cổ điển. **Novelty (nếu có) là thiết kế lớp học + chứng nhận hữu ích + bằng chứng fidelity/control**, không phải phép biến đổi input hoặc riêng mệnh đề perturbation.

### 6.2. Họ mô hình và ký hiệu

\[
\begin{aligned}
\text{S0 (paper):}\quad &\dot z=Az+Bu,\\
\text{S1 (Appendix G.5 / class A):}\quad &\dot z=Az+B M(z)u,\\
\text{S2 (unconstrained input-affine):}\quad &\dot z=Az+[B+\Delta B(z)]u,\\
\text{S3 (certified candidate):}\quad &\dot z=Az+[B+\Delta B(z)]M(z)u.
\end{aligned}
\]

`A∈R^{N×N}`, `B∈R^{N×m}`, `C∈R^{n×N}`; `ΔB(z)∈R^{N×m}`, `M(z)∈R^{m×m}`. Trong S3 yêu cầu `M(z)` khả nghịch và `ΔB` bounded/Lipschitz theo điều kiện sau. Đây là **candidate class**, không phải thuật toán đã chứng minh ưu thế.

### 6.3. Lớp A — đổi biến đầu vào, bảo toàn reachability không ràng buộc [M]

Giả sử `M:R^N→GL(m)` liên tục. Với physical input `u`, đặt `v(t)=M(z(t))u(t)`; hệ đổi thành:

\[\dot z=Az+Bv.\]

Với bất kỳ quỹ đạo LTI khả dĩ sinh bởi input liên tục `v(·)` trên `[0,T]`, đặt `u(t)=M(z_v(t))^{-1}v(t)`, khi đó cùng quỹ đạo được thực hiện bởi S1; tính liên tục và khả nghịch trên quỹ đạo compact đảm bảo `u(·)` liên tục. Vì vậy **reachable outputs với physical inputs không bị chặn trùng với LTI**. Không suy ra điều này khi `u∈U` có bounds.

Nếu có uniform bound `sup_z ||M(z)^{-1}||₂≤K_M`, ta có:

\[\|u\|_{L^2}\le K_M\|v\|_{L^2}.\]

**Hạn chế expressiveness:** `Im(BM(z))=Im(B)` ở mọi `z`; chỉ thay tỷ lệ/trộn input channels, **không thay không gian cột** trong lifted state. Đây chính là ranh giới để H3 tìm kiếm bổ sung.

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

### 6.5. S3 kết hợp M và ΔB

\[
\boxed{\dot z=Az+[B+\Delta B(z)]M(z)u.}
\]

Khi `M(z)∈GL(m)` liên tục, đổi `v=M(z)u` đưa S3 về dạng P-margin. Nếu `sup_z||M(z)^{-1}||≤K_M`, bound **trên physical control energy** trở thành:

\[
\boxed{\|u\|_{L^2}\le
K_M\frac{\|y^\star-Ce^{AT}z_0\|}{\sqrt{\lambda_T}-\delta\kappa_T}.}
\]

Nếu không ràng buộc `M^{-1}`, một phép scale làm effective input rất lớn vẫn có thể cần physical effort không thực dụng (hoặc ngược lại). Nếu có `u` bounded, tập `v=M(z)U` phụ thuộc state và bài toán MPC không còn linear-convex tiêu chuẩn. **Chứng nhận reachability không tự động chứng nhận constrained controllability**.

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

### 6.7. Đo ngân sách Gramian đúng cách

Một lỗi diễn giải cần loại bỏ: `λ_min(W_y)` tăng **không tự động** khiến `ρ_T` tăng vì `κ_T` cũng có thể thay đổi. Cần đo **trực tiếp** `ρ_T` theo latent size `N`, horizon `T`, learning seed và Gramian-loss weight. Matrix scaling/reparameterization có thể ảnh hưởng các chuẩn, đặc biệt nếu đổi tọa độ lifted mà không cố định gauge; mọi so sánh margin phải ghi rõ tọa độ `z`, `C`, chuẩn và cách normalize.

### 6.8. Điều không được tuyên bố

- P-margin là **bảo đảm của surrogate** với `ΔB` bounded globally, không là bảo đảm của physical plant.
- `SOC` với unrestricted input không đồng nghĩa MPC tối ưu/lồi, input-bounded reachability, stability, safety.
- `d_orient<ρ_T` chỉ **cần** cho exact S3; điều kiện không xác nhận model tồn tại.
- Không gọi class A, generic LPV/bilinear Koopman, bounded-perturbation robustness hay state-dependent encoder là novelty độc lập.

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
| **RQ-H2-2** | Lợi ích có bền với action coverage khác nhau? | H2 | Không có gain trong điều kiện thu thập khả thi. |
| **RQ-H3-1** | S1 bảo toàn SOC đến đâu; cần `M` có điều kiện gì? | Theory, trước S3 | Chỉ có effective-input claim nhưng physical bounds vô dụng. |
| **RQ-H3-2** | P-margin có chứng minh đầy đủ và hữu ích không? | Theory + baseline diagnostics | Bound quá nhỏ; d★ luôn vượt bound. |
| **RQ-H3-3** | S3 có giảm error so với S0/S1/S2 và cải thiện MPC không? | Prototype H3 | S3 chỉ thêm tham số/solver cost. |
| **RQ-Future** | Adaptation có giữ được chứng nhận khi dynamics shift? | Sau hai tracks | Certificate không cập nhật được khi model học online. |

**Hypotheses trước khi chạy:**

- **H0:** Fixed-`B` + closed-loop re-encoding đã đủ tốt; không cần H2/H3 trên target tasks.
- **H2:** Factorized learning tăng out-of-distribution action-response fidelity sau matched-budget mà không mất control utility.
- **H3-A:** Lớp A đủ trong các trường hợp lifted input **không đổi hướng** (không suy từ physical `G_x` hằng).
- **H3-B:** Có ít nhất một target task mà `d★` dưới usable `ρ_T` trên phần miền test liên quan, cho phép S3 vừa certified vừa cải thiện response fidelity.
- **H3-C:** Gramian-weight/δ có đường đánh đổi hữu ích; tăng Gramian weight không bảo đảm tăng margin.
- **H-control:** Những improvement đáng kể về dynamics phải được kiểm chứng lại bằng candidate regret hoặc MPC; không suy tự động.

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

**Novelty check còn mở:** phải rà thêm certified LPV/bilinear control-affine Koopman, neural parametrizations đảm bảo output reachability, input-constrained certificates, và các công trình của chính nhóm [P1] sau ICLR 2026. “Chưa tìm thấy” ≠ “chưa ai làm”.

---

## 10. Dataset, đo đạc và phân tầng thực nghiệm

### 10.1. Reproduction: chỉ chạy lại paper ICLR 2026

Paper [P1, Appendix G] sinh full-state trajectories dưới random controls, khoảng 250 time steps mỗi rollout, training prediction horizon 15; ví dụ README của official repo có `--window_len 50`, **không** được coi hai protocol giống nhau khi chưa audit. Không bắt buộc huấn luyện/reproduce DKO, MAKO, DWM, Iso-Dream trong P0–P2.

| Dataset/hệ | Vai trò sớm | Lưu ý |
|---|---|---|
| **Pendulum** | Reproduction thấp rủi ro, test Appendix G.5, physical action map đơn giản | Lifted input có thể state-dependent. |
| **CartPole** | Benchmark sớm để đo influence direction và closed-loop control | Ground-truth vector field/oracle chỉ dùng evaluation. |
| **Mountain Car** | Bổ sung constant physical-input case | Không suy ra class A exact trong latent. |
| **Reacher** | Mở rộng multi-state/multi-input sau khi protocol ổn định | Kiểm tra state definition và MPC solver. |
| **Synthetic 1–2D controllable systems** | Tách ảnh hưởng của input-direction rotation, gain, drift | **Không thay thế** reproduce original paper; là diagnostic. |

### 10.2. Shared diagnostics P2 — không train model mới

1. `MSE_1`, `MSE_H` trên held-out trajectories, open-loop và periodic re-encode.
2. Same-state action-effect error ở physical space `||[(ẋ_model(x,u)-ẋ_model(x,u_ref))-(ẋ_true(x,u)-ẋ_true(x,u_ref))]||`.
3. Lifted input-map mismatch `||B−JφG_x||` và drift mismatch `||Aφ−Jφ f0||` (nếu oracle field có sẵn).
4. Gramian `λ_min(W_y)`, condition number, `κ_T`, `ρ_T`, dependency theo `N,T`, seeds.
5. `r_A`, `d_orient`, `d★` với singular-value intervals định trước. **Không sử dụng χ làm gate.**
6. MPC tracking/success, control effort, compute/solver time, and candidate regret trên candidate pool cố định (oracle **evaluation only**).
7. Coverage map: diagnostics trên training-like states **và** states do frozen-MPC policy ghé; báo cáo distributions riêng.

### 10.3. Track-specific experiments — chỉ sau gates riêng

**H2:** fixed-input baseline → input-affine one-branch → factorized two-branch → same-class/capacity monolithic control; action coverage, window length, random seeds, compute matching; later isolated actuation shift.

**H3:** S0 → S1 (Appendix G.5) → S2 unconstrained → S3 certified; vary `δ`, `M` singular-value bounds, Gramian weight, `T`; compare exact certificate pass/fail, prediction, response error, MPC latency, physical-input bounds and closed-loop success. Jacobian-based input `Jφ Ĝ` dùng làm **oracle-structure comparator**, không tự xem upper bound lý thuyết khi chưa đạt đủ optimization quality.

### 10.4. Nguyên tắc tính khoa học

- Chốt seeds/hyperparameters/stop criteria trước khi nhìn held-out results; report mean + uncertainty qua nhiều runs.
- Không trộn cùng một trajectory vào train/test; nếu generalization qua chế độ vật lý thì split theo chế độ.
- Counterfactual reset và true `f,G_x` từ simulator chỉ dùng cho diagnosis/evaluation, không cho training nếu không tuyên bố rõ protocol khác.
- So sánh với cùng `Δt`, solver tolerances, MPC horizon, input constraints và action candidate budget.

---

## 11. Roadmap và Go/No-Go độc lập

| Phase | Việc chính | Deliverable | Điều kiện chuyển pha |
|---|---|---|---|
| **P0 — Audit** | Lock code commit; paper-vs-README configurations; dataset generator; MPC scripts | Reproduction manifest | Source/evaluation protocol nhất quán hoặc khác biệt được ghi nhận. |
| **P1 — Reproduce** | Train/eval original model trên Pendulum, CartPole | Checkpoints, logs, metric table | Sai khác với paper nằm trong tolerance pre-registered; nếu không, giải thích và sửa protocol. |
| **P2 — Diagnose** | Shared diagnostics §10.2, toy direction-rotation response | Failure-mode report | Biết rõ sai số do model class/coverage/solver ở đâu; không phải “cứ có MSE là gap”. |
| **GATE H2** | Counterfactual mismatch có ý nghĩa + fair comparison design | H2 go/no-go | Nếu baseline hoặc S1 đủ tốt, không ép two-branch. |
| **GATE H3** | `ρ_T`, d_orient/d★, proof audit và literature | H3 go/no-go | Có ít nhất một test setting có meaningful certified correction window và động cơ control. |
| **P3-H2** | H2 prototype trên cùng data/simulator | Independent H2 report | Gain over matched model-class ablation. |
| **P3-H3** | Hoàn thiện theorem; S1 vs S2 vs S3 | Independent H3 report | Certificate đúng + usable + model/compute ablations. |
| **P4** | Robustness, significance, latent-size scaling, MPC | Paper-evidence report | Phù hợp research claims. |
| **P5 optional** | H2+H3 combination hoặc online adaptation | New scoped RRD | Chỉ làm khi track tương ứng đã có kết quả độc lập. |

**Pre-registration gợi ý:** ≥5 seeds cho kết quả học, các mức dữ liệu như paper, time budget rõ ở `ROADMAP.md`. Các ngưỡng hiệu quả (MSE/control gain/bound utility) phải chốt trước khi chạy. Không chốt con số tuỳ ý trong RRD nếu chưa biết noise/variance baseline.

**Nguyên tắc No-Go:** một kết quả âm tính là chấp nhận được; không tự ý đổi benchmark, metric hoặc giới hạn `M` sau khi nhìn kết quả để “cứu” H2/H3.

---

## 12. Đóng góp khoa học kỳ vọng và tiêu chuẩn thành paper

**Track H2 — Có thể viết paper khi** có một learning mechanism phân rã dynamics có thể nhận dạng/kiểm chứng, vượt các alternative cùng model class/parameter budget trên counterfactual action fidelity và có ý nghĩa trong closed-loop MPC; không chỉ hai heads hoặc add-loss.

**Track H3 — Có thể viết paper khi** tồn tại certified model class/training scheme có chứng nhận đúng, hữu ích định lượng (không chỉ tồn tại hình thức), thể hiện đường đánh đổi với action response và chi phí MPC. Nếu P-margin quá bảo thủ, có thể chuyển sang trajectory/local certificate nhưng phải **đổi lại scope/định lý**, không giữ global claim.

**Paper tổng hợp H2+H3** chỉ có ý nghĩa nếu kiểm chứng được synergy rõ rệt: representation được học tốt hơn mà chứng nhận vẫn hữu ích. Việc kết hợp đơn thuần hai modules không cấu thành đóng góp.

**Chưa xác nhận:** novelty so với toàn literature, khả năng tái lập reproduction, chất lượng định lý mới sau phản biện, H2/H3 có cải thiện hay không. Không dự đoán venue acceptance trước khi có kết quả.

---

## 13. References và provenance

**Nguồn đã đối chiếu với chính paper/web trong lần cập nhật:**

- **[P1]** Miao, K., Wang, H., Ding, X., Gatsis, K., Krause, A., Papachristodoulou, A. (2026). *Learning Koopman Representations with Controllability Guarantees*. **ICLR 2026**, §3–4, Appendix B–G, Table 10. https://proceedings.iclr.cc/paper_files/paper/2026/file/e3781fb5d2c17ea7283cc2a543793bdb-Paper-Conference.pdf
- **[P4]** Iacob, L. C., Tóth, R., Schoukens, M. (2024). *Koopman form of nonlinear systems with inputs*. **Automatica 162**, 111525. https://doi.org/10.1016/j.automatica.2024.111525
- **[P6]** *Conformal Online Learning of Deep Koopman Linear Embeddings*. **NeurIPS 2025**. https://proceedings.neurips.cc/paper_files/paper/2025/hash/784fd5a46dfe303e5b51c8621b84cf3f-Abstract-Conference.html

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

## Appendix B. Checklists trước khi viết code phương pháp mới

**Baseline reproduction:** đã lock commit, seeds, sampling, train window, solver, evaluation; so sánh theo paper Appendix G; đủ checkpoint để đo `A,B,ψ,C`.

**H2:** đã có same-state action diagnostics, action-coverage test và matched model-class ablations; không lẫn supervision privileged.

**H3:** đã fix lỗi χ, kiểm tra margin trên đúng output Gramian, chọn bounds M, có proof review, phân biệt pointwise fit vs global smooth functions, có input-constrained MPC protocol.

**Novelty:** kiểm tra latest related work về certified LPV/bilinear Koopman; cập nhật thư mục reference trước paper submission.

## Appendix C. Hướng đồng bộ `EXPERIMENT.md` và `ROADMAP.md`

1. Đổi nhãn H2-only thành **shared P0–P2 + H2 gate + H3 gate**.
2. Thêm metrics `d_orient`, `d★(σmin,σmax)`, `ρ_T`, lifted Jacobian và drift mismatch.
3. Xóa mọi điều kiện Go/No-Go dựa trên `χ<1` / `χ≫1`.
4. Giữ yêu cầu **chỉ chạy lại [P1] trong P0–P2**; không reproduce baseline ngoài trước.
5. Bổ sung theory audit H3 và matched-capacity/data/compute ablation H2 sau gate.
6. Không báo “paper-ready” nếu mới có supervised prediction MSE hoặc surrogate rank certificate.

---

**Decision log:**
- **v0.2:** ưu tiên H2; hai branches là giả thuyết, không phải claim novelty.
- **v0.3:** thêm H3 cùng perturbation margin; vô tình ghi H3 thay H2; metric χ có suy luận feasibility sai.
- **v0.4:** khôi phục **H2 và H3 độc lập**, chỉnh chặt định nghĩa P-margin và cơ chế đầu vào, thay χ bằng orientation/conditioned factorization diagnostics; baseline reproduction vẫn là bước đầu tiên.
