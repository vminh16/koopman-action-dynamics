# RRD — Adaptive Learning of Action-Dependent Latent Dynamics

**Version:** 0.1 (research framing; chưa chốt phương pháp)  
**Ngày:** 12/11/2026  
**Research domain:** System Identification · Structured Latent Dynamics · Koopman Learning · Adaptive Control · Model-Based Planning  
**Baseline chính:** *Learning Koopman Representations with Controllability Guarantees* (ICLR 2026), Miao et al.  
**Trạng thái:** **Research gap chưa được xác nhận là mới**; cần tái lập, đối chiếu các phương pháp liên quan, rồi mới thiết kế phương pháp.

> **Nguyên tắc phạm vi:** Tài liệu này chỉ đặc tả **hướng nghiên cứu đã thống nhất trước đó**: học và thích ứng trực tiếp cơ chế tác động của hành động trong latent dynamics. Ý tưởng kiến trúc hai nhánh prediction/controllability là một giả thuyết **riêng**, chưa được đưa vào giải pháp của RRD này.

## 1. Bối cảnh & bài toán lớn

Trong control và model-based RL, một dynamics model dự đoán trạng thái tương lai dưới chuỗi hành động để phục vụ planning. Với hệ phi tuyến không biết trước phương trình, dữ liệu tương tác gồm trạng thái/quan sát và hành động. Koopman learning tìm biểu diễn nâng chiều để xấp xỉ động lực học bằng các toán tử đơn giản hơn, giúp dự đoán nhiều bước và thiết kế MPC.

Thách thức của ứng dụng thực tế không chỉ là **học mô hình dự đoán tốt trong điều kiện huấn luyện**, mà còn là mô hình hóa đúng **cơ chế tác động của hành động** và **cập nhật được cơ chế đó khi động lực học thay đổi**. Ví dụ actuator gain thay đổi, tải trọng hoặc ma sát thay đổi, hay hiệu quả điều khiển phụ thuộc trạng thái.

Đây là nghiên cứu về **dynamics shift**; không đồng nhất với visual corruption, camera shift hoặc reward shift.

## 2. Baseline ICLR 2026: đã làm gì, chưa làm gì?

Mô hình cơ sở:

\[
 z=\phi_\theta(x)=\begin{bmatrix}x\\\psi_\theta(x)\end{bmatrix},\qquad
 \dot z=A_\theta z+B_\theta u,\qquad \hat x=Cz.
\]

- **Mạng học:** observable encoder \(\psi_\theta\), các tham số xác định toán tử \(A_\theta,B_\theta\), và phép biến đổi khả nghịch \(P_\theta\). ODE solver và MPC **không** phải mạng policy cần huấn luyện.
- **Tính chất đặc trưng:** tham số hóa dạng canonical để đảm bảo tính controllability của mô hình surrogate theo điều kiện định lý; Gramian regularization cải thiện conditioning; multi-step prediction loss huấn luyện end-to-end.
- **Đầu vào:** trạng thái vật lý \(x\) và chuỗi hành động \(u\), không phải raw pixels; latent là **lifted representation**, không nhất thiết là nén chiều.
- **Đánh giá:** prediction và MPC trên các hệ phi tuyến danh định; không đặt online identification dưới nhiều chế độ dynamics làm nhiệm vụ chính.

**Điểm quan trọng trong Appendix G.5:** Tác giả thử input encoder \(\hat u=g_\theta(\hat x)\odot u\), tăng khả năng biểu diễn action effect phụ thuộc trạng thái và cải thiện prediction, nhưng khiến bài toán MPC không còn giữ nguyên cấu trúc tuyến tính lồi theo tín hiệu điều khiển vật lý \(u\). Đây là **trade-off đã được chỉ ra**, không phải ý tưởng chưa ai thử.

**Bảo đảm cần diễn giải đúng:** controllability trên **surrogate đã học** không tự động bảo đảm môi trường thật luôn controllable hoặc MPC ổn định dưới model mismatch, actuator constraints hay dynamics shift. Mô hình không thể vừa trung thực vừa giả định controllable nếu hệ thật đã mất controllability.

## 3. Problem Statement

Giả sử hệ thực có dynamics biến thiên theo chế độ chưa biết:

\[
\dot x(t)=f_{\eta_t}(x(t),u(t)),\quad x\in\mathbb R^n,\;u\in\mathcal U\subseteq\mathbb R^m.
\]

Dữ liệu huấn luyện được thu từ một số chế độ \(\eta\); tại deployment, \(\eta_t\) có thể thay đổi. Mô hình chỉ quan sát chuỗi chuyển trạng thái và hành động **đã thực hiện**, không được biết nhãn \(\eta_t\) và không được truy cập kết quả của các hành động counterfactual ở test time.

**Bài toán:** *Có thể học một biểu diễn latent dynamics đủ biểu cảm đối với state-dependent action effects, đồng thời nhận dạng/cập nhật cơ chế tác động của hành động từ ít transition online, để cải thiện prediction và closed-loop control sau dynamics shift mà không phải học lại toàn bộ mô hình?*

Hai yêu cầu độc lập về tiêu chí đánh giá nhưng liên quan về cơ chế là:

1. **Faithful dynamics:** mô hình phản ánh đúng chuyển trạng thái và sự khác biệt hệ quả giữa các hành động.
2. **Adaptable dynamics:** mô hình thay đổi được cơ chế tác động của hành động khi có dữ liệu mới, với sample/compute budget hữu hạn.

**Không mặc định lời giải:** Dạng control-affine \(\dot z=f(z)+G(z,\eta)u\), bilinear Koopman, meta-learning hay online operator updates chỉ là các **ứng viên**, chưa là đóng góp của nghiên cứu.

## 4. Giả định giai đoạn đầu

- **Quan sát đầy đủ trạng thái** để cô lập dynamics learning khỏi lỗi state estimation. Mở rộng partial observation/raw pixels chỉ sau khi có bằng chứng.
- **Dynamics shift có kiểm soát:** piecewise-stationary hoặc thay đổi đủ chậm để tồn tại cửa sổ nhận dạng. Phải nêu rõ khi nào giả định này không đúng.
- **Hành động bị chặn**; chính sách thu thập dữ liệu có độ đa dạng tối thiểu để kiểm tra nhận dạng (persistent excitation). Không khẳng định có thể học được mọi chiều action từ lịch sử thiếu excitation.
- **Không dùng oracle tại inference:** nhãn tham số môi trường hoặc nhiều rollout đối chứng từ cùng trạng thái chỉ dùng để chấm điểm/diagnosis, không đưa vào model update.
- **Tách hai họ shift:** (i) chế độ vẫn điều khiển được; (ii) suy giảm/mất chiều tác động vật lý. Không đặt bảo đảm controllability sai cho họ (ii).

## 5. Research Questions & Hypotheses

| ID | Research Question | Giả thuyết cần bác bỏ/kiểm chứng |
|---|---|---|
| RQ1 | Lớp surrogate Koopman tuyến tính theo input sai lệch ở đâu khi action effect phụ thuộc trạng thái? | H1: Có trường hợp model-class mismatch mà thêm dữ liệu nhưng giữ nguyên lớp biểu diễn không khắc phục hết. |
| RQ2 | Cấu trúc nào giúp phân biệt thay đổi action effect với autonomous dynamics và nhiễu quan sát? | H2: Cấu trúc học tác động điều khiển phù hợp giúp tăng counterfactual action-response fidelity. |
| RQ3 | Khi dynamics shift, có thể nhận dạng/update thành phần liên quan bằng ít transition hơn fine-tuning rộng hay không? | H3: Cập nhật có cấu trúc đạt recovery nhanh hơn baseline online Koopman khi được đối chứng cùng compute/data. |
| RQ4 | Cải thiện prediction và action-response có dẫn đến cải thiện điều khiển thực tế không? | H4: Có cải thiện closed-loop performance trên nhiều tác vụ/shift, không chỉ giảm on-policy MSE. |

**Lưu ý novelty:** Đây là giả thuyết, **không phải kết quả, đóng góp đã đạt hoặc khẳng định chưa có công trình khác**.

## 6. Literature & novelty boundary

| Nhóm | Công trình | Tác động đến novelty |
|---|---|---|
| Controllability-aware representation | **Miao et al., ICLR 2026** | Đã có canonical Koopman + Gramian + MPC. Không được trình bày Gramian, controllability-aware loss hoặc neural ODE là mới. |
| Online Koopman | **MAKO, Automatica 2026**; **COLoKe, NeurIPS 2025** | Đã học/điều chỉnh Koopman online; không thể chỉ cộng thêm online gradient/RLS và tuyên bố mới. |
| Structured stable embeddings | **Learning Stable Koopman Embeddings, Automatica 2026** | Stability/stabilizability parameterization cũng đã được nghiên cứu. |
| Action/world effect decomposition | **DWM, arXiv 2026** | Đã tách tín hiệu action-driven và action-invariant; không thể coi tách nhánh theo nguồn dynamics là mới. |
| Factorized control representations | **CausalDreamer, arXiv 2026** | Đã chia controllable/uncontrollable và reward relevance; cần phân biệt *action-sensitive features* với bảo đảm controllability Kalman. |
| Bilinear state-dependent controls | **K-PWM, arXiv 11/2026**; nghiên cứu Koopman bilinear khác | Bilinear latent dynamics, partial observation và efficient planning đã có tiền lệ; phải đối chiếu trước khi đề xuất. |

**Điểm nghi vấn hợp lệ cần kiểm định**, chưa xác nhận là research gap toàn ngành: liệu có cơ chế học và cập nhật action-dependent latent operators vừa **đúng sự khác biệt giữa các hành động**, vừa **giữ tính sử dụng được cho control**, vừa **sample-efficient dưới dynamics shift**, vượt các mô hình online/structured hiện tại không?

## 7. Thí nghiệm: ưu tiên chứng minh failure mode

### Phase A — Reproduce & establish controls

1. Reproduce baseline ICLR 2026 ít nhất Pendulum và Reacher/CartPole; ghi đúng seed, data fraction, trajectory length, sampling rate, control limits và MPC solver.
2. Huấn luyện một **unstructured dynamics predictor** có kích thước và training budget đối chứng; kiểm tra model-class limitation.
3. Xây harness thay đổi dynamics độc lập với huấn luyện; xác nhận trạng thái, actuator và quan sát được định nghĩa nhất quán.

### Phase B — Isolate model errors

Mỗi shift phải có *training regime*, *test regime*, *held-out regimes* và *recovery window*.

| Shift | Câu hỏi kiểm chứng | Nhóm kiểm soát |
|---|---|---|
| Actuator gain / saturation | Bản đồ action-response có thay đổi được nhận dạng từ ít transition? | Shift vẫn controllable và trường hợp hỏng hoàn toàn tách riêng |
| State-dependent actuator effectiveness | Fixed-\(B\) surrogate có mismatch có hệ thống? | So sánh với biến thể input encoder Appendix G.5 |
| Actuator coupling (multi-input) | Mô hình có nhận dạng các ảnh hưởng chéo? | Đối chứng mức độ excitation theo từng input |
| Mass/friction change | Mismatch nằm ở action effect hay autonomous evolution? | Tránh quy mọi shift vào \(B\) |
| Observation noise | Không nhầm nhiễu thành shift? | Giữ ground-truth state chỉ cho diagnosis |

Trước khi đề xuất phương pháp mới: xác định lỗi nào thuộc **representation**, **identification**, **optimization** hay **planning**. Chỉ chọn 1 failure mode trung tâm.

### Phase C — Established adaptation baselines

- Frozen ICLR 2026 Koopman.
- Full fine-tuning trên transition gần nhất.
- RLS/online least squares cho các thành phần phù hợp.
- MAKO và COLoKe (khi code/protocol có thể đối sánh).
- Biến thể input encoder từ Appendix G.5.
- Bilinear/state-dependent dynamics baseline có cùng dữ liệu và compute.

Không lấy một bộ dữ liệu hoặc một planner khác làm kết luận trực tiếp nếu backbone, training protocol và action constraints không tương thích.

### Phase D — Only then method development

Chỉ thiết kế learning objective/parameterization/identification algorithm mới sau khi baseline đã thất bại trong một failure mode tái lập được. Không thêm gate/router làm contribution chính. Mọi đề xuất phải có ablation về representation, training, adaptation, control solver.

## 8. Metrics & validation

| Nhóm | Metrics | Chú ý |
|---|---|---|
| Dự đoán | One-/multi-step physical-state MSE | Held-out initial states và held-out regimes |
| Action effect | Finite-difference action response error; action ranking | Counterfactual ground truth **chỉ dùng evaluation**, cần matched states |
| Online adaptation | Recovery curves theo số transition; forgetting khi shift quay lại | Tách adaptation sample vs computation |
| Control | Success rate, tracking cost, violations, control energy | Nhiều seeds, cùng MPC horizon và action bounds |
| Cấu trúc | Rank/Gramian spectrum, conditioning | Đánh giá mô hình và hệ thật riêng; không đồng nhất |
| Efficiency | Trainable parameters, FLOPs, wall-clock latency, memory | Cùng hardware và implementation protocol |

Yêu cầu báo cáo confidence intervals, distribution qua seeds và negative results; không chỉ một best run. Chi phí MPC phải tính vào tổng chi phí suy luận.

## 9. Go / No-Go

**Go** nếu: (i) baseline gốc được tái lập; (ii) xuất hiện model-class/online-identification failure mode rõ ràng; (iii) các đối chứng online và nonlinear input hiện có còn thiếu sót trong điều kiện matched budget; (iv) một sửa đổi vào lõi dynamics mang lại lợi ích ổn định về control hoặc prediction + identification, có ablations.

**No-Go / pivot** nếu: (i) failure mode chỉ do thiếu training data hoặc solver; (ii) RLS/MAKO/input encoder xử lý tương đương; (iii) cải thiện prediction không dẫn tới lợi ích control; (iv) tính đúng đắn đòi truy cập oracle tham số/transition không thể có tại deployment; (v) chính giả định controllability của môi trường test bị vi phạm mà phương pháp vẫn khẳng định guarantee.

## 10. Đóng góp kỳ vọng, không khẳng định trước

- **C1 — Representation:** Đề xuất (nếu cần) một lớp action-dependent dynamics biểu cảm hơn nhưng có cấu trúc phục vụ control.
- **C2 — Identification:** Một thuật toán cập nhật lõi dynamics từ transition online, với phân tích điều kiện có thể nhận dạng.
- **C3 — Evidence:** Benchmark/protocol chứng minh trade-off giữa prediction, action-response fidelity, sample efficiency và closed-loop control dưới dynamics shift.

Không có C1/C2 nào mặc nhiên mới; đóng góp chỉ được xác nhận sau khi literature review sâu hơn và thực nghiệm nghiêm ngặt.

## 11. Ưu tiên thực thi

**P0:** đọc paper gốc, Appendix F/G.5, kiểm tra GitHub, reproduce → **P1:** controlled shifts + failure taxonomy → **P2:** so sánh MAKO/COLoKe/input encoder/bilinear → **P3:** chọn đúng 1 gap có bằng chứng → **P4:** toán học/kiến trúc mới và ablations → **P5:** mở rộng benchmark và tổng hợp paper.

## 12. Tài liệu tham khảo

1. Miao et al. (2026). *Learning Koopman Representations with Controllability Guarantees*, ICLR 2026. https://openreview.net/pdf?id=jITPFROpWN ; code: https://github.com/KYMiao/Controllable-Koopman
2. Han et al. (2026). *MAKO: Meta-Adaptive Koopman Operators for learning-based MPC...*, Automatica. https://www.sciencedirect.com/science/article/pii/S0005109826000117
3. Gao et al. (2025). *Conformal Online Learning of Deep Koopman Linear Embeddings*, NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/784fd5a46dfe303e5b51c8621b84cf3f-Abstract-Conference.html
4. Fan et al. (2026). *Learning stable Koopman embeddings for identification and control*, Automatica. https://www.sciencedirect.com/science/article/pii/S0005109826001871
5. Zhang et al. (2026). *DWM: Separating World Effects from Actions in Latent World Models*. https://arxiv.org/abs/2607.18715
6. Jha et al. (2026). *CausalDreamer: Learning Predictive World Models with Latent Disentanglement*. https://arxiv.org/abs/2610.12016
7. Rajkumar et al. (2026). *K-PWM: Control-Oriented Structured Bilinear World Models under Partial Observation*. https://arxiv.org/abs/2611.11521
8. Zhang et al. (2026). *On the Identifiability of Controlled World Models*. https://arxiv.org/abs/2607.22430

---

**Decision log — 12/11/2026:** Giữ research direction **Adaptive Action-Dependent Latent Dynamics** và baseline **ICLR 2026**. Chưa chốt kiến trúc. Đề xuất **hai nhánh prediction/controllability** được tách khỏi RRD này và sẽ được phản biện độc lập trước khi bổ sung.
