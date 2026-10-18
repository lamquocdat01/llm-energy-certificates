# PREREG_38_v1 — Đề tài 38 (B5) · đăng ký trước 16-10-2026 (trước khi chạy E1/E2-full/E3)
**Sau khi ghi, KHÔNG sửa file này.** Mọi lệch → mục "deviation" trong `P1\CP_B1_38_<ngày>.md`.

## 0. Minh bạch về thứ tự
- Theo PROMPT_38_P0P1, T03 (đường giá **sơ bộ**, lưới q thô {0,90; 0,95; 0,99; 0,999; max}) chạy **trước** PREREG. Các số sơ bộ của H1/H4 đã được xem (ghi trong `P0\RUNLOG_38.md`). Định nghĩa H1/H4 dưới đây lấy **nguyên văn** từ PROMPT (soạn trước khi có số); phép kiểm xác nhận (confirmatory) là E2 (`certificate_curve.py --full`: lưới q mịn + bootstrap theo ngày × 5 seed).
- E1, E3 chưa chạy, code E3 (`adaptive_vs_cap.py`) chưa viết khi đăng ký; thiết kế E3 được cố định ở §3.

## 1. Dữ liệu, mô hình, tham số (đóng băng)
- Trace (sub-trace = đơn vị phân tích; "họ" = family): Azure2023 {code, conv}, Azure2024 {code, conv}, BurstGPT {ChatGPT+GPT-4, without_fails 1–3}, Mooncake {conversation, toolagent}. LMSYS-Chat-1M: không có (thiếu HF token).
- Mô hình năng lượng: `code/energy_model.py`; tham số `results/energy_params.json` (3 mô hình: llama31_8b, llama31_70b, qwen3_8b_reason; 3 chế độ b: **meas (chính)**, arch, t4fit). Context window C = 131 072.
- Seed: **42 + {0..4}**. Bootstrap: khối = ngày (Azure2024, BurstGPT), khối 5 phút (trace 1 giờ); B = 200 mẫu/seed; báo khoảng 2,5–97,5 % và dải qua 5 seed.

## 2. Giả thuyết & tiêu chí PASS/FAIL
| H | Phát biểu | Đo | PASS khi |
|---|---|---|---|
| **H1** | Cap ℓ tại p99 giảm E_max ≥ 5× với tỷ lệ cắt ≤ 1 % trên ≥ 2 trace | E2, biến thể `l`, q = 0,99, regime meas, P_max = prompt lớn nhất của trace | Có ≥ 2 sub-trace thuộc ≥ 2 họ khác nhau, với **mọi** mô hình trong 3, đạt red ≥ 5 và cut ≤ 1 %. Báo **riêng hai baseline**: H1-ctx (không cap: p+ℓ ≤ C) và H1-max (M = max ℓ của trace). Theo PROMPT ("so với max-trace (hoặc context length)"), H1 PASS nếu H1-ctx hoặc H1-max PASS; cả hai đều được báo. |
| **H2** | worst-case(gate thích nghi ∘ cap) = worst-case(cap) trên mọi ô E3 | E3 §3, 4 gate × 3 cap × 4 trace | Sai lệch tương đối |WC(g∘L) − WC(L)| / WC(L) ≤ 1e-6 ở **mọi** ô, dưới mô hình đối thủ (B) "reachable set công bố". (A) đối thủ lý thuyết (controllability) là kiểm code, phải cho 0 sai lệch. |
| **H3** | E_max bậc hai theo M | E1: E_max(P_max, M) tính bằng mô phỏng từng token trên lưới (P_max, M, a, b) ≥ 500 ô | Với mọi bộ (P_max, a, b) có b > 0: fit đa thức bậc 2 theo M có R² ≥ 0,999; và \|hệ số bậc 2 − b/2\| ≤ 1e-6·b/2 |
| **H4** | Cap tổng (p+ℓ) cho E_max thấp hơn cap ℓ ở cùng tỷ lệ cắt trên ≥ 2 trace | E2, q = 0,99 (cut cả hai ≈ 1 %), regime meas | ≥ 2 sub-trace thuộc ≥ 2 họ có E_max(tot) < E_max(l) với ≥ 2/3 mô hình |
| H5 | Γ batching đo ≤ cận Thm 4, 0 vi phạm | E4 (P2) | — không kiểm ở P1 |
| H6 | Laptop e(ctx) tuyến tính R² ≥ 0,95 | E5 (P2, tuỳ chọn) | — không kiểm ở P1 |

Phân tích phụ (thăm dò, không phải giả thuyết): biến thể `pareto` (cap theo p của Thm 3: chứng chỉ = quantile của E(p,ℓ)); OPF theo cửa sổ SLA W ∈ {60 s, 3600 s, 86 400 s}; độ nhạy theo regime arch/t4fit.

## 3. Thiết kế E3 (adaptive-vs-cap) — cố định trước khi code
- Trace (1 sub-trace/họ): azure2023_conv, azure2024_conv, burstgpt, mooncake_conv. Mô hình năng lượng: llama31_8b|meas (chính) + 2 mô hình còn lại báo phụ.
- Cap (3): M = Q_0,99(ℓ), M = Q_0,999(ℓ), M = C − P_max (không max_tokens). P_max = prompt lớn nhất của trace.
- Gate (mô hình ẩn, mỗi request lành có độ dài tự nhiên ℓ; ngẫu nhiên theo seed):
  1. **TALE-like budget**: ℓ̂ = ℓ·exp(0,5·Z), Z ~ N(0,1); budget = ⌈ℓ̂⌉; phục vụ min(ℓ, budget, M).
  2. **Entropy early-exit (DEER/EAT-like)**: thời điểm "đáp án sẵn sàng" τ = ⌈U·ℓ⌉, U ~ Beta(2, 2); gate dừng tại τ + 32 (trễ phát hiện) nếu < ℓ.
  3. **Conformal stop**: ước lượng τ̂ = τ·exp(0,3·Z); dừng tại ⌈c_δ·τ̂⌉, c_δ hiệu chỉnh trên nửa calibration (tách ngẫu nhiên theo seed) sao cho P(c_δ·τ̂ < τ) ≤ δ = 0,05; đánh giá trên nửa test.
  4. **Loop detector**: request lành có ℓ ≥ Q_0,999(ℓ) coi là runaway/lặp; khởi phát lặp tại ⌈V·ℓ⌉, V ~ U(0,25; 0,75); dừng sau cửa sổ 64 token. Request khác không bị ảnh hưởng.
- Benign: tiết kiệm kỳ vọng S_benign = 1 − E[E(g∘L)] / E[E(L)] trên toàn trace (trung bình 5 seed + dải).
- Đối thủ:
  - (A) lý thuyết: chọn p ≤ P_max và nội dung thoả controllability (gate không bắn trước M) ⇒ WC = E(P_max, M).
  - (B) reachable set công bố: mỗi họ tấn công f (Engorgio, OverThink, ThinkTrap, LoopLLM, ExtendAttack, Deadlock; thêm ReasoningBomb/Excessive Reasoning nếu có số) có độ dài đạt được ℓ_f = max_tokens của thí nghiệm gốc nếu bài báo báo có request chạm cap, ngược lại = độ dài output tấn công lớn nhất (hoặc trung bình nếu chỉ có trung bình) được công bố; số + vị trí bảng/hình trong `results/attack_lengths_published.json`. Ma trận fires(g, f) ∈ {0,1} (gate g có dừng được họ f trước ℓ_f không) ghi kèm lý do trích từ bài: loop detector bắn với họ lặp (LoopLLM); TALE-budget không bắn với họ làm câu hỏi "có vẻ khó" (ExtendAttack, OverThink); entropy/conformal không bắn với họ không hội tụ (Deadlock, ThinkTrap). Đối thủ chọn p = P_max (độ dài prompt là đại lượng đối thủ kiểm soát) và họ f tối đa năng lượng: WC_B(g∘L) = max_f E(P_max, min(ℓ_f, M, τ_g,f)), với τ_g,f = ℓ_f nếu fires = 0, = điểm dừng công bố/mô hình nếu fires = 1. WC_B(L) = max_f E(P_max, min(ℓ_f, M)).
  - Tiết kiệm worst-case S_wc = 1 − WC(g∘L)/WC(L).
- Bảng kết quả: 4 gate × 3 cap × 4 trace: S_benign (%) và S_wc (%) theo (A) và (B). Hình `results/fig_adaptive_vs_cap.png` (2 cột: benign vs worst-case).

## 4. Kiểm cận (T11, `verify_bounds.py --seed`)
Thm 1: mọi request trace (và 10⁵ request ngẫu nhiên/seed trong miền p ≤ P_max, ℓ ≤ M) có E ≤ E_max; Thm 2: WC(g∘L) ≤ WC(L) và = dưới (A); Thm 3: không chính sách cap cố định nào (lưới M) có E_max ≤ T với cut < cut_pareto(T). **0 vi phạm** = PASS.

## 5. Đóng băng
sha256 (16-10-2026) tại thời điểm đăng ký:
```
27358ed120e514245643b53523cdfaf1d6b8e71be58bcccf9459b799c50b863c code/common.py
ec38e825897fb1efb563da02733b7473e243f303017cd1492ec828c0101fa55f code/energy_model.py
391e46b46a82d0ea25b567c7e3d7d2821d2fc761d53e521e57393d0f4d46569e code/build_energy_params.py
3d9084a6537da6cea48c526218047dedd87944a7ca6432716b2dbb5c6dbb898b code/trace_summary.py
52978f3e295114b9e5d04d8728a563c7876cd96152751e6252ea4e16b9a5828b code/certificate_curve.py
8600a88ae6dc49bdfdc02f4abdced0ef14970fe8de8796868a18a7d0221b3440 code/tests/test_energy_model.py
4f6e91193b60382787507f8524b0c4ada98e3c3736efbb92fea72500ee362053 results/energy_params.json
```
Mọi thay đổi code sau mốc này phải ghi trong CP_B1 (lý do + ảnh hưởng tới H nào).
