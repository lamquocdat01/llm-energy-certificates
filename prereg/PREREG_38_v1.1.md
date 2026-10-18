# PREREG_38_v1.1 — Đề tài 38 (B5) · bổ sung H5–H7 · đăng ký 17-10-2026 (trước khi chạy E4, E5, probe)
**Bổ sung cho `PREREG_38_v1.md` (không sửa H1–H4, không sửa file v1).** Sau khi ghi, KHÔNG sửa file này; mọi lệch → mục "deviation" trong `P2/CP_B2_38_<ngày>.md`.
Quyết định CP-B1 áp dụng (anh Đạt 16-10-2026): Shapley = attribution chính, proportional = phụ (supp); regime b chính giữ `meas` (TokenPowerBench), b tự đo (E5) báo song song.

## 0. Minh bạch về thứ tự
- Lúc đăng ký: `code/batch_kv.py`, `code/e5_laptop.py`, `code/e5_probe.py` **chưa viết**; chưa có mô hình GGUF, chưa có số đo laptop nào. Đã làm *trước* PREREG (chỉ thăm dò thiết bị, không đo LLM): kiểm quyền admin (không có), phát hiện bộ đếm Windows `\Energy Meter(RAPL_Package0_PKG)\Energy` (đơn vị pWh, độ phân giải ~20 ms) đọc được không cần admin; 1 lần lấy mẫu 1,2 s lúc máy đang chạy nền (~18–21 W).
- sha256 của code mới sẽ được ghi vào `P2/RUNLOG_38_P2.md` ở lần chạy xác nhận đầu tiên; mọi sửa code sau lần chạy đó → deviation.

## 1. Giả thuyết & tiêu chí PASS/FAIL
| H | Phát biểu | Đo | PASS khi |
|---|---|---|---|
| **H5** | Γ batching đo ≤ cận Thm 4(a)–(c), 0 vi phạm | E4 (§2) | Ở **100 %** ô của lưới §2 và mọi request nạn nhân (benign + victim), 5 seed: Γ_đo ≤ cận tương ứng (sai số số học ≤ 1e-9 tương đối). Cận: (i) reservation → Thm 4(a); (ii) FCFS-priority recompute + watermark h = B → Thm 4(b); (iii) size-based preempt **không** chunked → Thm 4(c2) (và (d) khi đếm số lần admission của tenant khác); (iii) + chunked → không có cận hữu hạn, kiểm riêng ở H5-L. **1 vi phạm = FAIL**; khi FAIL: sửa phát biểu định lý (không "ép" số), ghi deviation. |
| H5-L (đi kèm) | Livelock (iii) + chunked prefill, Thm 4(c1) | E4 cấu hình dựng theo chứng minh (c1) | Γ tăng đơn điệu theo horizon H ∈ {10³, 10⁴, 10⁵} iteration và nạn nhân không hoàn tất prefill trong H; cùng luồng tấn công dưới (i)/(ii) cho Γ ≤ cận. Báo số, không phải điều kiện PASS của H5. |
| **H6** | e_dec(c) tuyến tính theo ngữ cảnh trên laptop | E5 (§3) | Fit OLS e_dec(c) = a + b·c trên năng lượng **net** (trừ công suất idle) theo cửa sổ 64 token: **R² ≥ 0,95** và **CI 95 % bootstrap của b nằm hoàn toàn > 0**. Báo thêm (không phải tiêu chí): bản gross, b/a so với κ của regime meas/arch/t4fit, E_pre(p). |
| **H7** | Probe controllability (chỉ chạy nếu nguồn công suất OK và tổng thời gian probe ≤ 1 ngày) | E5-probe (§4) | ≥ 1 prompt kiểu Deadlock/ExtendAttack đẩy độ dài sinh **> 20 000 token** mà **cả** gate entropy (EAT-like) **và** loop detector mô phỏng trên chuỗi token thật **không dừng** trước 20 000. Kết quả âm cũng báo. Không chạy → "NOT RUN" + lý do. |

Seed: **42 + {0..4}** (như v1).

## 2. E4 — `code/batch_kv.py` (cố định trước khi code)
**Mô hình.** Theo THEORY_38 §5. Mỗi iteration: mỗi request đang decode sinh 1 token; KV của j = κ_j = p_j + t_j (cộng số token đã prefill khi chunked). Năng lượng iteration: e_k = A_0·1[D_k ≠ ∅] + Σ_{j∈D_k}(a_1 + b·κ_j) + Σ prefill chunk α·c. D_k = tập đang decode. Token thứ t sinh ở ngữ cảnh κ = p + t (khớp E(p, ℓ) của §1). Prefill không chunked: prefill toàn bộ κ (p + số token đã sinh khi recompute) trong iteration admission, request decode token kế ngay trong iteration đó. Chunked: chunk = 512 token/iteration/request, request chưa sinh token cho tới khi prefill xong. Preempt = recompute (bỏ KV, xếp lại **đầu** hàng đợi).
**Attribution.** Shapley (chính): φ^S_v = A_0/|D_k| + a_1 + bκ_v; proportional (supp): φ^P_v = (A_0 + Σ_{D_k}(a_1 + bκ_j))/|D_k|. Prefill/recompute tính cho chủ request. Năng lượng solo: E(p_v, ℓ_v) với ℓ_v = min(ℓ, M) (served length).
**Tham số năng lượng.** a, b, α từ `results/energy_params.json`: llama31_8b|meas (chính) và llama31_8b|arch (b lớn, stress). Tách a = A_0 + a_1 với a_1/a ∈ {0,1 (chính); 0,5}. Attribution là tuyến tính theo (A_0, a_1, b, α) nên mọi bộ tham số tính trên cùng một lần mô phỏng.
**Luật admission/preempt.**
- (i) KV-reservation: nhận nếu |R|+1 ≤ B và (|R|+1)(P_max + M) ≤ K; không preempt.
- (ii) FCFS-priority recompute: nhận head-of-line FCFS khi |R|+1 ≤ B và Σκ + κ_j + h ≤ K, h = B; khi bước kế không vừa K → preempt request **đến muộn nhất** trước.
- (iii) size-based: nhận như (ii) với h = 0; khi không vừa → preempt request có **κ lớn nhất** trước.
- Chunked prefill: bật/tắt cho cả 3 luật.

**Lưới.** Trace nạn nhân: azure2024_conv (chính) và burstgpt; limiter (P_max, M) = (prompt lớn nhất của trace, Q_0,99(ℓ) của trace); benign ℓ > M bị cắt về M. B ∈ {8, 32}; K = κ_K·(P_max + M) với κ_K ∈ {1, 2, 4, 8}; luật ∈ {i, ii, iii}; chunked ∈ {tắt, bật}; số request đối thủ n_adv ∈ {0, 1, 3} với (p, ℓ) = (P_max, M), đến tại thời điểm request nạn nhân ngẫu nhiên (theo seed). Cửa sổ replay: mỗi seed × mỗi ngày (7 ngày Azure 2024; BurstGPT: 7 ngày đầu có dữ liệu) lấy 1 000 request liên tiếp bắt đầu tại thời điểm ngẫu nhiên trong ngày. Thời gian → iteration: Δ chọn sao cho tải đề nghị = u·n_cap token/iteration, n_cap = min(B, ⌊K/(p̄ + ℓ̄)⌋), u ∈ {0,8; 1,2} (1,2 = quá tải để có preempt). Mỗi ô chạy tới khi mọi request xong (giới hạn cứng 2·10⁶ iteration; chạm giới hạn → ghi "không hoàn tất", không loại).
**Đo.** Γ_v = E^attr_v / E(p_v, ℓ_v) cho **mọi** request không phải đối thủ; Γ_ô = max_v. Cận dùng đúng ℓ_v, B, n* = min(B, ⌊K/(P_max+M)⌋), ρ, α/a. Báo: tỷ lệ Γ/cận lớn nhất, số preempt/nạn nhân, Γ vs K/(P_max+M) và B (hình F4).
**Livelock (H5-L).** Dựng như chứng minh Thm 4(c1): đối thủ p_a = 1, ℓ_a = 256 (< chunk − 1), luồng liên tục giữ hàng đợi không rỗng; nạn nhân p_v = 4 096 > chunk; K = p_v + 16·257; B = 32. Báo Γ(H) dưới (iii)+chunked, (iii) không chunked, (ii), (i).

## 3. E5 — đo laptop (`code/e5_laptop.py`)
**Nguồn công suất** (thứ tự PROMPT): (a) LibreHardwareMonitor → (b) HWiNFO64 → (c) Intel Power Gadget; cả ba cần driver kernel/admin. Session không có quyền admin ⇒ (a)–(c) ghi "không dùng được (không admin)" và dùng **(d) bộ đếm Windows Energy Meter `RAPL_Package0_PKG`** (cùng thanh ghi MSR RAPL, do driver EMI của Windows phơi ra; pWh tích luỹ, ~20 ms). Đây là **deviation** so với PROMPT (nguồn d không có trong danh sách) — ghi trong CP-B2. Không đọc được (d) → `P2/STOP_E5.md`.
**Phần mềm/mô hình.** llama.cpp bản CPU chính thức (release Windows x64, AVX2/AVX-512), `llama-server` gọi qua HTTP stream (để có dấu thời gian từng token); mô hình Qwen2.5-1.5B-Instruct Q4_K_M GGUF (Apache-2.0) tại `D:\Datasets\LLM_Models\`. 4 luồng, `cache_prompt = false`, `ignore_eos`, n_predict = 256, nhiệt độ 0,7, seed = 42 + r.
**Thiết kế.** p ∈ {128, 512, 1k, 2k, 4k, 8k, 16k} (bỏ mức nào làm RAM trống < 6 GB); prompt tổng hợp (văn bản ngẫu nhiên theo seed, cắt đúng p token theo tokenizer của server); 3 lần lặp (r = 0, 1, 2), thứ tự p xáo theo seed. Idle 60 s trước và sau toàn bộ chuỗi + 10 s nghỉ giữa các lần chạy.
**Phân tích (đóng băng).** P_idle = trung bình công suất 2 đoạn idle. Năng lượng net của khoảng [t₀, t₁] = ΔE_RAPL − P_idle·(t₁ − t₀) (nội suy tuyến tính bộ đếm). E_pre(p) = năng lượng net từ lúc gửi request đến token đầu. Decode: chia 256 token thành cửa sổ 64 token (4 cửa sổ/lần chạy); e_dec của cửa sổ = năng lượng net / 64, gán ngữ cảnh c = p + trung điểm cửa sổ. Fit OLS e_dec = a + b·c trên mọi cửa sổ (7 × 3 × 4 = 84 điểm nếu đủ p); CI 95 % bằng bootstrap theo lần chạy (resample run, 2 000 mẫu, seed 42). Báo κ_laptop = b/a so với κ meas = 5,2e-5, arch ≈ 1,0e-3. Fit phụ: E_pre(p) = αp + βp².
**Limitations ghi sẵn.** CPU laptop ≠ GPU cluster; chỉ xác nhận **dạng** tuyến tính và **dấu** b; không chuyển số tuyệt đối sang regime GPU. Công suất package gồm cả tiến trình nền (trừ bằng P_idle).

## 4. E5-probe — controllability (`code/e5_probe.py`, chỉ khi §3 OK)
3–5 prompt tự soạn theo **mô tả** trong bài Deadlock (2510.15965) / ExtendAttack (2506.13737) / yêu cầu liệt kê-vô-hạn (không tải payload lạ). max_tokens = 24 000, ngữ cảnh 32 768, nhiệt độ 0,7, seed 42. Gate mô phỏng trên chuỗi token thật (logprob top-k từ server): (1) **entropy EAT-like**: dừng khi trung bình trượt 64 token của entropy top-k < 0,1 nat liên tục 128 token (tín hiệu "đã hội tụ"); (2) **loop detector**: dừng khi một n-gram 32 token lặp ≥ 3 lần trong cửa sổ 2 048 token gần nhất. Ghi độ dài đạt được, vị trí gate bắn (nếu có). Giới hạn thời gian probe ≤ 1 ngày; vượt → dừng, báo phần đã có.

## 5. Không thay đổi
H1–H4, dữ liệu, seed, tham số `energy_params.json` của v1 giữ nguyên. Số E1–E3 không chạy lại trừ khi `make_numbers.py` cần sinh lại khoá.
