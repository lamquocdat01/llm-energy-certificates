# Published per-token energy numbers for LLM inference (raw source mining)

Retrieved 2026-10-16. Full machine-readable data is in `energy_sources_raw.json`: 559 entries and 92 context-scaling points, each with source and exact location. Every value that is not a printed number has `approx=true` and a stated derivation.

## Sources
| id | what | prefill/decode split? |
|---|---|---|
| arXiv:2505.06371v2 (ML.ENERGY Benchmark, NeurIPS'25 D&B) | J per generation (Tables 1, 4, 5); GPU-only energy (Zeus) | Only App. B.3 **Table 4** (PD disaggregation: J/gen and % decode). Per-token prefill/decode is **derived** by us. |
| ML.ENERGY Leaderboard v3 data, github.com/ml-energy/leaderboard @2bdd007, `public/data/models/*.json` | `energy_per_token_joules` (steady-state GPU energy / output tokens) + `energy_per_request_joules`, `avg_output_len` | **No**: per output token, with prefill folded in |
| arXiv:2512.03024v1 (TokenPowerBench) | figures only; 4×H100 (94 GB) nodes; node-level and GPU-level | Fig. 2(a) prefill (**unit unlabeled, not per token**) and Fig. 2(b) "Decode energy per token" (the body text calls it "total"). Values were read off the figure. |

## Representative values (J per token)
| model | GPU × n | batch / conc | phase | value | source / location |
|---|---|---|---|---|---|
| Llama-3.1-8B | 1P3D, TP1 (4 GPUs; GPU type not stated) | in 512 / out 512 | decode | 0.0569 /out-tok (derived) | 2505.06371 Tab. 4 |
| Llama-3.1-8B | same | in 512 / out 512 | prefill | 0.0168 /in-tok (derived) | 2505.06371 Tab. 4 |
| Llama-3.1-8B | same | in 4096 / out 512 | decode | 0.273 /out-tok (derived) | 2505.06371 Tab. 4 |
| Llama-3.1-8B | same | in 4096 / out 512 | prefill | 0.0167 /in-tok (derived) | 2505.06371 Tab. 4 |
| Llama-3.1-70B | 1P1D, TP4 (8 GPUs) | in 512 / out 512 | decode / prefill | 0.350 / 0.190 (derived) | 2505.06371 Tab. 4 |
| Llama-3.1-70B | 1P1D, TP4 | in 4096 / out 512 | decode / prefill | 1.458 / 0.182 (derived) | 2505.06371 Tab. 4 |
| Llama-3.1-8B-Instruct | H100 × 1 | max_num_seqs 8 → 512 | total/out-tok | 0.455 → 0.0702 | LB v3 json |
| Llama-3.1-70B-Instruct | H100 × 4 | 8 → 1024 | total/out-tok | 3.76 → 0.373 | LB v3 json |
| Llama-3.1-405B-Instruct | B200 × 8 | 8 → 4096 | total/out-tok | 21.9 → 1.02 | LB v3 json |
| Qwen3-8B (chat) | H100 × 1 | 8 → 512 | total/out-tok | 0.444 → 0.0721 | LB v3 json |
| Qwen3-8B (GPQA, reasoning) | H100 × 1 | 8 → 96 | total/out-tok | 0.604 → 0.221 (avg out ≈ 8k tok) | LB v3 json |
| DeepSeek-R1-0528 (FP8) | B200 × 8 | 8 → 128 | total/out-tok | 9.87 → 2.38 (avg out ≈ 11.3k) | LB v3 json |
| Qwen3-235B-A22B-Thinking | H100 × 8 | 8 → 32 | total/out-tok | 7.20 → 3.61 | LB v3 json |
| gpt-oss-20b | H100 × 1 | 8 → 2048 | total/out-tok | 0.357 → 0.045 | LB v3 json |
| Llama3-8B | H100 × 4, vLLM | not stated | "decode" | ≈0.145 (figure-read) | 2512.03024 Fig. 2(b) |
| Llama3-70B | H100 × 4, vLLM | not stated | "decode" | ≈0.88 (figure-read) | 2512.03024 Fig. 2(b) |
| Reasoning models, per generation | H100, TP 1–8 | max bs 4–64 | per response | e.g. DeepSeek-distill-Qwen3-8B 9713.7 → 2770.8 J | 2505.06371 Tab. 1 (no output lengths, so no J/tok) |

Derivation (Tab. 4): decode = E_gen × decode% / out_mean; prefill = E_gen × (1 − decode%) / in_mean. These are ratios of dataset means: inputs are Pareto(α = 2.5) and outputs Exponential.

## Context-length dependence (92 points in `context_scaling`)
- **ML.ENERGY Tab. 4 (12 points, derived):** decode J/out-tok at mean decode KV length ≈ 1024 vs ≈ 4608. For Llama-3.1-8B 1P3D the values are 0.057 → 0.160 (long output) and 0.273 (long input). But configs with the same mean KV length disagree. For 70B 1P1D, long output gives 0.198 and long input gives 1.458, and the long-output value is lower than the 512/512 value of 0.350. Serving dynamics dominate, so Table 4 alone cannot identify a + b·ctx.
- **TokenPowerBench Fig. 3 (60 points, figure-read):** 10 models × 3 prompt-length bins (0–2K, 2K–5K, 5K–10K), total and GPU-only. Example: Llama3-70B total goes 0.87 → 0.98 → 1.2 J/tok, and Llama3-8B goes 0.145 → 0.164 → 0.191. The rise is modest (about 1.3–1.4×).
- **Leaderboard, indirect (20 points):** Qwen3-8B on chat vs GPQA at the same max_num_seqs, using avg_output_len/2 as a context proxy. The prompt length is unknown.
- **Key quotes:** TPB says "energy grows steadily as the prompt gets longer because the prefill stage must process every input token while the compute cost of each new output token stays the same", and that Llama3-70B 2K→10K is "roughly a factor of three". **That factor contradicts its own Fig. 3 (~1.4×).** ML.ENERGY App. B.3 says "decode consumes the majority of energy, with some amount shifting to prefill when input length is long". It attributes long-context and reasoning per-token cost mostly to the batch-size limits that KV memory imposes (Sec. 4.2).

## Caveats
1. No source gives a clean per-token decode energy measured as a function of KV length. A true e_dec(ctx) = a + b·ctx fit needs raw ML.ENERGY runs (github.com/ml-energy/data, `LLMRuns.from_hf()`, which has timelines and prompt lengths) or our own measurement.
2. Leaderboard data is v3 (H100/B200, 2025-late), not the early-2025 A100/H100 iteration in the paper.
3. TokenPowerBench large-model numbers (Figs. 4–6: 89–286 "J/token" for 405B/180B/480B on 16 H100) are about 100× ML.ENERGY's numbers and probably use a different normalization. Do not use them for calibration without checking the code. Fig. 5 and Fig. 6 also disagree slightly with each other.
4. TPB Fig. 2(a) prefill unit is unlabeled. The batch size and engine for Figs. 3 and 4 are not stated.
5. ML.ENERGY measures GPU only. The TPB "Total" series is node-level.
