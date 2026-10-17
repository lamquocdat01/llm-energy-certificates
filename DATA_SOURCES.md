# Data sources

No dataset is redistributed here. Code reads raw data from `$THS_DATASETS` (default `D:\Datasets`) and caches parsed traces under `$THS_DATASETS/_derived/38/`. Everything in `results/` is derived from the sources below.

## Production traces (`$THS_DATASETS/LLM_Traces/`)
| Trace | URL | License | Citation | SHA-256 |
|---|---|---|---|---|
| Azure LLM Inference 2023 (code, conv) | https://github.com/Azure/AzurePublicDataset (`data/AzureLLMInferenceTrace_{code,conv}.csv`) | CC BY 4.0 | Patel et al., Splitwise, ISCA 2024, doi:10.1109/ISCA59077.2024.00019 | code 54e9a6d2…7fb6, conv 2f1e5b66…0bf8 |
| Azure LLM Inference 2024 (1 week) | https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-llm-2024 | CC BY 4.0 | Stojkovic et al., DynamoLLM, HPCA 2025, doi:10.1109/HPCA61900.2025.00102 | code 71de5c55…4c8e, conv a0cc9b96…3941 |
| BurstGPT v2.0 (without fails, parts 1–3) | https://github.com/HPMLL/BurstGPT/releases/tag/v2.0 | CC BY 4.0 | Wang et al., BurstGPT, KDD 2025, doi:10.48550/arXiv.2401.17644 | a4d068a7…f1d4, 56193aa9…6e501, 3326259f…3a40 |
| Mooncake FAST'25 (conversation, toolagent) | https://github.com/kvcache-ai/Mooncake (`FAST25-release/traces/`) | Apache-2.0 (repository) | Qin et al., Mooncake, USENIX FAST 2025, arXiv:2407.00079 | conv b8cbb061…b9df, tool 48a2db1a…5fa71 |

## Energy parameters (`results/energy_params.json`, `results/energy_sources_raw.*`)
Read from published measurements: ML.ENERGY Benchmark (arXiv:2505.06371) and leaderboard data, TokenPowerBench (arXiv:2512.03024). See `results/energy_sources_raw.md` for the table/figure each value comes from.

## Laptop measurement (E5, `$THS_DATASETS/LLM_Models/`)
| Item | URL | License | SHA-256 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct Q4_K_M GGUF | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF | Apache-2.0 | 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e |
| llama.cpp b11706, win-cpu-x64 (`llama-server`) | https://github.com/ggml-org/llama.cpp/releases/tag/b11706 | MIT | zip f426ae6d20afeb6970f51cc182dedc3b21cf23ec278192da1af262de43be6115 |

Power: Windows performance counter `\Energy Meter(RAPL_Package0_PKG)\Energy` (Intel RAPL package energy, unit pWh, ~20 ms update), read through `win32pdh` without administrator rights. Raw samples are written to `$THS_DATASETS/_derived/38/e5/` and are not included.
