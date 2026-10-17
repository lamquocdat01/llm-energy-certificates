"""Shared paths and trace loaders for topic 38 (LLM decoding energy certificates).

Dataset root comes from the THS_DATASETS environment variable (default D:\\Datasets);
no dataset path is hard-coded beyond that convention. Parsed traces are cached as
parquet under <THS_DATASETS>/_derived/38/ (outside the Google-Drive-synced folder).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TOPIC_DIR = Path(__file__).resolve().parents[1]
RESULTS = TOPIC_DIR / "results"
DATASETS = Path(os.environ.get("THS_DATASETS", r"D:\Datasets"))
TRACE_ROOT = DATASETS / "LLM_Traces"
DERIVED = DATASETS / "_derived" / "38"

SEEDS = [42 + i for i in range(5)]  # PREREG: seed = 42 + {0..4}

# name -> (loader key, files, description)
TRACES = {
    "azure2023_code": ("azure", ["Azure_LLM_2023/AzureLLMInferenceTrace_code.csv"], "Azure LLM 2023, coding (1 h)"),
    "azure2023_conv": ("azure", ["Azure_LLM_2023/AzureLLMInferenceTrace_conv.csv"], "Azure LLM 2023, conversation (1 h)"),
    "azure2024_code": ("azure", ["Azure_LLM_2024/AzureLLMInferenceTrace_code_1week.csv"], "Azure LLM 2024, coding (1 week)"),
    "azure2024_conv": ("azure", ["Azure_LLM_2024/AzureLLMInferenceTrace_conv_1week.csv"], "Azure LLM 2024, conversation (1 week)"),
    "burstgpt": ("burst", [f"BurstGPT/BurstGPT_without_fails_{i}.csv" for i in (1, 2, 3)], "BurstGPT v2.0 without fails (ChatGPT+GPT-4)"),
    "mooncake_conv": ("mooncake", ["Mooncake_FAST25/conversation_trace.jsonl"], "Mooncake FAST'25 conversation (1 h)"),
    "mooncake_tool": ("mooncake", ["Mooncake_FAST25/toolagent_trace.jsonl"], "Mooncake FAST'25 tool&agent (1 h)"),
}

# Trace groups used as "the 4 traces" of the PLAN (Azure 2023, Azure 2024, BurstGPT, Mooncake).
TRACE_GROUPS = {
    "Azure2023": ["azure2023_code", "azure2023_conv"],
    "Azure2024": ["azure2024_code", "azure2024_conv"],
    "BurstGPT": ["burstgpt"],
    "Mooncake": ["mooncake_conv", "mooncake_tool"],
}


def _load_raw(name: str) -> pd.DataFrame:
    kind, files, _ = TRACES[name]
    frames = []
    for i, f in enumerate(files):
        path = TRACE_ROOT / f
        if kind == "azure":
            df = pd.read_csv(path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
            ts = pd.to_datetime(df["TIMESTAMP"], utc=True, format="mixed")
            t = (ts - ts.min()).dt.total_seconds().to_numpy()
            out = pd.DataFrame({"t": t, "p": df["ContextTokens"].to_numpy(), "l": df["GeneratedTokens"].to_numpy()})
        elif kind == "burst":
            df = pd.read_csv(path, usecols=["Timestamp", "Model", "Request tokens", "Response tokens"])
            out = pd.DataFrame({"t": df["Timestamp"].astype(float).to_numpy(), "p": df["Request tokens"].to_numpy(),
                                "l": df["Response tokens"].to_numpy(), "model": df["Model"].astype("category"),
                                "part": i + 1})
        elif kind == "mooncake":
            rows = [json.loads(x) for x in open(path, encoding="utf-8")]
            out = pd.DataFrame({"t": np.array([r["timestamp"] for r in rows], float) / 1000.0,
                                "p": [r["input_length"] for r in rows], "l": [r["output_length"] for r in rows]})
        else:
            raise ValueError(kind)
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df["p"] = df["p"].astype("int64")
    df["l"] = df["l"].astype("int64")
    return df


def load_trace(name: str, refresh: bool = False) -> pd.DataFrame:
    """Columns: t (s from trace start; BurstGPT parts keep their own clocks), p (prompt tokens), l (output tokens)."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    cache = DERIVED / f"{name}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = _load_raw(name)
    df.to_parquet(cache, index=False)
    return df


def load_group(group: str) -> pd.DataFrame:
    return pd.concat([load_trace(n).assign(sub=n) for n in TRACE_GROUPS[group]], ignore_index=True)


def ram_guard(min_free_gb: float = 6.0) -> None:
    """Workspace rule: heavy jobs stop automatically when free RAM < 6 GB."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return
    free = psutil.virtual_memory().available / 2**30
    if free < min_free_gb:
        raise SystemExit(f"[ram_guard] free RAM {free:.1f} GB < {min_free_gb} GB -> stopping (rerun later)")
