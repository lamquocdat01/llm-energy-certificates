"""T01: length statistics of public LLM serving traces -> results/trace_summary.csv, results/fig_len_dist.png."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, TRACES, TRACE_GROUPS, load_trace

QS = [0.5, 0.9, 0.99, 0.999]


def stats_row(trace: str, subset: str, df: pd.DataFrame) -> dict:
    row = {"trace": trace, "subset": subset, "n_requests": len(df),
           "duration_h": (df["t"].max() - df["t"].min()) / 3600.0 if len(df) else np.nan}
    for col, lab in (("p", "prompt"), ("l", "output"), ("tot", "total")):
        x = df[col].to_numpy()
        for q in QS:
            row[f"{lab}_p{str(q)[2:]:0<2}"] = float(np.quantile(x, q))
        row[f"{lab}_max"] = int(x.max())
        row[f"{lab}_mean"] = float(x.mean())
    row["frac_output_zero"] = float((df["l"] == 0).mean())
    return row


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for name in TRACES:
        df = load_trace(name)
        df = df.assign(tot=df["p"] + df["l"])
        rows.append(stats_row(name, "all", df))
        if "model" in df:
            for m, g in df.groupby("model", observed=True):
                rows.append(stats_row(name, f"model={m}", g))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "trace_summary.csv", index=False)
    print(out[["trace", "subset", "n_requests", "output_p99", "output_max", "prompt_p99", "prompt_max"]].to_string())

    # log-log CCDF of output length, one panel line per sub-trace, coloured by family
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"Azure2023": "#1f77b4", "Azure2024": "#ff7f0e", "BurstGPT": "#2ca02c", "Mooncake": "#d62728"}
    styles = ["-", "--"]
    for fam, subs in TRACE_GROUPS.items():
        for k, name in enumerate(subs):
            l = np.sort(load_trace(name)["l"].to_numpy())
            l = l[l > 0]
            ccdf = 1.0 - np.arange(len(l)) / len(l)
            idx = np.unique(np.geomspace(1, len(l), 2000).astype(int) - 1)
            ax.loglog(l[idx], ccdf[idx], styles[k], color=colors[fam], lw=1.4, label=name)
    ax.set_xlabel("output length ℓ (tokens)")
    ax.set_ylabel("P(ℓ_r ≥ ℓ)")
    ax.set_title("Output-length CCDF, public LLM serving traces")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_len_dist.png", dpi=200)


if __name__ == "__main__":
    main()
