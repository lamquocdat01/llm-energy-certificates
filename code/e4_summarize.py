"""E4 summary: PREREG v1.1 H5 verdict (Thm 4 bounds, 0 violations), stall/livelock counts, figure F4.

Reads results/e4_grid_<trace>.csv and results/e4_livelock*.csv; writes results/e4_summary.csv, results/e4_h5.json,
results/e4_gamma.{png,pdf}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS

MAIN = dict(regime="meas", a1f=0.1)
CHECKED = {("i", False), ("i", True), ("ii", False), ("ii", True), ("iii", False)}   # cells with a finite Thm 4 bound


def load_grid() -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(RESULTS.glob("e4_grid_*.csv"))], ignore_index=True)


def summarize() -> None:
    g = load_grid()
    g["checked"] = [(r, c) in CHECKED for r, c in zip(g.rule, g.chunked)]
    keys = ["trace", "rule", "chunked"]
    s = g.groupby(keys).agg(cells=("viol_shap", "size"), windows=("seed", lambda x: x.size),
                            viol_shap=("viol_shap", "sum"), viol_prop=("viol_prop", "sum"),
                            gamma_shap_max=("gamma_shap_max", "max"), gamma_prop_max=("gamma_prop_max", "max"),
                            ratio_shap_max=("ratio_shap_max", "max"), ratio_prop_max=("ratio_prop_max", "max"),
                            preempt_max=("preempt_max", "max"), stalled=("stalled", "sum"),
                            incomplete=("complete", lambda x: int((~x.astype(bool)).sum()))).reset_index()
    s.to_csv(RESULTS / "e4_summary.csv", index=False)
    chk = g[g.checked]
    by = chk.groupby(["rule", "chunked"])[["viol_shap", "viol_prop"]].sum()
    ii = g[(g.rule == "ii")]
    out = {
        "traces": sorted(g.trace.unique().tolist()),
        "cells_total": int(len(g)), "cells_checked": int(len(chk)),
        "victim_checks": int(chk.n_done.sum()),
        "violations_shap": int(chk.viol_shap.sum()), "violations_prop": int(chk.viol_prop.sum()),
        "violations_by_rule": {f"{r}|{'chunked' if c else 'whole'}": dict(shap=int(v.viol_shap), prop=int(v.viol_prop))
                               for (r, c), v in by.iterrows()},
        "max_ratio_to_bound_shap": float(chk.ratio_shap_max.max()), "max_ratio_to_bound_prop": float(chk.ratio_prop_max.max()),
        "ii_preempt_gt_Bm1_cells": {("chunked" if c else "whole"): int((~sub.preempt_le_Bm1.astype(bool)).sum())
                                    for c, sub in ii.groupby("chunked")},
        "ii_preempt_max": {("chunked" if c else "whole"): int(sub.preempt_max.max()) for c, sub in ii.groupby("chunked")},
        "stalled_cells": {f"{r}|{'chunked' if c else 'whole'}|nadv{n}": int(v)
                          for (r, c, n), v in g[g.regime.eq("meas") & g.a1f.eq(0.1)].groupby(
                              ["rule", "chunked", "n_adv"]).stalled.sum().items() if v},
        "gamma_shap_max_main": {f"{r}|{'chunked' if c else 'whole'}": float(v) for (r, c), v in
                                g[g.regime.eq("meas") & g.a1f.eq(0.1)].groupby(["rule", "chunked"]).gamma_shap_max.max().items()},
        "gamma_prop_max_main": {f"{r}|{'chunked' if c else 'whole'}": float(v) for (r, c), v in
                                g[g.regime.eq("meas") & g.a1f.eq(0.1)].groupby(["rule", "chunked"]).gamma_prop_max.max().items()},
    }
    out["H5_pass"] = bool(out["violations_shap"] == 0 and out["violations_prop"] == 0)
    # post hoc (deviation D-E4-1): THEORY v1.1 corrected bounds; chunked (ii) and (iii) have no finite bound
    out["corrected"] = {"violations_shap": int(g.viol_shap_corr.sum()), "violations_prop": int(g.viol_prop_corr.sum()),
                        "cells_with_finite_bound": int(len(g[g.rule.eq("i") | ~g.chunked.astype(bool)]))}
    gm = g[g.regime.eq("meas") & g.a1f.eq(0.1)]
    out["gamma_shap_max_by_nadv"] = {f"{r}|{'chunked' if c else 'whole'}|nadv{n}": float(v) for (r, c, n), v in
                                     gm.groupby(["rule", "chunked", "n_adv"]).gamma_shap_max.max().items()}
    for f in sorted(RESULTS.glob("e4_livelock*.csv")):
        d = pd.read_csv(f)
        d = d[d.regime == "meas"]
        out[f.stem] = {f"{r.rule}|{'chunked' if r.chunked else 'whole'}|H{r.H}": dict(
            gamma_shap=float(r.gamma_shap), done=bool(r.victim_done), preempt=int(r.victim_preempt)) for r in d.itertuples()}
    (RESULTS / "e4_h5.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    plot(g)


def plot(g: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42          # P3: embedded TrueType, no Type 3 (C1)
    import matplotlib.pyplot as plt
    m = g[g.regime.eq("meas") & g.a1f.eq(0.1) & g.trace.eq("azure2024_conv")]
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
    sty = {("i", False): ("(i) reservation", "#1f5aa6", "-"), ("ii", False): ("(ii) FCFS-prio", "#2a9d5c", "-"),
           ("ii", True): ("(ii) + chunked", "#2a9d5c", "--"), ("iii", False): ("(iii) size-based", "#c0392b", "-")}
    for (rule, ch), (lab, col, ls) in sty.items():
        for B, mk in ((8, "o"), (32, "s")):
            d = m[(m.rule == rule) & (m.chunked == ch) & (m.B == B)].groupby("kK").gamma_shap_max.max()
            if len(d):
                ax[0].plot(d.index, d.values, ls=ls, marker=mk, ms=3, lw=1, color=col,
                           label=f"{lab}, B={B}" if B == 32 else None)
    ax[0].axhline(1, color="k", lw=0.6, ls=":")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("K / (P_max + M)"); ax[0].set_ylabel("max Γ (Shapley)")
    ax[0].set_title("(a) Azure 2024 replay, worst window", fontsize=8)
    from matplotlib.lines import Line2D
    h, _ = ax[0].get_legend_handles_labels()
    h += [Line2D([], [], color="#444", marker="o", ms=3, lw=0, label="circles: B=8"),
          Line2D([], [], color="#444", marker="s", ms=3, lw=0, label="squares: B=32")]
    ax[0].set_ylim(0.8, 8e3)
    ax[0].legend(handles=h, fontsize=5.5, frameon=False, ncol=2, loc="upper right")
    for f, col in (("e4_livelock.csv", "#7f7f7f"), ("e4_livelock_B64.csv", "#c0392b")):
        p = RESULTS / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[(d.regime == "meas") & (d.rule == "iii") & d.chunked]
        d = d[d.gamma_shap > 0]                       # victim not yet admitted at the horizon -> nothing to plot
        ax[1].plot(d.H, d.gamma_shap, color=col, lw=1, label=f"(iii)+chunked, B={d.B.iloc[0]}")
        for done, face in ((True, col), (False, "white")):
            q = d[d.victim_done.astype(bool) == done]
            ax[1].plot(q.H, q.gamma_shap, lw=0, marker="o", ms=3.5, color=col, markerfacecolor=face)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("horizon H (iterations)"); ax[1].set_ylabel("victim Γ (Shapley)")
    ax[1].set_title("(b) Thm 4(c1) livelock (open = victim unfinished)", fontsize=8)
    ax[1].legend(fontsize=6, frameon=False)
    for a_ in ax:
        a_.tick_params(labelsize=7); a_.xaxis.label.set_size(8); a_.yaxis.label.set_size(8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"e4_gamma.{ext}", dpi=300)


if __name__ == "__main__":
    summarize()
