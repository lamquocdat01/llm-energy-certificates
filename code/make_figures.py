"""T17: the five main figures (IEEE column sizes, 300 dpi, PDF + PNG) and table T1.

F1 length CCDF + certificate price curve (Thm 3: fixed output cap vs prompt-aware budget), 4 traces
F2 adaptive gates vs caps (benign vs worst-case saving), 4 gates x 3 caps
F3 E_max quadratic in M and the role of P_max (Thm 1)
F4 batching amplification (Thm 4) -> e4_summarize.plot
F5 laptop e_dec(c) (E5) -> e5_laptop.plot
T1 SLA-window over-provisioning factor (results/table1_opf_sla.csv)
"""
from __future__ import annotations

import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, load_trace  # noqa: E402
from energy_model import E_request, load_params  # noqa: E402

TRACES4 = {"azure2023_conv": "Azure 2023", "azure2024_conv": "Azure 2024", "burstgpt": "BurstGPT",
           "mooncake_conv": "Mooncake"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # categorical slots 1-4 (validated reference palette)
MARKERS = ["o", "s", "^", "D"]
COL1, COL2 = 3.5, 7.16
plt.rcParams.update({"font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "lines.linewidth": 1.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # P3: embedded TrueType, no Type 3 (C1)


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def fig1() -> None:
    cc = pd.read_csv(RESULTS / "e2_cert_curve.csv")
    cc = cc[(cc.model == "llama31_8b") & (cc.regime == "meas")]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for (tr, lab), col, mk in zip(TRACES4.items(), COLORS, MARKERS):
        l = np.sort(load_trace(tr).l.to_numpy())
        l = l[l > 0]
        ccdf = 1 - np.arange(len(l)) / len(l)
        idx = np.unique(np.geomspace(1, len(l), 400).astype(int) - 1)
        ax[0].loglog(l[idx], ccdf[idx], color=col, label=lab)
        for var, ls in (("l", "-"), ("pareto", "--")):
            d = cc[(cc.trace == tr) & (cc.variant == var) & (cc.cut > 0)].sort_values("cut")
            ax[1].loglog(d.cut * 100, d.E_max, ls=ls, color=col, marker=mk if var == "l" else None, ms=2.5,
                         label=f"{lab}, {'fixed cap M=Q(ℓ)' if var == 'l' else 'budget M_T(p)'}")
    ax[0].set_xlabel("output length ℓ (tokens)"); ax[0].set_ylabel("P[ℓ > x]")
    ax[0].set_title("(a) Output-length CCDF"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("truncated requests (%)"); ax[1].set_ylabel("certificate E_max (J)")
    ax[1].set_title("(b) Certificate price curve, Llama-3.1-8B")
    ax[1].invert_xaxis()
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], color="#444", ls="-", marker="o", ms=2.5, label="fixed output cap M = Q(ℓ)"),
                          Line2D([], [], color="#444", ls="--", label="prompt-aware budget M_T(p) (Thm 3)")],
                 frameon=False, loc="lower left")
    save(fig, "fig1_price_curve")


def fig2() -> None:
    d = pd.read_csv(RESULTS / "e3_adaptive_vs_cap.csv")
    d = d[d.model == "llama31_8b"]
    gates = ["tale", "entropy", "conformal", "loop"]
    glab = {"tale": "TALE-like", "entropy": "entropy exit", "conformal": "conformal", "loop": "loop det."}
    caps = sorted(d.cap.unique(), key=lambda c: {"q0.99": 0, "q0.999": 1}.get(c, 2))
    clab = {"q0.99": "M = p99", "q0.999": "M = p99.9"}
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
    w = 0.26
    for k, (col_, title) in enumerate((("S_benign_mean", "(a) benign saving (median, range: 4 traces)"),
                                        ("S_wc_B", "(b) worst-case saving (published attack reach)"))):
        for j, (cap, col) in enumerate(zip(caps, COLORS)):
            g = d[d.cap == cap].groupby("gate")[col_]
            mid, lo, hi = g.median() * 100, g.min() * 100, g.max() * 100
            x = np.arange(len(gates)) + (j - 1) * w
            y = [mid.get(gt, 0) for gt in gates]
            ax[k].bar(x, y, width=w - 0.03, color=col, label=clab.get(cap, "M = C − P_max (no cap)"))
            ax[k].errorbar(x, y, yerr=[np.subtract(y, [lo.get(gt, 0) for gt in gates]),
                                       np.subtract([hi.get(gt, 0) for gt in gates], y)],
                           fmt="none", ecolor="#555", elinewidth=0.6, capsize=1.5)
        ax[k].set_xticks(np.arange(len(gates))); ax[k].set_xticklabels([glab[g] for g in gates])
        ax[k].set_title(title); ax[k].grid(axis="x", visible=False)
    ax[0].set_ylabel("energy saving (%)")
    ax[0].legend(frameon=False)
    save(fig, "fig2_adaptive_vs_cap")


def fig3() -> None:
    prm = load_params(RESULTS / "energy_params.json")
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.4))
    M = np.unique(np.geomspace(16, 131072, 200).astype(int))
    for (P, col) in zip((1024, 8192, 32768, 125591), COLORS):
        for reg, ls in (("meas", "-"), ("arch", "--")):
            pr = prm[f"llama31_8b|{reg}"]
            Mv = M[M <= pr.context_window - P]
            ax[0].loglog(Mv, E_request(pr, P, Mv), ls=ls, color=col,
                         label=f"P_max={P:,}" if reg == "meas" else None)
            D = lambda m: E_request(pr, P, m) - E_request(pr, P, 0)
            ax[1].semilogx(Mv, D(Mv) / D(Mv / 2), ls=ls, color=col)
    ax[0].set_xlabel("output cap M (tokens)"); ax[0].set_ylabel("E_max(P_max, M) (J)")
    ax[0].set_title("(a) E_max, Llama-3.1-8B (solid: b meas, dashed: b arch)")
    ax[0].legend(frameon=False)
    ax[1].axhline(2, color="#888", lw=0.6, ls=":"); ax[1].axhline(4, color="#888", lw=0.6, ls=":")
    ax[1].set_xlabel("output cap M (tokens)"); ax[1].set_ylabel("D(M) / D(M/2)")
    ax[1].set_title("(b) Halving M: decode part shrinks 2× → 4×")
    save(fig, "fig3_quadratic")


def fig4_5() -> None:
    import e4_summarize
    e4_summarize.summarize()
    shutil.copy(RESULTS / "e4_gamma.png", RESULTS / "fig4_batching.png")
    shutil.copy(RESULTS / "e4_gamma.pdf", RESULTS / "fig4_batching.pdf")
    if (RESULTS / "e5_laptop_fit.png").exists():
        for ext in ("png", "pdf"):
            shutil.copy(RESULTS / f"e5_laptop_fit.{ext}", RESULTS / f"fig5_laptop.{ext}")


def table1() -> None:
    s = pd.read_csv(RESULTS / "e2_sla_windows.csv")
    s = s[s.model == "llama31_8b"]
    t = s.pivot_table(index=["trace", "q"], columns="W_s", values="OPF_W_vs_max").round(1)
    t.columns = [f"W={int(c)}s" for c in t.columns]
    t.to_csv(RESULTS / "table1_opf_sla.csv")
    print(t.to_string())


CLAIMS = [  # T2: claim -> evidence keys in results/numbers.json (PREREG H in brackets)
    ("B1", "Thm 1: closed-form certificate, quadratic in M (H3)", ["e1.max_rel_err", "e1.h3.min_r2", "bounds.Thm1.n_violations"]),
    ("B2", "Uncapped service: certificate/mean amplification (H1-ctx)", ["e2.amp_unc_min", "e2.amp_unc_max"]),
    ("B3", "Thm 3: p-aware budget M_T(p) beats fixed output cap at equal cut",
     ["e2.p99.pareto_better_cells", "e2.p99.pareto_gain_min", "e2.p99.pareto_gain_max", "bounds.Thm3.n_violations"]),
    ("B4", "Tightening max_tokens alone gains little (prompt limit matters)", ["e2.p99.meas.red_vs_max_min", "e2.p99.meas.red_vs_max_max"]),
    ("B5", "Thm 2: adaptive gates save on benign traffic, 0 % worst case (H2)",
     ["e3.entropy.S_benign_max", "e3.max_dev_A", "e3.cells_fail_B", "bounds.Thm2.n_violations"]),
    ("B6", "SLA-window over-provisioning factor", ["e2.sla.azure2024_conv.W3600.q0.99.OPF_vs_max"]),
    ("B7", "Thm 4 (v1.1): batching amplification bounds (H5)",
     ["e4.violations_shap", "e4.corrected.violations_shap", "e4.corrected.violations_prop", "e4.ii_preempt_max.chunked",
      "bounds.Thm4.n_violations"]),
    ("B8", "Laptop: e_dec(c) linear, b > 0 (H6)", ["e5.net.r2", "e5.net.b", "e5.net.b_lo", "e5.net.b_hi", "e5.gross.r2", "e5.gross.b", "e5.time.r2", "e5.P_idle_pre_W", "e5.P_idle_post_W"]),
    ("B9", "Controllability probe: > 20k tokens ungated (H7)", ["probe.max_tokens_reached", "probe.n_pass_20k_ungated"]),
]


def table2() -> None:
    import json
    nb = json.loads((RESULTS / "numbers.json").read_text(encoding="utf-8"))
    N, P = nb["numbers"], nb.get("prereg", {})
    rows = [dict(id=cid, claim=txt, key=k, value=N.get(k, "MISSING")) for cid, txt, keys in CLAIMS for k in keys]
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "table2_claim_evidence.csv", index=False)
    print(t.to_string()); print({h: v.get("pass_") for h, v in P.items()})


if __name__ == "__main__":
    fig1(); fig2(); fig3(); table1(); fig4_5(); table2()
