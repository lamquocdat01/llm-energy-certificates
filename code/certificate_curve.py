"""T03 (preliminary) and T09/E2 (--full): certificate price curve on public traces.

For every sub-trace x model x energy regime and every quantile q:
  variant 'l'   : output cap M_q = ceil(Q_q(l)); prompt limit P_max = max prompt in trace (not binding on trace).
                  E_max = E(P_max, M_q) (Thm 1); cut = P(l > M_q); E[E] = mean E(p, min(l, M_q)); OPF = E_max / E[E].
  variant 'tot' : total cap T_q = ceil(Q_q(p + l)); requests with p >= T_q are rejected (0 J, counted as cut),
                  others served min(l, T_q - p). E_max = max_{p+l<=T_q} E(p, l).
Baselines for the E_max reduction factor: 'max' = (P_max, M = max l in trace); 'ctx' = total cap at the
model context window C (no max_tokens: the service only enforces p + l <= C).

--full adds: fine q grid 0.80..0.9999, SLA windows W in {1 min, 1 h, 1 day} (Azure 2024, BurstGPT),
day-block bootstrap CI (5 seeds x B resamples) of cut rate and OPF with caps fixed on the full trace.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ram_guard, RESULTS, SEEDS, TRACE_GROUPS, TRACES, load_trace
from energy_model import E_max, E_max_total, E_pre, E_request, load_params

Q_COARSE = [0.90, 0.95, 0.99, 0.999, 1.0]
Q_FINE = sorted(set(np.round(np.concatenate([np.arange(0.80, 0.99, 0.01), np.arange(0.99, 0.999, 0.001),
                                              [0.999, 0.9995, 0.9999, 1.0]]), 4)))
MODELS = ["llama31_8b", "llama31_70b", "qwen3_8b_reason"]
FAMILY = {s: f for f, subs in TRACE_GROUPS.items() for s in subs}


def cap_at(x: np.ndarray, q: float) -> int:
    return int(x.max()) if q >= 1.0 else int(np.ceil(np.quantile(x, q)))


def served_total(p, l, T):
    m = np.clip(T - p, 0, None)
    return np.minimum(l, m), p >= T


def m_budget(prm, p, T):
    """Thm 3 p-aware cap: M_T(p) = max{m >= 0 : E(p, m) <= T}; -1 means rejected (E_pre(p) > T)."""
    p = np.asarray(p, float)
    c = E_pre(prm, p) - T
    lin = prm.a + prm.b * p + prm.b / 2.0
    if prm.b > 0:
        disc = lin * lin - 2.0 * prm.b * c
        m = (-lin + np.sqrt(np.maximum(disc, 0.0))) / prm.b
    else:
        m = -c / lin
    m = np.floor(m + 1e-9)
    return np.where(c > 0, -1.0, m)


def pareto_at_cut(prm, p, l, target_cut, e_sorted=None):
    """Smallest certificate T whose p-aware cap M_T(p) truncates at most target_cut of requests.

    M_T(p) >= l  <=>  E(p, l) <= T, so cut(T) = P(E(p, l) > T) and the optimum is the empirical
    (1 - target_cut)-quantile of the per-request energy E(p, l) (order statistic, 'higher' rule)."""
    e = np.sort(E_request(prm, p, l)) if e_sorted is None else e_sorted
    k = int(np.ceil((1.0 - target_cut) * len(e))) - 1
    k = min(max(k, 0), len(e) - 1)
    return float(e[k])


def curve_rows(name: str, df: pd.DataFrame, prm, model: str, regime: str, qs, pareto: bool = False) -> list[dict]:
    p, l = df["p"].to_numpy(), df["l"].to_numpy()
    tot = p + l
    P_max, L_max = int(p.max()), int(l.max())
    C = prm.context_window
    E_base_max = float(E_max(prm, P_max, L_max))
    E_base_ctx = float(E_max_total(prm, C))
    E_uncapped_mean = float(E_request(prm, p, l).mean())
    rows = []
    e_sorted = np.sort(E_request(prm, p, l)) if pareto else None
    for q in qs:
        M = cap_at(l, q)
        Em = float(E_max(prm, P_max, M))
        EE = float(E_request(prm, p, np.minimum(l, M)).mean())
        rows.append(dict(trace=name, family=FAMILY[name], model=model, regime=regime, variant="l", q=q, cap=M,
                         P_max=P_max, cut=float((l > M).mean()), E_max=Em, E_mean=EE, OPF=Em / EE,
                         red_vs_max=E_base_max / Em, red_vs_ctx=E_base_ctx / Em, E_mean_uncapped=E_uncapped_mean,
                         E_base_max=E_base_max, E_base_ctx=E_base_ctx))
        T = cap_at(tot, q)
        m, rej = served_total(p, l, T)
        Et = float(E_max_total(prm, T))
        EEt = float(np.where(rej, 0.0, E_request(prm, p, m)).mean())
        rows.append(dict(trace=name, family=FAMILY[name], model=model, regime=regime, variant="tot", q=q, cap=T,
                         P_max=P_max, cut=float((tot > T).mean()), E_max=Et, E_mean=EEt, OPF=Et / EEt,
                         red_vs_max=E_base_max / Et, red_vs_ctx=E_base_ctx / Et, E_mean_uncapped=E_uncapped_mean,
                         E_base_max=E_base_max, E_base_ctx=E_base_ctx, frac_rejected=float(rej.mean())))
        if pareto:
            cut_l = float((l > M).mean())
            Tp = pareto_at_cut(prm, p, l, cut_l, e_sorted)
            mT = m_budget(prm, p, Tp)
            rejp = mT < 0
            EEp = float(np.where(rejp, 0.0, E_request(prm, p, np.minimum(l, np.maximum(mT, 0)))).mean())
            rows.append(dict(trace=name, family=FAMILY[name], model=model, regime=regime, variant="pareto", q=q,
                             cap=np.nan, P_max=P_max, cut=float((rejp | (l > mT)).mean()), E_max=Tp, E_mean=EEp,
                             OPF=Tp / EEp, red_vs_max=E_base_max / Tp, red_vs_ctx=E_base_ctx / Tp,
                             E_mean_uncapped=E_uncapped_mean, E_base_max=E_base_max, E_base_ctx=E_base_ctx,
                             frac_rejected=float(rejp.mean())))
    return rows


def block_index(name: str, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    span = df["t"].max() - df["t"].min()
    if span > 2 * 86400:
        idx = np.floor(df["t"].to_numpy() / 86400).astype(np.int64)
        if "part" in df:
            idx = idx + 100000 * df["part"].to_numpy().astype(np.int64)
        return idx, "day"
    return np.floor((df["t"].to_numpy() - df["t"].min()) / 300).astype(np.int64), "5min"


def bootstrap(name, df, prm, model, qs, n_boot: int) -> list[dict]:
    """Caps fixed on the full trace; resample blocks (days) with replacement; 5 seeds."""
    p, l = df["p"].to_numpy(), df["l"].to_numpy()
    blk, kind = block_index(name, df)
    ub, inv = np.unique(blk, return_inverse=True)
    nb = len(ub)
    P_max = int(p.max())
    out = []
    for q in qs:
        M = cap_at(l, q)
        e = E_request(prm, p, np.minimum(l, M))
        n_b = np.bincount(inv, minlength=nb).astype(float)
        cut_b = np.bincount(inv, weights=(l > M).astype(float), minlength=nb)
        e_b = np.bincount(inv, weights=e, minlength=nb)
        Em = float(E_max(prm, P_max, M))
        for s in SEEDS:
            rng = np.random.default_rng(s)
            idx = rng.integers(0, nb, size=(n_boot, nb))
            N = n_b[idx].sum(1)
            cut = cut_b[idx].sum(1) / N
            opf = Em / (e_b[idx].sum(1) / N)
            out.append(dict(trace=name, model=model, q=q, cap=M, seed=s, block=kind, n_blocks=nb,
                            cut_lo=np.quantile(cut, 0.025), cut_med=np.median(cut), cut_hi=np.quantile(cut, 0.975),
                            opf_lo=np.quantile(opf, 0.025), opf_med=np.median(opf), opf_hi=np.quantile(opf, 0.975),
                            cut_max_block=float((cut_b / np.maximum(n_b, 1)).max())))
    return out


def sla_windows(name, df, prm, model, qs) -> list[dict]:
    t = df["t"].to_numpy() - df["t"].min()
    if "part" in df:  # BurstGPT parts have independent clocks: offset them so windows never mix parts
        t = t + df["part"].to_numpy() * 1e9
    p, l = df["p"].to_numpy(), df["l"].to_numpy()
    P_max = int(p.max())
    out = []
    for W in (60, 3600, 86400):
        w = np.floor(t / W).astype(np.int64)
        uw, inv = np.unique(w, return_inverse=True)
        cnt = np.bincount(inv)
        N_max = int(cnt.max())
        for q in qs:
            M = cap_at(l, q)
            e_w = np.bincount(inv, weights=E_request(prm, p, np.minimum(l, M)))
            Em = float(E_max(prm, P_max, M))
            cert = N_max * Em  # E_W <= lambda_max W E_max (+ E_idle W, idle not modelled -> 0)
            out.append(dict(trace=name, model=model, W_s=W, q=q, cap=M, n_windows=len(uw), N_max=N_max,
                            E_W_cert=cert, E_W_max_obs=float(e_w.max()), E_W_mean_obs=float(e_w.mean()),
                            OPF_W_vs_max=cert / e_w.max(), OPF_W_vs_mean=cert / e_w.mean()))
    return out


def fig_curve(df: pd.DataFrame, path, regime="meas", variant="l"):
    fams = list(TRACE_GROUPS)
    fig, axes2 = plt.subplots(2, len(MODELS), figsize=(13, 7.4), sharey="row")
    colors = {"Azure2023": "#1f77b4", "Azure2024": "#ff7f0e", "BurstGPT": "#2ca02c", "Mooncake": "#d62728"}
    for row, base in enumerate(("red_vs_ctx", "red_vs_max")):
      axes = axes2[row]
      for ax, model in zip(axes, MODELS):
        for name in TRACES:
            d = df[(df.model == model) & (df.regime == regime) & (df.variant == variant) & (df.trace == name)]
            if d.empty:
                continue
            d = d.sort_values("cut")
            ls = "-" if TRACE_GROUPS[FAMILY[name]].index(name) == 0 else "--"
            ax.plot(np.maximum(d["cut"], 1e-5) * 100, d[base], ls, marker=".", color=colors[FAMILY[name]],
                    label=name, lw=1.3)
        ax.axhline(5, color="k", lw=0.8, ls=":")
        ax.axvline(1, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("requests truncated (%)")
        ax.set_title(model if row == 0 else "")
        ax.grid(True, which="both", alpha=0.3)
      axes[0].set_ylabel({"red_vs_ctx": "E_max reduction vs no cap\n(p+ℓ ≤ context window) (×)",
                          "red_vs_max": "E_max reduction vs\nM = max ℓ in trace (×)"}[base])
    axes2[0][0].legend(fontsize=7)
    fig.suptitle(f"Certificate price curve (variant '{variant}', regime '{regime}')")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    params = load_params(RESULTS / "energy_params.json")
    regimes = ["meas", "arch", "t4fit"]
    qs = Q_FINE if args.full else Q_COARSE
    rows, boots, slas = [], [], []
    for name in TRACES:
        df = (ram_guard(), load_trace(name))[1]
        for model in MODELS:
            for reg in regimes:
                rows += curve_rows(name, df, params[f"{model}|{reg}"], model, reg, qs, pareto=True)
            if args.full:
                prm = params[f"{model}|meas"]
                boots += bootstrap(name, df, prm, model, [0.99, 0.999], args.n_boot)
                if FAMILY[name] in ("Azure2024", "BurstGPT"):
                    slas += sla_windows(name, df, prm, model, [0.99, 0.999, 1.0])
        print("done", name, flush=True)
    out = pd.DataFrame(rows)
    tag = "e2_cert_curve" if args.full else "cert_curve"
    out.to_csv(RESULTS / f"{tag}.csv", index=False)
    if args.full:
        pd.DataFrame(boots).to_csv(RESULTS / "e2_bootstrap.csv", index=False)
        pd.DataFrame(slas).to_csv(RESULTS / "e2_sla_windows.csv", index=False)
        fig_curve(out, RESULTS / "fig_cert_curve_full.png", "meas", "l")
        fig_curve(out, RESULTS / "e2_fig_cert_curve_tot.png", "meas", "tot")
    else:
        fig_curve(out, RESULTS / "fig_cert_curve.png", "meas", "l")
    # CP-B0 / H1 summary at q = 0.99, variant l
    s = out[(out.variant == "l") & (out.q == 0.99)][["trace", "family", "model", "regime", "cap", "P_max", "cut",
                                                       "red_vs_max", "red_vs_ctx", "OPF"]]
    s.to_csv(RESULTS / f"{tag}_p99_summary.csv", index=False)
    print(s[s.regime == "meas"].to_string(index=False))


if __name__ == "__main__":
    main()
