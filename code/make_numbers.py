"""T12 (C6): collect every number that may be printed in the paper -> results/numbers.json + submission_package/numbers.tex.

Keys are stable identifiers; values are computed from results/*.csv|json only (never typed by hand).
Also evaluates PREREG H1-H4 and writes the verdicts into numbers.json["prereg"].
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import RESULTS, TOPIC_DIR

MAIN = "llama31_8b"
MODELS = ["llama31_8b", "llama31_70b", "qwen3_8b_reason"]


def fam_pass(df: pd.DataFrame, ok_col: str) -> tuple[bool, list[str]]:
    """>= 2 sub-traces from >= 2 families where ok_col holds for every model."""
    g = df.groupby(["trace", "family"])[ok_col].all().reset_index()
    good = g[g[ok_col]]
    return (len(good) >= 2 and good.family.nunique() >= 2), sorted(good.trace.tolist())


def main() -> None:
    N: dict = {}
    P: dict = {}
    ts = pd.read_csv(RESULTS / "trace_summary.csv")
    for r in ts[ts.subset == "all"].itertuples():
        N[f"trace.{r.trace}.n"] = int(r.n_requests)
        N[f"trace.{r.trace}.out_p99"] = float(r.output_p99)
        N[f"trace.{r.trace}.out_max"] = int(r.output_max)
        N[f"trace.{r.trace}.prompt_max"] = int(r.prompt_max)
    ep = json.loads((RESULTS / "energy_params.json").read_text(encoding="utf-8"))
    for m in ep["models"]:
        N[f"energy.{m['key']}.a_ref"] = m["a_ref_J_per_tok"]
        N[f"energy.{m['key']}.alpha"] = m["alpha"]
        for reg, v in m["regimes"].items():
            N[f"energy.{m['key']}.{reg}.a"] = v["a"]
            N[f"energy.{m['key']}.{reg}.b"] = v["b"]
            N[f"energy.{m['key']}.{reg}.kappa"] = v["kappa"]
        N[f"energy.{m['key']}.tpb_r2"] = m["tpb_fit"]["r2"]
    N["energy.t4fit_8b.r2"] = ep["t4_fit_8b"]["r2"]
    N["energy.t4fit_8b.n"] = ep["t4_fit_8b"]["n"]

    # E1 / H3
    e1 = pd.read_csv(RESULTS / "e1_closed_vs_sim.csv")
    fit = pd.read_csv(RESULTS / "e1_quadratic_fit.csv")
    bp = fit[fit.b > 0]
    N["e1.cells"] = int(len(e1))
    N["e1.max_rel_err"] = float(e1.rel_err.max())
    N["e1.h3.curves"] = int(len(bp))
    N["e1.h3.min_r2"] = float(bp.r2_quad.min())
    N["e1.h3.max_c2_rel_err"] = float(bp.c2_rel_err.max())
    N["e1.halving_ratio_min"] = float(fit.decode_ratio_M_vs_M2.min())
    N["e1.halving_ratio_max"] = float(fit.decode_ratio_M_vs_M2.max())
    P["H3"] = dict(pass_=bool(bp.r2_quad.min() >= 0.999 and bp.c2_rel_err.max() <= 1e-6),
                   detail=f"min R2={bp.r2_quad.min():.10f}, max |c2-b/2|/(b/2)={bp.c2_rel_err.max():.2e}, curves={len(bp)}, cells={len(e1)}")

    # E2 / H1, H4
    e2 = pd.read_csv(RESULTS / "e2_cert_curve.csv")
    m99 = e2[(e2.regime == "meas") & (np.isclose(e2.q, 0.99))]
    l99 = m99[m99.variant == "l"].copy()
    t99 = m99[m99.variant == "tot"].set_index(["trace", "model"])
    p99 = m99[m99.variant == "pareto"].set_index(["trace", "model"])
    l99["ok_ctx"] = (l99.red_vs_ctx >= 5) & (l99.cut <= 0.01)
    l99["ok_max"] = (l99.red_vs_max >= 5) & (l99.cut <= 0.01)
    h1c, tr_c = fam_pass(l99, "ok_ctx")
    h1m, tr_m = fam_pass(l99, "ok_max")
    P["H1"] = dict(pass_=bool(h1c or h1m), pass_ctx=bool(h1c), pass_max=bool(h1m), traces_ctx=tr_c, traces_max=tr_m,
                   detail=f"red_vs_ctx {l99.red_vs_ctx.min():.1f}-{l99.red_vs_ctx.max():.1f}x; red_vs_max {l99.red_vs_max.min():.2f}-{l99.red_vs_max.max():.2f}x; cut {l99.cut.min():.4f}-{l99.cut.max():.4f}")
    for r in l99.itertuples():
        k = f"e2.p99.{r.trace}.{r.model}"
        N[k + ".cut"] = r.cut
        N[k + ".red_vs_ctx"] = r.red_vs_ctx
        N[k + ".red_vs_max"] = r.red_vs_max
        N[k + ".OPF"] = r.OPF
        N[k + ".E_max_l"] = r.E_max
        N[k + ".E_max_tot"] = float(t99.loc[(r.trace, r.model), "E_max"])
        N[k + ".E_max_pareto"] = float(p99.loc[(r.trace, r.model), "E_max"])
        N[k + ".pareto_gain"] = r.E_max / float(p99.loc[(r.trace, r.model), "E_max"])
        N[k + ".amp_unc"] = r.E_base_ctx / r.E_mean_uncapped
    l99["tot_better"] = [float(t99.loc[(a, b), "E_max"]) < e for a, b, e in zip(l99.trace, l99.model, l99.E_max)]
    g = l99.groupby(["trace", "family"]).tot_better.sum().reset_index()
    good = g[g.tot_better >= 2]
    h4 = len(good) >= 2 and good.family.nunique() >= 2
    P["H4"] = dict(pass_=bool(h4), traces=sorted(good.trace.tolist()),
                   detail=f"sub-traces where tot < l for >=2/3 models: {sorted(good.trace.tolist())}")
    l99["pareto_better"] = [float(p99.loc[(a, b), "E_max"]) <= e for a, b, e in zip(l99.trace, l99.model, l99.E_max)]
    N["e2.p99.pareto_better_cells"] = int(l99.pareto_better.sum())
    N["e2.p99.cells"] = int(len(l99))
    N["e2.p99.pareto_gain_min"] = float(min(N[k] for k in N if k.endswith(".pareto_gain")))
    N["e2.p99.pareto_gain_max"] = float(max(N[k] for k in N if k.endswith(".pareto_gain")))
    amp = [N[k] for k in N if k.endswith(".amp_unc")]
    N["e2.amp_unc_min"], N["e2.amp_unc_max"] = float(min(amp)), float(max(amp))
    # sensitivity regimes
    for reg in ("meas", "arch", "t4fit"):
        s = e2[(e2.regime == reg) & np.isclose(e2.q, 0.99) & (e2.variant == "l")]
        N[f"e2.p99.{reg}.red_vs_max_max"] = float(s.red_vs_max.max())
        N[f"e2.p99.{reg}.red_vs_max_min"] = float(s.red_vs_max.min())
        N[f"e2.p99.{reg}.red_vs_ctx_min"] = float(s.red_vs_ctx.min())
    bs = pd.read_csv(RESULTS / "e2_bootstrap.csv")
    b99 = bs[(np.isclose(bs.q, 0.99)) & (bs.model == MAIN)]
    for tr, g2 in b99.groupby("trace"):
        N[f"e2.boot.{tr}.cut_lo"] = float(g2.cut_lo.min())
        N[f"e2.boot.{tr}.cut_hi"] = float(g2.cut_hi.max())
        N[f"e2.boot.{tr}.opf_lo"] = float(g2.opf_lo.min())
        N[f"e2.boot.{tr}.opf_hi"] = float(g2.opf_hi.max())
        N[f"e2.boot.{tr}.cut_max_block"] = float(g2.cut_max_block.max())
    sla = pd.read_csv(RESULTS / "e2_sla_windows.csv")
    for r in sla[(sla.model == MAIN)].itertuples():
        N[f"e2.sla.{r.trace}.W{r.W_s}.q{r.q}.OPF_vs_max"] = r.OPF_W_vs_max
        N[f"e2.sla.{r.trace}.W{r.W_s}.q{r.q}.OPF_vs_mean"] = r.OPF_W_vs_mean

    # E3 / H2
    e3 = pd.read_csv(RESULTS / "e3_adaptive_vs_cap.csv")
    for r in e3[e3.model == MAIN].itertuples():
        k = f"e3.{r.trace}.{r.cap}.{r.gate}"
        N[k + ".S_benign"] = r.S_benign_mean
        N[k + ".S_wc_B"] = r.S_wc_B
    em = e3[e3.model == MAIN]
    for gname, g3 in em.groupby("gate"):
        N[f"e3.{gname}.S_benign_min"] = float(g3.S_benign_mean.min())
        N[f"e3.{gname}.S_benign_max"] = float(g3.S_benign_mean.max())
    N["e3.cells"] = int(len(e3))
    N["e3.max_dev_A"] = float(e3.dev_A.max())
    N["e3.cells_fail_B"] = int((e3.dev_B > 1e-6).sum())
    fail = e3[e3.dev_B > 1e-6]
    N["e3.cells_fail_B_main"] = int((em.dev_B > 1e-6).sum())
    N["e3.cells_main"] = int(len(em))
    N["e3.fail_B.M_min"] = float(fail.M.min()) if len(fail) else 0.0      # P3: caps of the 18 H2 cells
    N["e3.fail_B.M_max"] = float(fail.M.max()) if len(fail) else 0.0
    N["e3.fail_B.reach_max"] = float(fail.reach_B_g.max()) if len(fail) else 0.0
    P["H2"] = dict(pass_=bool(N["e3.cells_fail_B"] == 0), max_dev_A=N["e3.max_dev_A"],
                   failing_cells=sorted({(a, b, c) for a, b, c in zip(fail.cap, fail.gate, fail.model)}),
                   detail=f"(A) max dev {N['e3.max_dev_A']:.1e}; (B) {N['e3.cells_fail_B']}/{len(e3)} cells with dev>1e-6")
    P["H2"]["failing_cells"] = [list(x) for x in P["H2"]["failing_cells"]]

    # E4 / H5 (PREREG v1.1): e4_summarize.py output
    h5p = RESULTS / "e4_h5.json"
    if h5p.exists():
        h5 = json.loads(h5p.read_text(encoding="utf-8"))
        g = pd.concat([pd.read_csv(f) for f in sorted(RESULTS.glob("e4_grid_*.csv"))], ignore_index=True)
        for k in ("cells_total", "cells_checked", "victim_checks", "violations_shap", "violations_prop"):
            N[f"e4.{k}"] = int(h5[k])
        N["e4.max_ratio_to_bound_shap"] = h5["max_ratio_to_bound_shap"]
        for rk, v in h5["violations_by_rule"].items():
            N[f"e4.viol.{rk}.shap"], N[f"e4.viol.{rk}.prop"] = int(v["shap"]), int(v["prop"])
        for rk, v in h5["gamma_shap_max_main"].items():
            N[f"e4.gamma_shap_max.{rk}"] = v
        for rk, v in h5["gamma_prop_max_main"].items():
            N[f"e4.gamma_prop_max.{rk}"] = v
        for c, v in h5["ii_preempt_max"].items():
            N[f"e4.ii_preempt_max.{c}"] = int(v)
        N["e4.stalled_cells_iii_chunked"] = int(g[(g.rule == "iii") & g.chunked & (g.regime == "meas")
                                                  & (g.a1f == 0.1)].stalled.sum())
        N["e4.stalled_cells_iii_chunked_benign"] = int(g[(g.rule == "iii") & g.chunked & (g.regime == "meas")
                                                         & (g.a1f == 0.1) & (g.n_adv == 0)].stalled.sum())
        corr = g[~((g.rule == "iii") & g.chunked) & ~((g.rule == "ii") & g.chunked)]
        N["e4.corrected.violations_shap"] = int(corr.viol_shap_corr.sum())
        N["e4.corrected.violations_prop"] = int(corr.viol_prop_corr.sum())
        # P3: denominators and per-trace maxima for the manuscript (main slice regime meas, a1/a = 0.1)
        gm = g[(g.regime == "meas") & (g.a1f == 0.1)]
        N["e4.cells_iii_chunked_benign"] = int(len(gm[(gm.rule == "iii") & gm.chunked & (gm.n_adv == 0)]))
        N["e4.corrected.cells_with_finite_bound"] = int(h5["corrected"]["cells_with_finite_bound"])
        N["e4.ii_preempt_gt_Bm1_cells.whole"] = int(h5["ii_preempt_gt_Bm1_cells"]["whole"])
        for tr, gt in gm.groupby("trace"):
            iic = gt[(gt.rule == "ii") & gt.chunked]
            N[f"e4.{tr}.ii_preempt_max.chunked"] = int(iic.preempt_max.max())
            N[f"e4.{tr}.gamma_shap_max.ii|chunked"] = float(iic.gamma_shap_max.max())
            b3 = gt[(gt.rule == "iii") & gt.chunked & (gt.n_adv == 0)]
            N[f"e4.{tr}.stalled_iii_chunked_benign"] = int(b3.stalled.sum())
            N[f"e4.{tr}.cells_iii_chunked_benign"] = int(len(b3))
        for kk, v in h5["gamma_shap_max_by_nadv"].items():
            N[f"e4.gamma_shap_max_nadv.{kk}"] = v
        for stem in ("e4_livelock", "e4_livelock_B64"):
            for kk, v in h5.get(stem, {}).items():
                if kk.startswith("iii|chunked"):
                    N[f"{stem}.{kk}.gamma_shap"] = v["gamma_shap"]
                    N[f"{stem}.{kk}.preempt"] = int(v["preempt"])
        P["H5"] = dict(pass_=bool(h5["H5_pass"]), violations_by_rule=h5["violations_by_rule"],
                       corrected_violations=dict(shap=N["e4.corrected.violations_shap"],
                                                 prop=N["e4.corrected.violations_prop"]),
                       detail=f"registered bounds: {h5['violations_shap']} Shapley / {h5['violations_prop']} prop "
                              f"violations in {h5['victim_checks']} victim checks; corrected Thm 4 v1.1: "
                              f"{N['e4.corrected.violations_shap']}/{N['e4.corrected.violations_prop']}")

    # E5 / H6 and probe / H7
    e5p = RESULTS / "e5_laptop_fit.json"
    if e5p.exists():
        e5 = json.loads(e5p.read_text(encoding="utf-8"))
        for kind in ("net", "gross"):
            for k in ("a", "b", "r2", "kappa"):
                N[f"e5.{kind}.{k}"] = e5[kind][k]
            N[f"e5.{kind}.b_lo"], N[f"e5.{kind}.b_hi"] = e5[kind]["b_ci"]
            N[f"e5.{kind}.kappa_lo"], N[f"e5.{kind}.kappa_hi"] = e5[kind]["kappa_ci"]
        N["e5.P_idle_W"] = e5["P_idle_W"]
        N["e5.P_idle_pre_W"], N["e5.P_idle_post_W"] = e5["P_idle_pre_post_W"]
        N["e5.cpu_pct_median"] = e5["cpu_pct_median_during_runs"]
        N["e5.n_windows"], N["e5.n_runs"] = int(e5["n_windows"]), int(e5["n_runs"])
        N["e5.p_max_measured"] = int(max(e5["p_grid"]))
        N["e5.prefill.alpha"] = e5["prefill"]["alpha_J_tok"]
        N["e5.prefill.r2"] = e5["prefill"]["r2"]
        N["e5.alpha_over_a"] = e5["alpha_over_a_net"]
        N["e5.time.kappa"] = e5["time"]["kappa"]
        N["e5.time.r2"] = e5["time"]["r2"]
        P["H6"] = dict(pass_=bool(e5["H6_pass"]),
                       detail=f"net R2={e5['net']['r2']:.4f}, b={e5['net']['b']:.3e} J/tok/ctx, "
                              f"95% CI [{e5['net']['b_ci'][0]:.3e}, {e5['net']['b_ci'][1]:.3e}], "
                              f"kappa=b/a={e5['net']['kappa']:.2e}")
    elif (TOPIC_DIR / "P2" / "STOP_E5.md").exists():
        P["H6"] = dict(pass_=False, detail="NOT RUN: P2/STOP_E5.md")
    prp = RESULTS / "e5_probe.csv"
    if prp.exists():
        pr = pd.read_csv(prp)
        for r in pr.itertuples():
            N[f"probe.{r.prompt}.n_tokens"] = int(r.n_tokens)
            N[f"probe.{r.prompt}.mean_entropy"] = r.mean_entropy
            N[f"probe.{r.prompt}.prompt_tokens"] = int(r.prompt_tokens)
            for gate in ("entropy", "loop"):                  # P3: token at which the simulated gate fires (absent: never)
                v = getattr(r, f"{gate}_fire_at")
                if pd.notna(v):
                    N[f"probe.{r.prompt}.{gate}_fire_at"] = int(v)
        N["probe.entropy_fire_max"] = int(pr.entropy_fire_at.max())
        N["probe.entropy_fire_min"] = int(pr.entropy_fire_at.min())
        N["probe.n_prompts"] = int(len(pr))
        N["probe.max_tokens_reached"] = int(pr.n_tokens.max())
        N["probe.n_pass_20k_ungated"] = int(pr.passes_20k_ungated.sum())
        P["H7"] = dict(pass_=bool(pr.passes_20k_ungated.any()),
                       passing=pr[pr.passes_20k_ungated].prompt.tolist(),
                       detail="; ".join(f"{r.prompt}: {r.n_tokens} tok, entropy gate at {r.entropy_fire_at}, "
                                        f"loop gate at {r.loop_fire_at}" for r in pr.itertuples()))

    # bounds
    bc = pd.read_csv(RESULTS / "bounds_check.csv")
    N["bounds.n_checks"] = int(bc.n_checks.sum())
    N["bounds.n_violations"] = int(bc.n_violations.sum())
    for th, g4 in bc.groupby("theorem"):
        N[f"bounds.{th}.n_checks"] = int(g4.n_checks.sum())
        N[f"bounds.{th}.n_violations"] = int(g4.n_violations.sum())

    out = {"generated_from": "results/*.csv|json by code/make_numbers.py", "numbers": N, "prereg": P}
    (RESULTS / "numbers.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # numbers.tex (P3): \num{key} prints the default format; \num{key@d} d decimals; \num{key.pct@d} 100*v with d
    # decimals; \num{key.sci@d} m\times10^{e}. Thousands are separated by a comma (IEEE style).
    def fixed(v: float, d: int) -> str:
        return f"{v:,.{d}f}"

    def sci(v: float, d: int) -> str:
        if v == 0:
            return "0"
        e = int(np.floor(np.log10(abs(v))))
        m = v / 10 ** e
        if round(abs(m), d) >= 10:
            m, e = m / 10, e + 1
        return f"\\ensuremath{{{m:.{d}f}\\times10^{{{e}}}}}"

    def default(v) -> str:
        if isinstance(v, (bool, np.bool_)):
            return str(int(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        v = float(v)
        if v == 0:
            return "0"
        if abs(v) >= 100:
            return fixed(v, 0)
        if abs(v) >= 1:
            return fixed(v, 2)
        if abs(v) >= 1e-2:
            return f"{v:.3f}"
        return sci(v, 1)

    def defs(k: str, v) -> list[str]:
        out_ = [(k, default(v))]
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out_ += [(f"{k}.sci@{d}", sci(float(v), d)) for d in range(3)]
        if isinstance(v, (float, np.floating)) and not isinstance(v, bool):
            v = float(v)
            out_ += [(f"{k}@{d}", fixed(v, d)) for d in range(4)]
            out_ += [(f"{k}.sci@{d}", sci(v, d)) for d in range(3)]
            if abs(v) <= 1.5:
                out_ += [(f"{k}.pct@{d}", fixed(100 * v, d)) for d in range(3)]
        return [f"\\expandafter\\def\\csname nj@{a}\\endcsname{{{b}}}" for a, b in out_]

    lines = ["% AUTO-GENERATED by code/make_numbers.py from results/numbers.json -- do not edit by hand.",
             r"\providecommand{\num}[1]{\ifcsname nj@#1\endcsname\csname nj@#1\endcsname"
             r"\else\errmessage{numbers.tex: unknown key #1}\fi}"]
    for k, v in N.items():
        lines += defs(k, v)
    (TOPIC_DIR / "submission_package").mkdir(exist_ok=True)
    (TOPIC_DIR / "submission_package" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(N)} numbers; PREREG:", {h: v['pass_'] for h, v in P.items()})
    for h, v in P.items():
        print(h, v["detail"])


if __name__ == "__main__":
    main()
