"""T10 / E3: adaptive (content-dependent) gates vs content-independent caps — numerical test of Thm 2.

Design frozen in PREREG_38_v1.md §3:
  traces: azure2023_conv, azure2024_conv, burstgpt, mooncake_conv; caps: Q.99(l), Q.999(l), C - P_max.
  gates (hidden-variable models, seeded): TALE-like budget, entropy early-exit, conformal stop, loop detector.
  benign saving  S_benign = 1 - E[E(g o L)] / E[E(L)] on the trace (5 seeds);
  worst-case saving S_wc = 1 - WC(g o L) / WC(L) under (A) controllable adversary and (B) published reachable set.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ram_guard, RESULTS, SEEDS, load_trace
from energy_model import E_request, load_params

TRACES = ["azure2023_conv", "azure2024_conv", "burstgpt", "mooncake_conv"]
GATES = ["tale", "entropy", "conformal", "loop"]
CAPS = ["q0.99", "q0.999", "ctx"]
DELTA = 0.05

# Attack families of PREREG §3 (B) and the PREREG fires(g, f) rules:
#   loop detector fires only on repetitive families (LoopLLM);
#   TALE budget does NOT fire on difficulty-inflating families (ExtendAttack, OverThink), fires otherwise;
#   entropy / conformal do NOT fire on non-converging families (Deadlock, ThinkTrap), fire otherwise.
FAMILIES = {
    "Engorgio": "Engorgio", "OverThink": "OverThink", "ThinkTrap": "ThinkTrap", "LoopLLM": "LoopLLM",
    "ExtendAttack": "ExtendAttack", "Deadlock": "Deadlock", "ReasoningBomb": "ReasoningBomb",
    "Excessive Reasoning": "ExcessiveReasoning",
}
FIRES = {
    "loop": {f: f == "LoopLLM" for f in FAMILIES.values()},
    "tale": {f: f not in ("ExtendAttack", "OverThink") for f in FAMILIES.values()},
    "entropy": {f: f not in ("Deadlock", "ThinkTrap") for f in FAMILIES.values()},
    "conformal": {f: f not in ("Deadlock", "ThinkTrap") for f in FAMILIES.values()},
}


def _num(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def reachable_set() -> pd.DataFrame:
    """l_f = max_tokens of the original experiment if requests were reported to hit the cap, else the largest
    published attack output length; tau_benign_f = largest benign output length of the family (TALE stop point)."""
    rows = json.loads((RESULTS / "attack_lengths_published.json").read_text(encoding="utf-8"))
    out = []
    for key, fam in FAMILIES.items():
        ents = [e for e in rows if e["paper"].startswith(key)]
        reach, src = [], []
        for e in ents:
            mt, at, fr = _num(e.get("max_tokens_setting")), _num(e.get("attack_out_tokens")), _num(e.get("fraction_hitting_cap"))
            hit = (fr is not None and fr > 0) or (mt is not None and at is not None and at >= mt)
            if hit and mt is not None:
                reach.append(mt); src.append(f"max_tokens={mt:g} hit ({e['location']})")
            elif at is not None:
                reach.append(at); src.append(f"attack_out={at:g} ({e['location']})")
        benign = [_num(e.get("benign_out_tokens")) for e in ents if _num(e.get("benign_out_tokens")) is not None]
        if not reach:
            continue
        k = int(np.argmax(reach))
        out.append(dict(family=fam, l_f=float(reach[k]), source=src[k], n_entries=len(ents),
                        benign_max=float(max(benign)) if benign else np.nan))
    return pd.DataFrame(out)


def gate_stop_adversarial(gate: str, fam: str, l_f: float, benign_max: float) -> float:
    """Stop point of gate g on family f when it fires (worst case over the gate's own randomness)."""
    if not FIRES[gate][fam]:
        return np.inf
    if gate == "loop":
        return np.ceil(0.75 * l_f) + 64            # latest onset of the benign loop model + window
    if gate == "tale":
        return benign_max if np.isfinite(benign_max) else 0.0  # budget estimated as for the benign-looking task
    if gate in ("entropy", "conformal"):
        return np.ceil(1.0 * benign_max) + 32 if np.isfinite(benign_max) else 32.0  # answer ready no later than benign length
    raise ValueError(gate)


def served_benign(gate: str, p, l, M, rng) -> tuple[np.ndarray, np.ndarray]:
    """Returns (served length under g o L, mask of requests used for evaluation)."""
    n = len(l)
    base = np.minimum(l, M)
    mask = np.ones(n, bool)
    if gate == "tale":
        lhat = l * np.exp(0.5 * rng.standard_normal(n))
        s = np.minimum(base, np.ceil(lhat))
    elif gate == "entropy":
        tau = np.ceil(rng.beta(2, 2, n) * l)
        s = np.minimum(base, tau + 32)
    elif gate == "conformal":
        tau = np.ceil(rng.beta(2, 2, n) * l)
        that = np.maximum(tau * np.exp(0.3 * rng.standard_normal(n)), 1e-9)
        cal = rng.random(n) < 0.5
        ratio = np.sort(tau[cal] / that[cal])
        k = min(int(np.ceil((cal.sum() + 1) * (1 - DELTA))) - 1, cal.sum() - 1)
        c = ratio[k]
        s = np.minimum(base, np.ceil(c * that))
        mask = ~cal
    elif gate == "loop":
        thr = np.quantile(l, 0.999)
        runaway = l >= thr
        onset = np.ceil(rng.uniform(0.25, 0.75, n) * l) + 64
        s = np.where(runaway, np.minimum(base, onset), base)
    else:
        raise ValueError(gate)
    return s, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama31_8b,llama31_70b,qwen3_8b_reason")
    args = ap.parse_args()
    params = load_params(RESULTS / "energy_params.json")
    R = reachable_set()
    R.to_csv(RESULTS / "e3_reachable_set.csv", index=False)
    print(R.to_string(index=False))
    rows = []
    for name in TRACES:
        df = (ram_guard(), load_trace(name))[1]
        p, l = df["p"].to_numpy().astype(float), df["l"].to_numpy().astype(float)
        P_max = float(p.max())
        for model in args.models.split(","):
            prm = params[f"{model}|meas"]
            caps = {"q0.99": np.ceil(np.quantile(l, 0.99)), "q0.999": np.ceil(np.quantile(l, 0.999)),
                    "ctx": prm.context_window - P_max}
            for cap, M in caps.items():
                wc_A_L = float(E_request(prm, P_max, M))
                reach_L = R.l_f.max()
                wc_B_L = float(E_request(prm, P_max, min(reach_L, M)))
                for gate in GATES:
                    sav = []
                    for s in SEEDS:
                        rng = np.random.default_rng(s)
                        served, mask = served_benign(gate, p, l, M, rng)
                        e_g = E_request(prm, p[mask], served[mask]).sum()
                        e_L = E_request(prm, p[mask], np.minimum(l[mask], M)).sum()
                        sav.append(1 - e_g / e_L)
                    # (A): controllability -> envelope M_g(P_max) >= M
                    wc_A_g = float(E_request(prm, P_max, min(M, np.inf)))
                    # (B): published reachable set, adversary picks p = P_max and the best family
                    cand = [min(r.l_f, M, gate_stop_adversarial(gate, r.family, r.l_f, r.benign_max)) for r in R.itertuples()]
                    kbest = int(np.argmax(cand))
                    wc_B_g = float(E_request(prm, P_max, cand[kbest]))
                    rows.append(dict(trace=name, model=model, cap=cap, M=M, P_max=P_max, gate=gate,
                                     S_benign_mean=float(np.mean(sav)), S_benign_min=float(np.min(sav)),
                                     S_benign_max=float(np.max(sav)),
                                     WC_A_L=wc_A_L, WC_A_g=wc_A_g, S_wc_A=1 - wc_A_g / wc_A_L,
                                     dev_A=abs(wc_A_g - wc_A_L) / wc_A_L,
                                     WC_B_L=wc_B_L, WC_B_g=wc_B_g, S_wc_B=1 - wc_B_g / wc_B_L,
                                     dev_B=abs(wc_B_g - wc_B_L) / wc_B_L, best_family_B=R.family.iloc[kbest],
                                     reach_B_g=cand[kbest], reach_B_L=min(reach_L, M)))
        print("done", name, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e3_adaptive_vs_cap.csv", index=False)
    main_m = out[out.model == "llama31_8b"]
    tab = main_m.pivot_table(index=["gate", "cap"], columns="trace", values=["S_benign_mean", "S_wc_B"]).round(4)
    tab.to_csv(RESULTS / "e3_table_main.csv")
    print(tab.to_string())
    h2 = out.groupby("model").agg(max_dev_A=("dev_A", "max"), max_dev_B=("dev_B", "max"),
                                  cells=("dev_B", "size"), cells_fail_B=("dev_B", lambda x: int((x > 1e-6).sum())))
    print(h2.to_string())

    # figure: two columns, benign saving vs worst-case saving (main model, mean over traces; bars = range)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    width = 0.25
    for j, cap in enumerate(CAPS):
        d = main_m[main_m.cap == cap].groupby("gate").agg(b=("S_benign_mean", "mean"), blo=("S_benign_min", "min"),
                                                          bhi=("S_benign_max", "max"), w=("S_wc_B", "mean"),
                                                          wlo=("S_wc_B", "min"), whi=("S_wc_B", "max")).reindex(GATES)
        x = np.arange(len(GATES)) + (j - 1) * width
        axes[0].bar(x, 100 * d.b, width, yerr=[100 * (d.b - d.blo), 100 * (d.bhi - d.b)], label=f"cap {cap}", capsize=2)
        axes[1].bar(x, 100 * d.w, width, yerr=[100 * (d.w - d.wlo), 100 * (d.whi - d.w)], label=f"cap {cap}", capsize=2)
    for ax, t in zip(axes, ["Benign traffic: expected energy saving", "Adversarial: worst-case saving (published reach)"]):
        ax.set_xticks(np.arange(len(GATES)))
        ax.set_xticklabels(["TALE-like\nbudget", "entropy\nearly-exit", "conformal\nstop", "loop\ndetector"])
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("saving vs cap alone (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Adaptive gates ∘ cap vs cap (Llama-3.1-8B, mean over 4 traces; whiskers = range over traces × seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_adaptive_vs_cap.png", dpi=200)


if __name__ == "__main__":
    main()
