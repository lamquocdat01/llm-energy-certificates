"""T08 / E1: Thm 1 closed form vs token-by-token simulation on a (P_max, M, a, b) grid; quadratic fit (H3).

Simulation accumulates per-token energies (prefill increments then decode increments e(p+t)) with
np.cumsum; the closed form is energy_model.E_max. Grid: 5 P_max x 4 a x 5 b x 2 alpha/a = 200 curves,
each evaluated at 24 values of M -> 4800 cells (>= 500 required).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS
from energy_model import EnergyParams, E_max

P_GRID = [0, 128, 1024, 8192, 32768]
A_GRID = [0.045, 0.09, 0.21, 0.54]           # J/token, spans the 3 models (energy_params.json)
B_GRID = [0.0, 1e-6, 4.7e-6, 5e-5, 3e-4]     # J/token/ctx-token, spans regimes meas..t4fit
R_PRE = [0.295, 0.543]                       # alpha / a (ML.ENERGY Table 4, 8B and 70B)
M_GRID = np.unique(np.round(np.geomspace(1, 16384, 24)).astype(int))


def simulate_curve(prm: EnergyParams, P: int, M_max: int) -> np.ndarray:
    """Token-level accumulation: returns E(P, m) for m = 0..M_max."""
    pre = np.cumsum(prm.alpha + prm.beta * (2 * np.arange(1, P + 1) - 1))[-1] if P > 0 else 0.0
    dec = prm.a + prm.b * (P + np.arange(1, M_max + 1, dtype=float))
    return pre + np.concatenate([[0.0], np.cumsum(dec)])


def main() -> None:
    rows, fits = [], []
    for P in P_GRID:
        for a in A_GRID:
            for b in B_GRID:
                for r in R_PRE:
                    prm = EnergyParams("grid", a=a, b=b, alpha=r * a)
                    sim = simulate_curve(prm, P, int(M_GRID.max()))
                    cf = E_max(prm, P, M_GRID.astype(float))
                    s = sim[M_GRID]
                    rel = np.abs(cf - s) / np.maximum(np.abs(s), 1e-300)
                    for M, x, y, e in zip(M_GRID, s, cf, rel):
                        rows.append(dict(P_max=P, a=a, b=b, alpha=r * a, M=int(M), E_sim=x, E_closed=y, rel_err=e))
                    # H3: quadratic fit of the simulated E_max(M) on the dense grid 0..16384
                    Ms = np.arange(0, int(M_GRID.max()) + 1, 16, dtype=float)
                    y = sim[Ms.astype(int)]
                    c = np.polyfit(Ms, y, 2)
                    yhat = np.polyval(c, Ms)
                    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
                    c1 = np.polyfit(Ms, y, 1)
                    r2_lin = 1 - np.sum((y - np.polyval(c1, Ms)) ** 2) / np.sum((y - y.mean()) ** 2)
                    half = (E_max(prm, P, 16384) - E_max(prm, P, 0)) / (E_max(prm, P, 8192) - E_max(prm, P, 0))
                    fits.append(dict(P_max=P, a=a, b=b, alpha=r * a, r2_quad=r2, r2_lin=r2_lin, c2=c[0],
                                     c2_rel_err=abs(c[0] - b / 2) / (b / 2) if b > 0 else abs(c[0]),
                                     decode_ratio_M_vs_M2=float(half)))
    df, fit = pd.DataFrame(rows), pd.DataFrame(fits)
    df.to_csv(RESULTS / "e1_closed_vs_sim.csv", index=False)
    fit.to_csv(RESULTS / "e1_quadratic_fit.csv", index=False)
    bp = fit[fit.b > 0]
    print(f"cells={len(df)} max_rel_err={df.rel_err.max():.3e}")
    print(f"H3: curves with b>0: {len(bp)}; min R2_quad={bp.r2_quad.min():.8f}; max |c2-b/2|/(b/2)={bp.c2_rel_err.max():.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for b in B_GRID:
        prm = EnergyParams("x", a=0.09, b=b, alpha=0.0266)
        M = np.arange(0, 16385, 64)
        axes[0].plot(M, E_max(prm, 1024, M) / 1e3, label=f"b={b:g}")
    axes[0].set_xlabel("output cap M (tokens)")
    axes[0].set_ylabel("E_max(P_max=1024, M) (kJ)")
    axes[0].set_title("Thm 1: E_max is quadratic in M (a=0.09 J/tok)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for P in P_GRID:
        d = fit[(fit.P_max == P) & (fit.a == 0.09) & (fit.alpha == 0.295 * 0.09)]
        axes[1].semilogx(np.maximum(d.b, 1e-7), d.decode_ratio_M_vs_M2, "o-", label=f"P_max={P}")
    axes[1].axhline(4, color="k", ls=":", lw=0.8)
    axes[1].axhline(2, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("b (J/token/ctx-token; b=0 drawn at 1e-7)")
    axes[1].set_ylabel("[E_max(M)−E_pre] / [E_max(M/2)−E_pre], M=16384")
    axes[1].set_title("Halving M divides worst-case decode by 2..4")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_fig_quadratic.png", dpi=200)


if __name__ == "__main__":
    main()
