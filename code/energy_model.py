"""Energy model for autoregressive LLM decoding (topic 38, T02 / Thm 1).

Request r = (p prompt tokens, l output tokens). Energy attributed to r when served alone:

    E(p, l) = E_pre(p) + sum_{t=1}^{l} e_dec(p + t),
    E_pre(p) = alpha * p + beta * p^2          (prefill; beta = 0 in the main setting)
    e_dec(c) = a + b * c                        (decode step at context length c, KV-read term b)

Closed forms (Thm 1):
    E(p, l)          = alpha p + beta p^2 + a l + b (p l + l (l + 1) / 2)
    E_max(P_max, M)  = E(P_max, M)                               (E is increasing in p and l)
    E_max_tot(T)     = max_{p + l <= T, p >= p_min} E(p, l)       (cap on total tokens)

All functions are vectorised over numpy arrays. Units: joules per token (a, alpha), joules per
token per context token (b), joules per token^2 (beta).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EnergyParams:
    name: str
    a: float          # J per decode token at zero context
    b: float          # J per decode token per context token
    alpha: float      # J per prefill token
    beta: float = 0.0  # J per prefill token^2 (attention term of prefill)
    context_window: int = 131072

    def as_dict(self) -> dict:
        return asdict(self)


def E_pre(prm: EnergyParams, p):
    p = np.asarray(p, dtype=float)
    return prm.alpha * p + prm.beta * p * p


def e_dec(prm: EnergyParams, ctx):
    return prm.a + prm.b * np.asarray(ctx, dtype=float)


def E_decode(prm: EnergyParams, p, l):
    """sum_{t=1}^{l} e_dec(p + t) in closed form."""
    p = np.asarray(p, dtype=float)
    l = np.asarray(l, dtype=float)
    return prm.a * l + prm.b * (p * l + l * (l + 1.0) / 2.0)


def E_request(prm: EnergyParams, p, l):
    return E_pre(prm, p) + E_decode(prm, p, l)


def E_max(prm: EnergyParams, P_max, M):
    """Worst-case energy per request under limiter (P_max, M): closed form, quadratic in M."""
    return E_request(prm, P_max, M)


def E_max_total(prm: EnergyParams, T, p_min: int = 1):
    """Worst case under a total-token cap p + l <= T (and p >= p_min).

    f(l) = E(T - l, l) is a concave quadratic in l when beta <= b/2; we maximise it exactly over the
    integer range l in [0, T - p_min] by checking the vertex neighbours and the end points.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    out = np.empty_like(T)
    for i, Ti in enumerate(T):
        lo, hi = 0.0, max(Ti - p_min, 0.0)
        # f(l) = alpha (T-l) + beta (T-l)^2 + a l + b ((T-l) l + l(l+1)/2)
        # f'(l) = -alpha - 2 beta (T-l) + a + b (T - l + 1/2) ; quadratic coefficient: beta - b/2
        qa = prm.beta - prm.b / 2.0
        cands = [lo, hi]
        if qa < 0:
            lin = -prm.alpha - 2 * prm.beta * Ti + prm.a + prm.b * (Ti + 0.5)
            v = -lin / (2 * qa)
            for c in (np.floor(v), np.ceil(v)):
                if lo <= c <= hi:
                    cands.append(c)
        cands = np.array(cands)
        out[i] = np.max(E_request(prm, Ti - cands, cands))
    return out if out.size > 1 else float(out[0])


def argmax_total(prm: EnergyParams, T: int, p_min: int = 1):
    ls = np.arange(0, T - p_min + 1)
    vals = E_request(prm, T - ls, ls)
    k = int(np.argmax(vals))
    return int(T - ls[k]), int(ls[k]), float(vals[k])


def simulate_request(prm: EnergyParams, p: int, l: int) -> float:
    """Token-by-token accumulation (reference implementation used to check the closed form)."""
    e = 0.0
    for i in range(1, p + 1):  # prefill as per-token increments: alpha + beta (2i - 1) sums to alpha p + beta p^2
        e += prm.alpha + prm.beta * (2 * i - 1)
    for t in range(1, l + 1):
        e += prm.a + prm.b * (p + t)
    return e


def load_params(path: Path | str) -> dict[str, EnergyParams]:
    """Load results/energy_params.json -> {f"{model}|{regime}": EnergyParams}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for m in d["models"]:
        for reg, r in m["regimes"].items():
            key = f"{m['key']}|{reg}"
            out[key] = EnergyParams(name=key, a=r["a"], b=r["b"], alpha=m["alpha"], beta=m.get("beta", 0.0),
                                    context_window=m["context_window"])
    return out
