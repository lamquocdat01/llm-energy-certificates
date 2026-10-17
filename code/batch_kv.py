"""E4 (P2, PREREG v1.1 §2): iteration-level continuous-batching simulator with a KV budget; checks Thm 4.

Model (THEORY_38 §5). Each iteration every decoding request emits one token; request j holds kappa_j = p_j + t_j KV
tokens (plus prompt tokens prefilled so far under chunked prefill). Token t of a request is produced at context
p + t, so a request running alone pays exactly E(p, l) of energy_model.py. Iteration energy
    e_k = A0 * 1[D_k != {}] + sum_{j in D_k} (a1 + b kappa_j) + sum_{prefill chunks} alpha * c.
Preemption = recompute (KV dropped, prompt + generated tokens prefilled again, charged to the preempted request).

Rules: (i) KV reservation  (|R|+1)(P_max+M) <= K, no preemption;
       (ii) on-demand, FCFS-priority recompute, watermark h = B, preempt latest arrival first, requeue by arrival;
       (iii) on-demand, h = 0, preempt largest KV first, requeue at the queue head.
Attribution is linear in (A0, a1, b, alpha), so the simulator only accumulates per-request sufficient statistics
    S1 = sum 1/|D_k|, N = #decode steps, SK = sum kappa_v, SM = sum (sum_{D_k} kappa)/|D_k|, PF = prefilled tokens
and Shapley = A0 S1 + a1 N + b SK + alpha PF, proportional = A0 S1 + a1 N + b SM + alpha PF.

    python batch_kv.py grid        # H5 grid (PREREG v1.1 §2) -> results/e4_grid.csv
    python batch_kv.py livelock    # H5-L construction of Thm 4(c1) -> results/e4_livelock.csv
    python batch_kv.py summarize   # bound check, H5 verdict, figures -> results/e4_*.csv/png/pdf
"""
from __future__ import annotations

import argparse
import bisect
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import RESULTS, SEEDS, load_trace, ram_guard

CHUNK = 512
MAX_ITERS = 2_000_000
STALL = 20_000                             # no completion for this many iterations -> livelock, stop (>= 20x any legit gap)
A1_FRACS = (0.1, 0.5)                      # a1 / a (PREREG: 0.1 main, 0.5 sensitivity)
REGIMES = ("meas", "arch")                 # llama31_8b regimes (meas = main)
TRACES = {"azure2024_conv": dict(P_max=7999, M=694), "burstgpt": dict(P_max=125591, M=844)}


@dataclass
class Req:
    idx: int
    arr: int                 # arrival iteration
    p: int
    l: int                   # served length (already capped at M)
    adv: bool = False
    t: int = 0               # tokens generated so far
    kv: int = 0              # KV tokens currently held
    pre_left: int = 0        # prompt+generated tokens still to (re)prefill
    S1: float = 0.0
    N: int = 0
    SK: float = 0.0
    SM: float = 0.0
    PF: int = 0
    n_pre: int = 0           # preemptions
    n_adm: int = 0           # admissions (first + re-admissions)
    first_adm: int = -1
    done_at: int = -1
    oth_adm: int = 0         # admissions of other requests between this one's first admission and its completion
    g0: int = 0              # global admission count right after the first admission


@dataclass
class Sim:
    rule: str                # "i" | "ii" | "iii"
    B: int
    K: int
    P_max: int
    M: int
    chunked: bool
    reqs: list = field(default_factory=list)

    def run(self, max_iters: int = MAX_ITERS, stop_when=None) -> int:
        reqs = sorted(self.reqs, key=lambda r: (r.arr, r.idx))
        nxt, waiting, running = 0, [], []
        G, last_done = 0, 0
        self.stalled = False
        h = self.B if self.rule == "ii" else 0
        res_cap = self.P_max + self.M
        k = 0
        while k < max_iters:
            while nxt < len(reqs) and reqs[nxt].arr <= k:
                r = reqs[nxt]; nxt += 1
                r.pre_left = r.p
                self._enqueue(waiting, r, arrival=True)
            if not running and not waiting:
                if nxt >= len(reqs):
                    break
                k = reqs[nxt].arr
                continue
            # 1. growth of running requests in this iteration and preemption if KV would overflow
            if self.rule != "i":
                while running and sum(r.kv + self._step(r) for r in running) > self.K:
                    v = (max(running, key=lambda r: (r.arr, r.idx)) if self.rule == "ii"
                         else max(running, key=lambda r: (r.kv, r.arr)))
                    running.remove(v)
                    v.kv, v.n_pre, v.pre_left = 0, v.n_pre + 1, v.p + v.t
                    self._enqueue(waiting, v, arrival=False)
            # 2. head-of-line FCFS admission
            used = sum(r.kv + self._step(r) for r in running)
            while waiting and len(running) < self.B:
                j = waiting[0][1]
                if self.rule == "i":
                    ok = (len(running) + 1) * res_cap <= self.K
                else:                           # watermark only matters when something else is running
                    ok = used + self._step(j) + (h if running else 0) <= self.K
                if not ok:
                    break
                waiting.pop(0)
                used += self._step(j)
                j.n_adm += 1
                G += 1
                if j.first_adm < 0:
                    j.first_adm, j.g0 = k, G
                running.append(j)
            # 3. execute the iteration
            dec = []
            for r in running:
                if r.pre_left > 0:
                    c = min(CHUNK, r.pre_left) if self.chunked else r.pre_left
                    r.pre_left -= c; r.kv += c; r.PF += c
                    if r.pre_left > 0:
                        continue
                dec.append(r)
            if dec:
                for r in dec:
                    r.t += 1; r.kv = r.p + r.t
                n = len(dec)
                tot = float(sum(r.kv for r in dec))
                for r in dec:
                    r.S1 += 1.0 / n; r.N += 1; r.SK += r.kv; r.SM += tot / n
                fin = [r for r in dec if r.t >= r.l]
                for r in fin:
                    running.remove(r); r.done_at = k; r.kv = 0
                    r.oth_adm = G - r.g0 - (r.n_adm - 1)
                    last_done = k
            k += 1
            if k - last_done > STALL and (running or waiting):
                self.stalled = True
                break
            if stop_when is not None and stop_when(k):
                break
        for r in reqs:
            if r.done_at < 0 and r.first_adm >= 0:
                r.oth_adm = G - r.g0 - (r.n_adm - 1)
        return k

    def _step(self, r: Req) -> int:
        """KV tokens r needs this iteration on top of what it holds."""
        if r.pre_left > 0:
            c = min(CHUNK, r.pre_left) if self.chunked else r.pre_left
            return c + (1 if c == r.pre_left else 0)
        return 1

    def _enqueue(self, waiting: list, r: Req, arrival: bool) -> None:
        if arrival or self.rule == "ii":
            bisect.insort(waiting, ((r.arr, r.idx), r), key=lambda x: x[0])
        else:                                   # rule iii: preempted request goes back to the queue head
            waiting.insert(0, ((-1, -1), r))


# ------------------------------------------------------------------------------------------------ energy
def energy_params() -> dict:
    d = json.loads((RESULTS / "energy_params.json").read_text(encoding="utf-8"))
    m = [x for x in d["models"] if x["key"] == "llama31_8b"][0]
    return {reg: dict(a=m["regimes"][reg]["a"], b=m["regimes"][reg]["b"], alpha=m["alpha"]) for reg in REGIMES}


def gbar(n: int, rho: float) -> float:
    return max(1.0, 1.0 / n + (1.0 - 1.0 / n) * rho)


def attribute(r: Req, e: dict, a1f: float) -> tuple[float, float, float]:
    a, b, al = e["a"], e["b"], e["alpha"]
    A0, a1 = (1 - a1f) * a, a1f * a
    base = A0 * r.S1 + a1 * r.N + al * r.PF
    solo = al * r.p + a * r.l + b * (r.p * r.l + r.l * (r.l + 1) / 2)
    return (base + b * r.SK) / solo, (base + b * r.SM) / solo, solo


def bounds(rule: str, chunked: bool, B: int, K: int, P_max: int, M: int, l_v: int, oth: int, e: dict, a1f: float):
    """Thm 4 bounds (Shapley, proportional); inf where the theorem gives none."""
    a, b, al = e["a"], e["b"], e["alpha"]
    rho = (a1f * a + b * (P_max + M)) / a
    m = max(1.0, al / a)
    if rule == "i":
        n_star = min(B, K // (P_max + M))
        return 1.0, gbar(max(n_star, 1), rho)
    if rule == "ii":
        return 1.0 + (B - 1) * m, gbar(B, rho) + (B - 1) * m
    if chunked:
        return np.inf, np.inf
    npre = min(l_v, oth)                    # (c2) and (d) as registered in PREREG v1.1 / THEORY v1
    return 1.0 + npre * m, gbar(B, rho) + npre * m


def bounds_corrected(rule: str, chunked: bool, B: int, K: int, P_max: int, M: int, l_v: int, oth: int, e: dict,
                     a1f: float):
    """Thm 4 as corrected after E4 (THEORY v1.1): (b) only for whole-prefill admission; (d) counts the B - 1
    requests already running at the victim's first admission: #preemptions <= min(l_v, B - 1 + N_oth)."""
    a, b, al = e["a"], e["b"], e["alpha"]
    rho = (a1f * a + b * (P_max + M)) / a
    m = max(1.0, al / a)
    if rule == "i":
        return bounds(rule, chunked, B, K, P_max, M, l_v, oth, e, a1f)
    if rule == "ii" and not chunked:
        return 1.0 + (B - 1) * m, gbar(B, rho) + (B - 1) * m
    if rule == "iii" and not chunked:
        npre = min(l_v, B - 1 + oth)
        return 1.0 + npre * m, gbar(B, rho) + npre * m
    return np.inf, np.inf


# ------------------------------------------------------------------------------------------------ grid
def windows(trace: str):
    """(day, seed) -> 1000 consecutive requests (p >= 1, l >= 1) starting at a random time in that day."""
    df = load_trace(trace)
    if "part" in df.columns:
        df = df[df.part == 1]
    df = df[(df.p >= 1) & (df.l >= 1)].sort_values("t").reset_index(drop=True)
    t = df.t.to_numpy()
    day0 = np.floor(t[0] / 86400)
    out = {}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        for d in range(7):
            lo, hi = np.searchsorted(t, [(day0 + d) * 86400, (day0 + d + 1) * 86400])
            if hi - lo < 1000:
                continue
            i0 = int(rng.integers(lo, hi - 1000))
            out[(d, s)] = df.iloc[i0:i0 + 1000][["t", "p", "l"]].to_numpy()
    return out


def build(win: np.ndarray, P_max: int, M: int, B: int, K: int, u: float, n_adv: int, rng) -> list[Req]:
    t, p, l = win[:, 0], win[:, 1].astype(int), np.minimum(win[:, 2].astype(int), M)
    lam = len(t) / max(t[-1] - t[0], 1e-9)
    n_cap = max(1, min(B, int(K // (p.mean() + l.mean()))))
    delta = u * n_cap / (lam * l.mean())
    arr = np.floor((t - t[0]) / delta).astype(int)
    reqs = [Req(i, int(arr[i]), int(p[i]), int(l[i])) for i in range(len(t))]
    if n_adv:
        hit = int(arr[rng.integers(0, len(t))])
        reqs += [Req(len(t) + j, hit, P_max, M, adv=True) for j in range(n_adv)]
    return reqs


def grid(args):
    E = energy_params()
    rows = []
    t0 = time.time()
    for trace, lim in TRACES.items():
        if args.trace and trace != args.trace:
            continue
        P_max, M = lim["P_max"], lim["M"]
        W = windows(trace)
        print(f"[e4] {trace}: {len(W)} windows", flush=True)
        for (day, seed), win in sorted(W.items()):
            ram_guard()
            for B in (8, 32):
                for kk in (1, 2, 4, 8):
                    K = kk * (P_max + M)
                    for u in (0.8, 1.2):
                        for n_adv in (0, 1, 3):
                            for rule in ("i", "ii", "iii"):
                                for chunked in (False, True):
                                    rng = np.random.default_rng(seed * 1000 + day)
                                    reqs = build(win, P_max, M, B, K, u, n_adv, rng)
                                    sim = Sim(rule, B, K, P_max, M, chunked, reqs)
                                    iters = sim.run()
                                    complete = all(r.done_at >= 0 for r in reqs)
                                    rows += cell_rows(trace, day, seed, B, kk, K, u, n_adv, rule, chunked, iters,
                                                      complete, reqs, P_max, M, E, sim.stalled)
            print(f"[e4] {trace} day {day} seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / f"e4_grid_{trace}.csv", index=False)
        pd.DataFrame(rows).to_csv(RESULTS / f"e4_grid_{trace}.csv", index=False)


def cell_rows(trace, day, seed, B, kk, K, u, n_adv, rule, chunked, iters, complete, reqs, P_max, M, E, stalled):
    out = []
    vic = [r for r in reqs if not r.adv]
    for reg, e in E.items():
        for a1f in A1_FRACS:
            gs, gp, rs, rp, viol_s, viol_p, vc_s, vc_p = [], [], [], [], 0, 0, 0, 0
            for r in vic:
                if r.done_at < 0:
                    continue
                s, pr, _ = attribute(r, e, a1f)
                bs, bp = bounds(rule, chunked, B, K, P_max, M, r.l, r.oth_adm, e, a1f)
                gs.append(s); gp.append(pr); rs.append(s / bs); rp.append(pr / bp)
                viol_s += s > bs * (1 + 1e-9); viol_p += pr > bp * (1 + 1e-9)
                cs, cp = bounds_corrected(rule, chunked, B, K, P_max, M, r.l, r.oth_adm, e, a1f)
                vc_s += s > cs * (1 + 1e-9); vc_p += pr > cp * (1 + 1e-9)
            npre = [r.n_pre for r in vic]
            out.append(dict(trace=trace, day=day, seed=seed, B=B, kK=kk, K=K, u=u, n_adv=n_adv, rule=rule,
                            chunked=chunked, regime=reg, a1f=a1f, iters=iters, complete=complete, stalled=stalled,
                            n_victims=len(vic), n_done=len(gs),
                            gamma_shap_max=max(gs, default=np.nan), gamma_prop_max=max(gp, default=np.nan),
                            gamma_shap_mean=float(np.mean(gs)) if gs else np.nan,
                            gamma_prop_mean=float(np.mean(gp)) if gp else np.nan,
                            ratio_shap_max=max(rs, default=np.nan), ratio_prop_max=max(rp, default=np.nan),
                            viol_shap=viol_s, viol_prop=viol_p, viol_shap_corr=vc_s, viol_prop_corr=vc_p,
                            preempt_max=max(npre), preempt_total=sum(npre),
                            preempt_le_Bm1=all(x <= B - 1 for x in npre) if rule == "ii" else np.nan))
    return out


# ------------------------------------------------------------------------------------------------ livelock
def livelock(args):
    """Thm 4(c1) construction: tiny adversarial requests keep the queue full; the victim has a long prompt."""
    E = energy_params()
    p_v, l_v, p_a, l_a, B = 4096, 64, 1, 256, args.B      # B = 32 is the PREREG cell; B = 64 an addendum
    K = p_v + 16 * (l_a + 1)
    rows = []
    for rule in ("i", "ii", "iii"):
        for chunked in (False, True):
            for H in (1_000, 10_000, 100_000):
                ram_guard()
                P_max, M = p_v, max(l_v, l_a)
                Kr = K if rule != "i" else max(K, P_max + M)
                # the victim enters a loaded system (steady state of the stream), as in the proof of (c1)
                v = Req(0, 2 * l_a, p_v, l_v)
                # adversarial stream keeps the queue non-empty (Thm 4(c1)): B/2 tiny requests every l_a/4
                # iterations = 2x the rate at which B running tiny requests can finish
                arr = [a for a in range(0, H, l_a // 4) for _ in range(B // 2)]
                adv = [Req(1 + i, a, p_a, l_a, adv=True) for i, a in enumerate(arr)]
                sim = Sim(rule, B, Kr, P_max, M, chunked, [v] + adv)
                iters = sim.run(max_iters=H)
                for reg, e in E.items():
                    s, pr, solo = attribute(v, e, A1_FRACS[0])
                    rows.append(dict(B=B, K=Kr, rule=rule, chunked=chunked, H=H, iters=iters, regime=reg,
                                     victim_done=v.done_at >= 0, victim_tokens=v.t, victim_preempt=v.n_pre,
                                     victim_prefilled=v.PF, gamma_shap=s, gamma_prop=pr,
                                     bound_shap=bounds(rule, chunked, B, Kr, P_max, M, l_v, v.oth_adm, e, A1_FRACS[0])[0]))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / ("e4_livelock.csv" if B == 32 else f"e4_livelock_B{B}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "livelock", "summarize"])
    ap.add_argument("--trace", default=None, choices=list(TRACES))
    ap.add_argument("--B", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "summarize":
        from e4_summarize import summarize
        summarize()
    else:
        {"grid": grid, "livelock": livelock}[a.cmd](a)
