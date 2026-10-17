"""T11: verify Thm 1-3 bounds numerically; 0 violations required (PREREG §4). -> results/bounds_check.csv

Thm 1: every trace request served under (P_max, M) has E <= E_max(P_max, M); 1e5 random requests per seed in the
       admissible box; E_max attained at (P_max, M).
Thm 2: synthetic prompt universes (p, l(x), tau_g(x)): brute-force WC(L o g) equals the envelope formula
       max_p E(p, min(M, M_g(p))); WC(L o g) <= WC(L); equality iff M_g(P_max) >= M.
Thm 3: on trace samples, (a) cut of the p-aware cap at certificate T equals P(E(p,l) > T); (b) no fixed cap (P, M)
       with E(P, M) <= T truncates less; (c) random certified content-dependent policies truncate at least as much.
Thm 4 (v1.1, P2): random small batching instances in the E4 simulator (rules i / ii-whole / iii-whole, random B, K,
       arrivals); every request's attributed energy <= the corrected Thm 4 bound (Shapley and proportional), and
       rule ii-whole never preempts a request more than B - 1 times.
       `--theorems Thm4` runs only Thm 4 and replaces the Thm 4 rows of an existing bounds_check.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import ram_guard, RESULTS, SEEDS, TRACES, load_trace
from energy_model import E_max, E_request, EnergyParams, load_params

TOL = 1e-9


def m_budget(prm, p, T):
    from certificate_curve import m_budget as mb
    return mb(prm, p, T)


def thm1(prm, name, df, rng, seed) -> list[dict]:
    p, l = df["p"].to_numpy(), df["l"].to_numpy()
    P = int(p.max())
    out = []
    for q in (0.99, 0.999, 1.0):
        M = int(l.max()) if q >= 1 else int(np.ceil(np.quantile(l, q)))
        bound = float(E_max(prm, P, M))
        e = E_request(prm, p, np.minimum(l, M))
        v1 = int((e > bound * (1 + TOL)).sum())
        rp, rl = rng.integers(0, P + 1, 100_000), rng.integers(0, M + 1, 100_000)
        v2 = int((E_request(prm, rp, rl) > bound * (1 + TOL)).sum())
        att = abs(float(E_request(prm, P, M)) - bound) / bound
        out.append(dict(theorem="Thm1", check=f"{name} q={q} trace+random<=E_max; attained", seed=seed,
                        n_checks=len(e) + 100_000 + 1, n_violations=v1 + v2 + int(att > TOL)))
    return out


def thm2(prm, rng, seed, n_univ=2000) -> list[dict]:
    viol_formula = viol_le = viol_iff = n_iff = 0
    for _ in range(n_univ):
        P = int(rng.integers(1, 4000))
        M = int(rng.integers(1, 4000))
        n = int(rng.integers(5, 200))
        px = rng.integers(0, P + 1, n)
        lx = rng.integers(0, 3 * M, n)
        # gate: content-dependent stop; some prompts evade (tau = inf), others stop at random t
        tau = np.where(rng.random(n) < rng.random(), np.inf, rng.integers(0, 3 * M, n).astype(float))
        rich = rng.random() < 0.5
        if rich:  # Assumption R at P: some prompt of length P generates >= M tokens
            px[0], lx[0] = P, 3 * M
        served = np.minimum(np.minimum(lx, tau), M)
        wc_brute = float(E_request(prm, px, served).max())
        env = {}
        for pi, li, ti in zip(px, lx, tau):
            env[pi] = max(env.get(pi, -1), min(li, ti))
        wc_env = max(float(E_request(prm, pi, min(M, m))) for pi, m in env.items())
        viol_formula += int(abs(wc_brute - wc_env) > TOL * max(wc_env, 1))
        # pure limiter on the same universe
        wc_L = float(E_request(prm, px, np.minimum(lx, M)).max())
        viol_le += int(wc_brute > wc_L * (1 + TOL))
        # iff (Thm 2(ii)) on universes satisfying Assumption R at P (a prompt of length P with l >= M exists):
        # WC(L o g) == WC(L)  <=>  M_g(P) >= M
        if rich:
            n_iff += 1
            eq = abs(wc_brute - wc_L) <= TOL * wc_L
            viol_iff += int(eq != (env.get(P, -1) >= M))
    return [dict(theorem="Thm2", check="envelope formula == brute force", seed=seed, n_checks=n_univ, n_violations=viol_formula),
            dict(theorem="Thm2", check="WC(L o g) <= WC(L)", seed=seed, n_checks=n_univ, n_violations=viol_le),
            dict(theorem="Thm2", check="WC(L o g)==WC(L) iff M_g(P_max)>=M (universes with R)", seed=seed,
                 n_checks=n_iff, n_violations=viol_iff)]


def thm3(prm, name, df, rng, seed, n_sample=200_000) -> list[dict]:
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    p, l = df["p"].to_numpy()[idx].astype(float), df["l"].to_numpy()[idx].astype(float)
    e = E_request(prm, p, l)
    va = vb = vc = na = nb = nc = 0
    for T in np.quantile(e, [0.5, 0.9, 0.99, 0.999]):
        mT = m_budget(prm, p, T)
        cut_star = float(((mT < 0) | (l > mT)).mean())
        va += int(abs(cut_star - float((e > T).mean())) > 1e-12); na += 1
        # fixed caps with E(P, M) <= T
        for P in np.unique(np.quantile(p, np.linspace(0, 1, 25)).astype(int)):
            Ms = np.arange(0, int(l.max()) + 2)
            ok = E_request(prm, P, Ms) <= T
            if not ok.any():
                continue
            M = int(Ms[ok].max())
            cut_fixed = float(((p > P) | (l > M)).mean())
            vb += int(cut_fixed < cut_star - 1e-12); nb += 1
        # random certified content-dependent policies: serve each x up to a random fraction of M_T(p)
        for _ in range(20):
            frac = rng.random(len(p)) ** rng.uniform(0.01, 2)
            cap = np.where(mT < 0, -1, np.floor(frac * np.maximum(mT, 0) + (rng.random(len(p)) < 0.5) * (mT - np.floor(frac * np.maximum(mT, 0)))))
            served = np.minimum(l, cap)
            adm = cap >= 0  # rejected requests are not served (0 J)
            assert (E_request(prm, p[adm], served[adm]) <= T * (1 + TOL)).all()
            cut_pi = float(((cap < 0) | (l > cap)).mean())
            vc += int(cut_pi < cut_star - 1e-12); nc += 1
    return [dict(theorem="Thm3", check=f"{name} cut(pi_T*) == P(E>T)", seed=seed, n_checks=na, n_violations=va),
            dict(theorem="Thm3", check=f"{name} fixed caps never truncate less", seed=seed, n_checks=nb, n_violations=vb),
            dict(theorem="Thm3", check=f"{name} certified random policies never truncate less", seed=seed, n_checks=nc, n_violations=vc)]


def thm4(rng, seed, n_inst=300) -> list[dict]:
    from batch_kv import A1_FRACS, Req, Sim, attribute, bounds_corrected, energy_params
    E = energy_params()
    nb = vb = npre = vpre = 0
    for _ in range(n_inst):
        P_max, M = int(rng.integers(64, 3000)), int(rng.integers(8, 600))
        B = int(rng.integers(1, 17))
        K = int((P_max + M) * rng.uniform(1.0, 6.0))
        rule = ["i", "ii", "iii"][int(rng.integers(0, 3))]
        n = int(rng.integers(5, 80))
        reqs = [Req(i, int(rng.integers(0, 400)), int(rng.integers(1, P_max + 1)), int(rng.integers(1, M + 1)))
                for i in range(n)]
        Sim(rule, B, K, P_max, M, False, reqs).run()
        for r in reqs:
            if r.done_at < 0:
                continue
            for e in E.values():
                for a1f in A1_FRACS:
                    s, pr, _ = attribute(r, e, a1f)
                    bs, bp = bounds_corrected(rule, False, B, K, P_max, M, r.l, r.oth_adm, e, a1f)
                    nb += 2
                    vb += int(s > bs * (1 + TOL)) + int(pr > bp * (1 + TOL))
            if rule == "ii":
                npre += 1
                vpre += int(r.n_pre > B - 1)
    return [dict(theorem="Thm4", check="attributed energy <= corrected Thm 4 bound (random instances)", seed=seed,
                 n_checks=nb, n_violations=vb),
            dict(theorem="Thm4", check="rule ii (whole-context admission): preemptions <= B - 1", seed=seed,
                 n_checks=npre, n_violations=vpre)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--theorems", nargs="*", default=["Thm1", "Thm2", "Thm3", "Thm4"])
    args = ap.parse_args()
    params = load_params(RESULTS / "energy_params.json")
    rows = []
    for seed in args.seed:
        rng = np.random.default_rng(seed)
        if {"Thm1", "Thm2", "Thm3"} & set(args.theorems):
            for key, prm in params.items():
                if not key.endswith("|meas") and seed != args.seed[0]:
                    continue  # all regimes on the first seed, main regime on every seed
                for name in TRACES:
                    df = (ram_guard(), load_trace(name))[1]
                    for r in thm1(prm, name, df, rng, seed) + thm3(prm, name, df, rng, seed):
                        rows.append({**r, "params": key})
                for r in thm2(prm, rng, seed):
                    rows.append({**r, "params": key})
        if "Thm4" in args.theorems:
            rows += [{**r, "params": "llama31_8b|meas+arch"} for r in thm4(np.random.default_rng(seed + 4000), seed)]
        print("seed", seed, "done", flush=True)
    out = pd.DataFrame(rows)
    if set(args.theorems) == {"Thm4"} and (RESULTS / "bounds_check.csv").exists():
        old = pd.read_csv(RESULTS / "bounds_check.csv")
        out = pd.concat([old[old.theorem != "Thm4"], out], ignore_index=True)
    out.to_csv(RESULTS / "bounds_check.csv", index=False)
    print(out.groupby("theorem")[["n_checks", "n_violations"]].sum().to_string())
    print("TOTAL violations:", int(out.n_violations.sum()))


if __name__ == "__main__":
    main()
