"""E5 (P2, PREREG v1.1 §3): measured decode energy vs context on the laptop CPU.

Power source: Windows "Energy Meter" PDH counter RAPL_Package0_PKG (cumulative package energy in pWh,
~20 ms refresh, readable without admin; source (d) of PREREG v1.1). Runtime: llama.cpp llama-server
(CPU build) streaming /completion, so every token gets a timestamp. Model: Qwen2.5-1.5B-Instruct Q4_K_M.

    python e5_laptop.py run       # idle 60 s, 7 prompt lengths x 3 reps (shuffled), idle 60 s
    python e5_laptop.py analyze   # net energy per 64-token window, OLS e_dec = a + b c, bootstrap CI, figure

Raw samples go to <THS_DATASETS>/_derived/38/e5/; per-run / per-window tables and the figure to results/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests

from common import DATASETS, DERIVED, RESULTS, SEEDS, ram_guard

MODEL_DIR = DATASETS / "LLM_Models"
SERVER = MODEL_DIR / "_bin" / "llama-b11706" / "llama-server.exe"
MODEL = MODEL_DIR / "Qwen2.5-1.5B-Instruct-GGUF" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
RAW = DERIVED / "e5"
PORT = 8091
URL = f"http://127.0.0.1:{PORT}"
PWH_TO_J = 3.6e-9                     # 1 pWh = 3.6e-9 J
P_GRID = [128, 512, 1024, 2048, 4096, 8192, 16384]
N_PREDICT, N_REPS, WIN = 256, 3, 64
IDLE_S, REST_S, THREADS = 60.0, 10.0, 4


class RaplSampler(threading.Thread):
    """Polls the package-energy counter every ~20 ms; stores (perf_counter s, energy J, system CPU %)."""

    def __init__(self, period: float = 0.02):
        super().__init__(daemon=True)
        import win32pdh
        self.pdh = win32pdh
        self.q = win32pdh.OpenQuery()
        self.h = win32pdh.AddEnglishCounter(self.q, r"\Energy Meter(RAPL_Package0_PKG)\Energy")
        self.fmt = win32pdh.PDH_FMT_LARGE | win32pdh.PDH_FMT_NOSCALE
        self.period, self.rows, self.stop_flag = period, [], threading.Event()
        psutil.cpu_percent(None)

    def read(self) -> float:
        self.pdh.CollectQueryData(self.q)
        return self.pdh.GetFormattedCounterValue(self.h, self.fmt)[1] * PWH_TO_J

    def run(self):
        k = 0
        while not self.stop_flag.is_set():
            e = self.read()
            cpu = psutil.cpu_percent(None) if k % 25 == 0 else np.nan
            self.rows.append((time.perf_counter(), e, cpu))
            k += 1
            time.sleep(self.period)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["t", "E_J", "cpu_pct"])


def start_server(ctx: int) -> subprocess.Popen:
    log = open(RAW / "llama_server.log", "a", encoding="utf-8")
    proc = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--port", str(PORT), "-t", str(THREADS), "-tb", str(THREADS),
                             "-c", str(ctx), "-np", "1", "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if requests.get(f"{URL}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.kill()
    raise SystemExit("llama-server did not become healthy")


def synthetic_prompt(p: int, rng: np.random.Generator) -> list[int]:
    """Random English-like word salad, tokenised by the server and cut to exactly p tokens."""
    words = ("the of and to in a is that for it as with was on be by at this from or have an they which one you were "
             "all we when there can more has some time two may first then do any like my now over such our man me even "
             "most made after also did many before must through back years where much your way well down should because "
             "each just those people how too little state good very make world still own see men work long get here "
             "between both life being under never day same another know while last might us great old year off come since "
             "against go came right used take three energy service request token model budget limit window cache").split()
    ids: list[int] = []
    while len(ids) < p:
        text = " ".join(rng.choice(words, size=max(256, p)))
        ids += requests.post(f"{URL}/tokenize", json={"content": text}, timeout=60).json()["tokens"]
    return ids[:p]


def one_request(ids: list[int], seed: int) -> dict:
    body = {"prompt": ids, "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.7, "seed": seed,
            "cache_prompt": False, "stream": True}
    t_send = time.perf_counter()
    stamps, timings = [], {}
    with requests.post(f"{URL}/completion", json=body, stream=True, timeout=3600) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            msg = json.loads(line[6:])
            if msg.get("content", "") != "" or msg.get("tokens"):
                stamps.append(time.perf_counter())
            if msg.get("stop"):
                timings = msg.get("timings", {})
    return {"t_send": t_send, "stamps": stamps, "timings": timings}


def run(args):
    RAW.mkdir(parents=True, exist_ok=True)
    ram_guard()
    rapl = RaplSampler()
    rapl.read()
    ctx = max(P_GRID) + N_PREDICT + 64
    proc = start_server(ctx)
    print(f"[e5] server pid {proc.pid}, ctx {ctx}", flush=True)
    rapl.start()
    rng = np.random.default_rng(SEEDS[0])
    plan = [(p, r) for r in range(N_REPS) for p in P_GRID]
    order = rng.permutation(len(plan))
    runs, toks = [], []
    try:
        t_idle0 = time.perf_counter(); time.sleep(IDLE_S); t_idle1 = time.perf_counter()
        runs.append({"kind": "idle_pre", "t_send": t_idle0, "t_end": t_idle1})
        for k in order:
            p, rep = plan[k]
            ram_guard()
            seed = SEEDS[0] + rep
            ids = synthetic_prompt(p, np.random.default_rng(seed * 100003 + p))
            time.sleep(REST_S)
            out = one_request(ids, seed)
            st = out["stamps"]
            row = {"kind": "run", "p": p, "rep": rep, "seed": seed, "t_send": out["t_send"], "t_first": st[0],
                   "t_end": st[-1], "n_tok": len(st), "free_ram_gb": psutil.virtual_memory().available / 2**30}
            row.update({f"srv_{k2}": v for k2, v in out["timings"].items()})
            runs.append(row)
            toks += [{"p": p, "rep": rep, "i": i + 1, "t": t} for i, t in enumerate(st)]
            print(f"[e5] p={p:5d} rep={rep} tokens={len(st)} prefill={st[0]-out['t_send']:.1f}s "
                  f"decode={st[-1]-st[0]:.1f}s", flush=True)
        t_idle0 = time.perf_counter(); time.sleep(IDLE_S); t_idle1 = time.perf_counter()
        runs.append({"kind": "idle_post", "t_send": t_idle0, "t_end": t_idle1})
    finally:
        rapl.stop_flag.set(); rapl.join(2)
        proc.terminate()
        stamp = time.strftime("%Y%m%d_%H%M")
        rapl.frame().to_csv(RAW / f"rapl_{stamp}.csv.gz", index=False)
        pd.DataFrame(runs).to_csv(RAW / f"runs_{stamp}.csv", index=False)
        pd.DataFrame(toks).to_csv(RAW / f"tokens_{stamp}.csv.gz", index=False)
        print(f"[e5] saved {RAW} *_{stamp}", flush=True)


# ----------------------------------------------------------------------------------------------- analysis
def latest(prefix: str) -> Path:
    return sorted(RAW.glob(f"{prefix}_*"))[-1]


def energy_at(rapl: pd.DataFrame, t) -> np.ndarray:
    return np.interp(np.asarray(t, float), rapl["t"].to_numpy(), rapl["E_J"].to_numpy())


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return coef[0], coef[1], r2


def analyze(args):
    rapl = pd.read_csv(latest("rapl"))
    runs = pd.read_csv(latest("runs"))
    toks = pd.read_csv(latest("tokens"))
    idle = runs[runs.kind.str.startswith("idle")]
    P_idle = float(np.mean([(energy_at(rapl, r.t_end) - energy_at(rapl, r.t_send)) / (r.t_end - r.t_send)
                            for r in idle.itertuples()]))
    P_idle_each = [(energy_at(rapl, r.t_end) - energy_at(rapl, r.t_send)) / (r.t_end - r.t_send) for r in idle.itertuples()]
    rr = runs[runs.kind == "run"].copy()
    gross = lambda t0, t1: energy_at(rapl, t1) - energy_at(rapl, t0)
    rr["E_pre_gross"] = gross(rr.t_send, rr.t_first)
    rr["E_pre_net"] = rr.E_pre_gross - P_idle * (rr.t_first - rr.t_send)
    rr["E_dec_gross"] = gross(rr.t_first, rr.t_end)
    rr["E_dec_net"] = rr.E_dec_gross - P_idle * (rr.t_end - rr.t_first)
    rr["P_run_W"] = (rr.E_pre_gross + rr.E_dec_gross) / (rr.t_end - rr.t_send)
    win = []
    for (p, rep), g in toks.groupby(["p", "rep"]):
        t = g.sort_values("i")["t"].to_numpy()
        # token i (1-based) is emitted at t[i-1]; window w covers tokens (w*64, (w+1)*64] measured from the
        # previous token's stamp, so the first window starts at token 1 (prefill excluded).
        for w in range(len(t) // WIN):
            i0, i1 = w * WIN, (w + 1) * WIN          # interval [t[i0], t[i1]] spans tokens i0+2 .. i1+1
            if i1 >= len(t):
                break
            dt = t[i1] - t[i0]
            eg = float(gross(t[i0], t[i1]))
            win.append({"p": p, "rep": rep, "w": w, "c_mid": p + (i0 + i1) / 2 + 1, "dt_s": dt,
                        "e_gross_J_tok": eg / (i1 - i0), "e_net_J_tok": (eg - P_idle * dt) / (i1 - i0),
                        "s_per_tok": dt / (i1 - i0)})
    win = pd.DataFrame(win)
    res = {"P_idle_W": P_idle, "P_idle_pre_post_W": P_idle_each, "n_windows": len(win), "n_runs": len(rr),
           "p_grid": sorted(rr.p.unique().tolist())}
    rng = np.random.default_rng(SEEDS[0])
    keys = list(win.groupby(["p", "rep"]).groups.keys())
    for kind in ("net", "gross"):
        x, y = win.c_mid.to_numpy(float), win[f"e_{kind}_J_tok"].to_numpy(float)
        a, b, r2 = ols(x, y)
        boot = []
        for _ in range(2000):
            pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            sub = pd.concat([win[(win.p == pp) & (win.rep == rp)] for pp, rp in pick])
            boot.append(ols(sub.c_mid.to_numpy(float), sub[f"e_{kind}_J_tok"].to_numpy(float))[:2])
        boot = np.array(boot)
        res[kind] = {"a": a, "b": b, "r2": r2, "kappa": b / a,
                     "a_ci": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                     "b_ci": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
                     "kappa_ci": np.percentile(boot[:, 1] / boot[:, 0], [2.5, 97.5]).tolist()}
    a_t, b_t, r2_t = ols(win.c_mid.to_numpy(float), win.s_per_tok.to_numpy(float))
    res["time"] = {"a_s": a_t, "b_s": b_t, "r2": r2_t, "kappa": b_t / a_t}
    # prefill: E_pre(p) = alpha p + beta p^2 (no intercept)
    pp, ee = rr.p.to_numpy(float), rr.E_pre_net.to_numpy(float)
    X = np.column_stack([pp, pp ** 2])
    (alpha, beta), *_ = np.linalg.lstsq(X, ee, rcond=None)
    pred = X @ np.array([alpha, beta])
    res["prefill"] = {"alpha_J_tok": alpha, "beta_J_tok2": beta,
                      "r2": 1 - ((ee - pred) ** 2).sum() / ((ee - ee.mean()) ** 2).sum(),
                      "alpha_lin_only": float(np.linalg.lstsq(pp[:, None], ee, rcond=None)[0][0]),
                      "J_per_tok_by_p": rr.groupby("p").apply(lambda g: (g.E_pre_net / g.p).mean()).to_dict()}
    prm = json.loads((RESULTS / "energy_params.json").read_text(encoding="utf-8"))
    l8 = [m for m in prm["models"] if m["key"] == "llama31_8b"][0]["regimes"]
    res["kappa_regimes_llama31_8b"] = {k: v["kappa"] for k, v in l8.items()}
    res["alpha_over_a_net"] = alpha / res["net"]["a"]
    res["H6_pass"] = bool(res["net"]["r2"] >= 0.95 and res["net"]["b_ci"][0] > 0)
    res["cpu_pct_median_during_runs"] = float(np.nanmedian(rapl["cpu_pct"]))
    rr.to_csv(RESULTS / "e5_laptop_runs.csv", index=False)
    win.to_csv(RESULTS / "e5_laptop_windows.csv", index=False)
    (RESULTS / "e5_laptop_fit.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    plot(win, rr, res)
    print(json.dumps(res, indent=1, default=float))


def plot(win, rr, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for kind, mk, col in (("net", "o", "#1f5aa6"), ("gross", "s", "#9a9a9a")):
        f = res[kind]
        ax[0].scatter(win.c_mid / 1000, win[f"e_{kind}_J_tok"], s=9, marker=mk, color=col, alpha=0.7,
                      label=f"{kind}: R$^2$={f['r2']:.3f}, b={f['b']*1e6:.2f} µJ/tok/ctx")
        xs = np.linspace(0, win.c_mid.max(), 50)
        ax[0].plot(xs / 1000, f["a"] + f["b"] * xs, color=col, lw=1)
    ax[0].set_xlabel("context c (k tokens)"); ax[0].set_ylabel("decode energy (J/token)")
    ax[0].legend(fontsize=6, frameon=False); ax[0].set_title("(a) e_dec(c) = a + b c", fontsize=8)
    g = rr.groupby("p").E_pre_net.agg(["mean", "std"])
    ax[1].errorbar(g.index / 1000, g["mean"], yerr=g["std"], fmt="o", ms=3, color="#1f5aa6")
    xs = np.linspace(0, g.index.max(), 50)
    ax[1].plot(xs / 1000, res["prefill"]["alpha_J_tok"] * xs + res["prefill"]["beta_J_tok2"] * xs ** 2, lw=1,
               color="#1f5aa6")
    ax[1].set_xlabel("prompt p (k tokens)"); ax[1].set_ylabel("prefill energy (J, net)")
    ax[1].set_title("(b) E_pre(p)", fontsize=8)
    for a_ in ax:
        a_.tick_params(labelsize=7); a_.xaxis.label.set_size(8); a_.yaxis.label.set_size(8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"e5_laptop_fit.{ext}", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
