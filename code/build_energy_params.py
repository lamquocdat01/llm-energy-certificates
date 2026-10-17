"""T02: build results/energy_params.json from published numbers (results/energy_sources_raw.json).

Per model we take
  * a_ref  : ML.ENERGY Leaderboard v3 steady-state GPU energy per output token at a stated batch
             (prefill folded in; github ml-energy/leaderboard @2bdd007, public/data/models/*.json);
  * alpha  : prefill J per input token = r_pre * a, where r_pre = (prefill J/in-tok)/(decode J/out-tok) from
             ML.ENERGY paper (arXiv 2505.06371) App. B.3 Table 4, in 512 / out 512 column (derived ratio);
  * beta=0 : prefill J per input token is flat in prompt length in the same Table 4 (in 512 vs in 4096);
  * b      : three regimes for the context slope kappa = b / a
      meas   least-squares fit of per-token GPU energy vs prompt-length bin midpoint, TokenPowerBench
             (arXiv 2512.03024) Fig. 3(b) (figure-read, 3 points per model)  -> MAIN regime;
      arch   memory-traffic model: kappa = KV bytes per context token * batch / weight bytes (assumption);
      t4fit  fit of derived decode J/out-tok vs mean decode KV length over ML.ENERGY Table 4 rows (8B, 9 pts).
  a is set so that e_dec(c_ref) = a_ref at the reference context c_ref (= 1000 tokens, assumption).
"""
from __future__ import annotations

import json

import numpy as np

from common import RESULTS

C_REF = 1000.0
RAW = json.loads((RESULTS / "energy_sources_raw.json").read_text(encoding="utf-8"))


def lb_value(model_sub: str, n_gpu: int, seqs: int, pick_max: bool = False) -> tuple[float, str]:
    """Leaderboard per-output-token energy. For Qwen3-8B two task sets exist; GPQA (reasoning) has the larger value."""
    vals = []
    for e in RAW["entries"]:
        if (e["phase"] == "total_per_token" and model_sub in e["model"] and e.get("n_gpu") == n_gpu
                and "H100" in str(e["gpu"]) and f"max_num_seqs={seqs} " in e["batch_or_conc"] + " "):
            vals.append((e["value"], e["location"]))
    if not vals:
        raise KeyError((model_sub, n_gpu, seqs))
    vals.sort()
    return vals[-1] if pick_max else vals[0]


def tpb_fit(model: str) -> dict:
    pts = [(c["ctx_len"], c["value"]) for c in RAW["context_scaling"]
           if "2512" in c["source_id"] and c["model"] == model and "GPU-only" in c["unit"]]
    x, y = np.array(pts, float).T
    b, a0 = np.polyfit(x, y, 1)
    yhat = a0 + b * x
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"kappa": float(b / a0), "slope": float(b), "intercept": float(a0), "r2": float(r2), "n": len(x),
            "points": pts, "source": "arXiv:2512.03024 Fig. 3(b) GPU-only (figure-read, approx)"}


def t4_fit(model: str = "Llama 3.1 8B") -> dict:
    """Decode J/out-tok vs mean decode context (in_mean + out_mean/2) across all PD configs in Table 4."""
    pts = []
    for e in RAW["entries"]:
        if e["phase"] == "decode" and e["model"] == model and "Table 4" in e["location"]:
            loc = e["location"]
            i = float(loc.split("Input mean ")[1].split(" ")[0])
            o = float(loc.split("Output mean ")[1].split("'")[0])
            pts.append((i + o / 2.0, e["value"]))
    x, y = np.array(pts, float).T
    b, a0 = np.polyfit(x, y, 1)
    yhat = a0 + b * x
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"kappa": float(b / a0), "slope": float(b), "intercept": float(a0), "r2": float(r2), "n": len(x),
            "points": pts, "source": "arXiv:2505.06371 App. B.3 Table 4 (derived decode J/out-tok; approx)"}


def r_pre(model: str) -> tuple[float, str]:
    dec = pre = None
    for e in RAW["entries"]:
        if (e["model"] == model and "Table 4" in e["location"] and "Input mean 512 / Output mean 512" in e["location"]
                and ("1P3D" in e["location"] or "1P1D" in e["location"])):
            if e["phase"] == "decode":
                dec = e["value"]
            elif e["phase"] == "prefill":
                pre = e["value"]
    return pre / dec, f"arXiv:2505.06371 Table 4, {model}, in 512/out 512: prefill {pre:.4f} J/in-tok / decode {dec:.4f} J/out-tok"


# architecture (bf16): KV bytes per context token = 2 (K,V) * layers * kv_heads * head_dim * 2 bytes
ARCH = {
    "llama31_8b": dict(layers=32, kv_heads=8, head_dim=128, params=8.03e9, src="Llama-3.1-8B config.json"),
    "llama31_70b": dict(layers=80, kv_heads=8, head_dim=128, params=70.6e9, src="Llama-3.1-70B config.json"),
    "qwen3_8b": dict(layers=36, kv_heads=8, head_dim=128, params=8.19e9, src="Qwen3-8B config.json"),
}

MODELS = [
    dict(key="llama31_8b", label="Llama-3.1-8B-Instruct, 1xH100", lb=("Llama-3.1-8B", 1, 128, False),
         tpb="Llama3-8B", t4="Llama 3.1 8B", reasoning=False),
    dict(key="llama31_70b", label="Llama-3.1-70B-Instruct, 4xH100", lb=("Llama-3.1-70B", 4, 128, False),
         tpb="Llama3-70B", t4="Llama 3.1 70B", reasoning=False),
    dict(key="qwen3_8b_reason", label="Qwen3-8B (GPQA, reasoning), 1xH100", lb=("Qwen3-8B", 1, 96, True),
         tpb="Qwen-8B", t4="Llama 3.1 8B", reasoning=True, arch="qwen3_8b"),
]


def main() -> None:
    t4_8b = t4_fit("Llama 3.1 8B")
    t4_70b = t4_fit("Llama 3.1 70B")
    out = {"generated": "2026-10-16", "c_ref": C_REF, "main_regime": "meas",
           "notes": [
               "a_ref = ML.ENERGY Leaderboard v3 per-output-token GPU energy (prefill folded in) at the stated max_num_seqs.",
               "alpha = r_pre * a with r_pre from ML.ENERGY Table 4 (derived ratio of dataset means).",
               "beta = 0: Table 4 prefill J/in-tok flat between in-mean 512 and 4096 (8B: 0.0168 vs 0.0167; 70B: 0.190 vs 0.182).",
               "Regime 'meas' (MAIN) = TokenPowerBench Fig.3(b) slope; 'arch' = KV-traffic assumption; 't4fit' = ML.ENERGY Table 4 slope (8B rows; low R2 -> stress regime).",
               "Conclusions of Thm 1-3 hold for any a, b > 0; regimes only change magnitudes.",
           ],
           "t4_fit_8b": t4_8b, "t4_fit_70b": t4_70b, "models": []}
    for m in MODELS:
        a_ref, lb_loc = lb_value(*m["lb"])
        rp, rp_src = r_pre(m["t4"])
        arch = ARCH[m.get("arch", m["key"])]
        kv_bytes = 2 * arch["layers"] * arch["kv_heads"] * arch["head_dim"] * 2
        batch = m["lb"][2]
        k_arch = kv_bytes * batch / (2 * arch["params"])
        tpb = tpb_fit(m["tpb"])
        kappas = {"meas": tpb["kappa"], "arch": k_arch, "t4fit": t4_8b["kappa"]}
        regimes = {}
        for reg, k in kappas.items():
            a = a_ref / (1.0 + k * C_REF)
            regimes[reg] = {"a": a, "b": a * k, "kappa": k}
        out["models"].append({
            "key": m["key"], "label": m["label"], "reasoning": m["reasoning"], "context_window": 131072,
            "a_ref_J_per_tok": a_ref, "a_ref_source": f"ML.ENERGY Leaderboard v3 @2bdd007, {m['lb'][0]}, {m['lb'][1]}xH100, max_num_seqs={batch}: {lb_loc}",
            "r_pre": rp, "r_pre_source": rp_src + ("" if m["t4"] in m["label"] or not m["reasoning"] else " (proxy: no Table-4 row for Qwen3)"),
            "alpha": rp * a_ref / (1.0 + kappas["meas"] * C_REF), "beta": 0.0,
            "kv_bytes_per_token": kv_bytes, "arch_source": arch["src"] + " (bf16)", "tpb_fit": tpb,
            "regimes": regimes,
        })
    (RESULTS / "energy_params.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in out["models"]:
        print(m["key"], "a_ref=%.4f r_pre=%.3f alpha=%.4g" % (m["a_ref_J_per_tok"], m["r_pre"], m["alpha"]),
              {k: "a=%.4g b=%.3g k=%.3g" % (v["a"], v["b"], v["kappa"]) for k, v in m["regimes"].items()},
              "TPB R2=%.4f" % m["tpb_fit"]["r2"])
    print("T4 fit 8B: kappa=%.3g R2=%.3f n=%d; 70B: kappa=%.3g R2=%.3f n=%d" % (
        t4_8b["kappa"], t4_8b["r2"], t4_8b["n"], t4_70b["kappa"], t4_70b["r2"], t4_70b["n"]))


if __name__ == "__main__":
    main()
