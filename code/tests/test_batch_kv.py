"""Sanity checks of the E4 simulator against the solo energy model and Thm 4(a)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_kv import Req, Sim, attribute, bounds, energy_params  # noqa: E402
from energy_model import EnergyParams, E_request  # noqa: E402

E = energy_params()


def test_solo_request_matches_closed_form():
    for chunked in (False, True):
        for rule in ("i", "ii", "iii"):
            r = Req(0, 0, 1500, 300)
            Sim(rule, 8, 10_000, 1500, 300, chunked, [r]).run()
            assert r.done_at >= 0 and r.t == 300 and r.n_pre == 0
            for e in E.values():
                s, p, solo = attribute(r, e, 0.1)
                prm = EnergyParams("x", e["a"], e["b"], e["alpha"])
                assert abs(solo - E_request(prm, 1500, 300)) < 1e-9 * solo
                assert abs(s - 1) < 1e-12 and abs(p - 1) < 1e-12


def test_reservation_shapley_le_one():
    rng = np.random.default_rng(0)
    reqs = [Req(i, int(rng.integers(0, 50)), int(rng.integers(1, 2000)), int(rng.integers(1, 400))) for i in range(60)]
    Sim("i", 8, 4 * 2400, 2000, 400, False, reqs).run()
    for r in reqs:
        for e in E.values():
            s, p, _ = attribute(r, e, 0.1)
            bs, bp = bounds("i", False, 8, 4 * 2400, 2000, 400, r.l, 0, e, 0.1)
            assert s <= bs + 1e-12 and p <= bp + 1e-12


def test_preemption_counts_recompute():
    # two long requests that cannot both fit -> rule ii must preempt the later arrival and recompute it
    a, b = Req(0, 0, 900, 200), Req(1, 0, 900, 200)
    Sim("ii", 4, 2000, 900, 200, False, [a, b]).run()
    assert a.n_pre == 0 and b.n_pre >= 1 and b.PF > b.p
