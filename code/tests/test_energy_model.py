"""Unit tests: closed forms of energy_model vs token-by-token simulation (tolerance 1e-9 relative)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_model import (EnergyParams, E_max, E_max_total, E_request, argmax_total, e_dec,  # noqa: E402
                          simulate_request)

RNG = np.random.default_rng(42)
PARAMS = [EnergyParams("t0", a=0.09, b=4.7e-6, alpha=0.027),
          EnergyParams("t1", a=0.05, b=5e-5, alpha=0.015, beta=1e-6),
          EnergyParams("t2", a=0.5, b=0.0, alpha=0.3)]


@pytest.mark.parametrize("prm", PARAMS)
def test_closed_form_matches_simulation(prm):
    for _ in range(50):
        p, l = int(RNG.integers(0, 3000)), int(RNG.integers(0, 3000))
        sim = simulate_request(prm, p, l)
        cf = float(E_request(prm, p, l))
        assert abs(sim - cf) <= 1e-9 * max(1.0, abs(sim))


@pytest.mark.parametrize("prm", PARAMS)
def test_monotone_and_emax(prm):
    P, M = 2000, 700
    grid_p, grid_l = np.meshgrid(np.arange(0, P + 1, 50), np.arange(0, M + 1, 25))
    assert np.all(E_request(prm, grid_p, grid_l) <= E_max(prm, P, M) + 1e-9)
    assert float(E_max(prm, P, M)) == pytest.approx(simulate_request(prm, P, M), rel=1e-12)


def test_quadratic_in_M():
    prm = PARAMS[0]
    M = np.arange(0, 5000, 7, dtype=float)
    y = E_max(prm, 1000, M)
    coef = np.polyfit(M, y, 2)
    assert coef[0] == pytest.approx(prm.b / 2, rel=1e-6)
    assert coef[1] == pytest.approx(prm.a + prm.b * (1000 + 0.5), rel=1e-6)


@pytest.mark.parametrize("prm", PARAMS)
def test_total_cap_exact(prm):
    for T in (1, 2, 10, 517, 4096):
        p, l, v = argmax_total(prm, T)
        assert float(E_max_total(prm, T)) == pytest.approx(v, rel=1e-12)
        assert p + l == T


def test_e_dec_linear():
    prm = PARAMS[0]
    assert float(e_dec(prm, 1000)) == pytest.approx(prm.a + 1000 * prm.b)
