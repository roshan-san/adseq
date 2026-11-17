"""Metrics and the Diebold-Mariano test."""

import numpy as np

from adseq.metrics import diebold_mariano, regression_metrics


def test_mape_and_adjusted_r2():
    y = np.array([100.0, 200.0, 100.0, 200.0, 150.0, 150.0])
    p = y * 1.1
    m = regression_metrics(y, p, k=2)
    assert np.isclose(m["mape"], 10.0)
    n = len(y)
    assert np.isclose(m["adj_r2"], 1 - (1 - m["r2"]) * (n - 1) / (n - 2 - 1))
    assert m["adj_r2"] < m["r2"]


def test_dm_hac_widens_for_autocorrelated_losses():
    rng = np.random.default_rng(0)
    n = 365
    y = np.zeros(n)
    noise = rng.normal(size=n)
    shared = np.convolve(rng.normal(size=n + 4), np.ones(5) / 5, "valid")  # MA(4) errors
    p1, p2 = 1.0 + shared + 0.3 * noise, 1.1 + shared
    plain, *_ = diebold_mariano(y, p1, p2, max_lag=0)
    hac, *_ = diebold_mariano(y, p1, p2)
    assert np.sign(plain) == np.sign(hac) and abs(hac) < abs(plain)


def test_dm_sign_and_identical_forecasts():
    y = np.arange(100.0)
    good, bad = y + np.sin(y), y + 3 * np.sin(y) + 1
    stat, _, p_better = diebold_mariano(y, good, bad)
    assert stat < 0 and p_better < 0.05
    assert np.isnan(diebold_mariano(y, good, good)[0])


def test_teei_is_mape_squared_times_rmse_over_adjusted_r2():
    y = np.array([100.0, 200.0, 120.0, 180.0, 150.0, 160.0, 140.0])
    p = y + np.array([5.0, -10.0, 3.0, 8.0, -4.0, 2.0, -6.0])
    m = regression_metrics(y, p, k=1)
    assert np.isclose(m["teei"], (m["mape"] / 100) ** 2 * m["rmse"] / m["adj_r2"])
