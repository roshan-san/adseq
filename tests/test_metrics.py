"""Diebold-Mariano test and loading edge cases."""

import numpy as np
import pandas as pd

from adseq.data import clean_data
from adseq.metrics import diebold_mariano


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


def test_text_columns_do_not_empty_the_dataset(tmp_path):
    f = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "Year": 2020,
            "Month": 1,
            "Day": range(1, 6),
            "Biogas": range(5),
            "a": range(5),
            "note": list("vwxyz"),
        }
    ).to_csv(f, index=False)
    assert list(clean_data(f).columns) == ["Biogas", "a"]
