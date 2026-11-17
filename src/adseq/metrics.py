"""Point metrics (Schroer & Just 2024: adjusted R2, MAPE, RMSE, TeEI) and the DM test."""

import numpy as np
from scipy import stats


def regression_metrics(y: np.ndarray, p: np.ndarray, k: int = 1) -> dict[str, float]:
    """k = number of input features per sample, for the adjusted R2."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    n = len(y)
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse, mape = float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e) / np.abs(y)))
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(e))),
        "mape": 100 * mape,  # %
        "r2": r2,
        "adj_r2": adj_r2,
        # Schroer & Just's testing error index: MAPE^2 x RMSE / adj R2 (MAPE as a fraction)
        "teei": mape**2 * rmse / adj_r2 if adj_r2 > 0 else float("nan"),
    }


def diebold_mariano(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray, max_lag: int | None = None
) -> tuple[float, float, float]:
    """DM test on squared errors with a Newey-West (HAC, Bartlett) long-run variance.

    Minute-by-minute forecasts of overlapping 24-h means have strongly autocorrelated loss differentials,
    so the plain variance would understate the uncertainty and give too-small p-values.
    max_lag defaults to the Newey-West (1994) rule floor(4 (n/100)^(2/9)); for h-step-ahead
    forecasts pass at least h - 1 (the report uses the 1440-min horizon).
    H0: equal accuracy. Negative statistic => model 1 has *lower* loss than model 2.
    Returns (statistic, two-sided p, one-sided p for "model 1 better"), t(n-1) reference.
    """
    y, p1, p2 = (np.asarray(a, float) for a in (y, p1, p2))
    d = (p1 - y) ** 2 - (p2 - y) ** 2
    n = len(d)
    if max_lag is None:
        max_lag = int(4 * (n / 100) ** (2 / 9))
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, min(max_lag, n - 1) + 1):
        lrv += 2 * (1 - k / (max_lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan"), float("nan")
    stat = float(d.mean() / np.sqrt(lrv / n))
    return stat, float(2 * stats.t.sf(abs(stat), df=n - 1)), float(stats.t.cdf(stat, df=n - 1))
