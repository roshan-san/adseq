"""Point metrics (base paper: RMSE, R2, +/-10% accuracy) and forecast-comparison tests."""

import numpy as np
from scipy import stats


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "acc10": float(np.mean(np.abs(e) <= 0.10 * np.abs(y))),  # paper's +/-10% accuracy
    }


def diebold_mariano(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray, max_lag: int | None = None
) -> tuple[float, float, float]:
    """DM test on squared errors with a Newey-West (HAC, Bartlett) long-run variance.

    Daily loss differentials are autocorrelated (lag-1 ACF ~0.4 on Muscatine), so the plain
    h = 1 variance would understate the uncertainty and give too-small p-values.
    max_lag defaults to the Newey-West (1994) rule floor(4 (n/100)^(2/9)) (= 5 for n = 365).
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
