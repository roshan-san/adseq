"""Point metrics (Schroer & Just 2024: adjusted R2, MAPE, RMSE, TeEI)."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def regression_metrics(y: np.ndarray, p: np.ndarray, k: int = 1) -> dict[str, float]:
    """scikit-learn's RMSE, MAE, MAPE and R2, plus the adjusted R2 (k = number of input features
    per sample) and Schroer & Just's TeEI, which no library has."""
    n = len(y)
    rmse = float(root_mean_squared_error(y, p))
    mape = float(mean_absolute_percentage_error(y, p))  # a fraction
    r2 = float(r2_score(y, p, force_finite=False))  # NaN, not a made-up 0/1, for a constant y
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, p)),
        "mape": 100 * mape,  # %
        "r2": r2,
        "adj_r2": adj_r2,
        # Schroer & Just's testing error index: MAPE^2 x RMSE / adj R2 (MAPE as a fraction)
        "teei": mape**2 * rmse / adj_r2 if adj_r2 > 0 else float("nan"),
    }
