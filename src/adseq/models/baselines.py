"""The untuned baselines: persistence (nothing to fit) and ridge (no training loop)."""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import Dataset

RIDGE_CV_GAP = 300  # their notebook: rows left out between each CV fold's train and val rows


def fit_ridge(ds: Dataset) -> tuple[Pipeline, dict]:
    """Their linear model (SCADA-linear-regression.ipynb): StandardScaler + RidgeCV with its
    default penalties (0.1, 1, 10); the one with the best R2 over 5 TimeSeriesSplit folds is
    refit on all rows. Fit on every pre-test row (train + val, as theirs saw their whole training
    set), so no val split is needed; every minute, whatever the stride. float64, as the lag
    features are collinear."""
    rows = np.flatnonzero(ds.split != "test")
    model = make_pipeline(StandardScaler(), RidgeCV(cv=TimeSeriesSplit(gap=RIDGE_CV_GAP)))
    model.fit(ds.X[rows].astype(float), ds.y[rows].astype(float))
    ridge = model.named_steps["ridgecv"]
    info = {"epochs": 0, "alpha": float(ridge.alpha_), "n_params": ridge.coef_.size + 1}
    return model, info


def predict_ridge(model: Pipeline, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample."""
    return ds.unscale(np.asarray(model.predict(ds.X.astype(float))))


def predict_persistence(ds: Dataset) -> np.ndarray:
    """The mean of the previous clock hour (ft^3/min), for every sample."""
    return ds.y_prev
