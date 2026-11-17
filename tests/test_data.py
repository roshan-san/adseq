"""Checks on Schroer & Just's processed data set and our splits / windows on top of it."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adseq.config import Config
from adseq.data import (
    HORIZON,
    MAX_WINDOW,
    PROCESSED,
    STEP,
    TARGET,
    build_dataset,
    load_processed,
    time_split,
)
from adseq.metrics import regression_metrics
from adseq.train import train_and_predict

CFG = Config()
if not PROCESSED.exists():
    pytest.skip("run `adseq fetch` and scripts/run_schroer.py first", allow_module_level=True)


@pytest.fixture(scope="module")
def df():
    return load_processed()


@pytest.fixture(scope="module")
def ds(df):
    return build_dataset(df, CFG)


def test_processed_data_matches_the_paper(df, ds):
    assert len(df) == 491_761 and df.shape[1] - 1 == 36  # their n and feature count
    te = ds.part("test")
    assert len(te) == 98_353  # every row of their test set is scored
    m = regression_metrics(ds.y_true[te], ds.y_prev[te])
    assert round(m["mape"], 1) == 17.9 and round(m["teei"], 2) == 1.45  # their persistence


def test_split_is_chronological_with_gaps(ds):
    tr, va, te = (ds.dates[ds.part(p)] for p in ("train", "val", "test"))
    gap = pd.Timedelta(minutes=HORIZON)
    assert tr.max() + gap <= va.min() and va.max() + gap <= te.min()  # labels never overlap


def test_windows_look_back_in_hourly_steps(ds):
    idx = ds.part("test")[:5]
    w = ds.windows(idx, 24)
    assert w.shape == (5, 24, ds.n_features) and not np.isnan(w).any()
    assert (w[:, -1] == ds.X[idx]).all()  # last step is minute t itself
    assert (w[:, -2] == ds.grid[ds.pos[idx] - STEP]).all()  # one step = 60 min earlier


def test_scaling_is_fit_on_the_training_rows_only(df, ds):
    train_rows = df[TARGET][time_split(df.index, CFG.val_frac) == "train"]
    assert ds.y_offset == train_rows.min()
    assert np.isclose(ds.y_scale, train_rows.max() - train_rows.min())
    assert np.allclose(ds.unscale(ds.y), ds.y_true, atol=1e-3)


HP = {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 256, "dropout": 0.0, "loss": "l1"}
HP |= {"hidden_size": 8, "num_layers": 1, "window": 12}
HP |= {"num_leaves": 15, "min_data_in_leaf": 20, "feature_fraction": 0.5}
HP |= {"bagging_fraction": 0.8, "lambda_l2": 1e-3}


@pytest.mark.parametrize("name", ["ridge", "gru_res", "lgbm", "lgbm_res"])
def test_models_train_and_predict(ds, name):
    pred, _, _ = train_and_predict(name, HP, ds, replace(CFG, max_epochs=1, patience=1), seed=0)
    assert pred.shape == ds.y_true.shape and np.isfinite(pred).all()


def test_untrained_residual_net_is_persistence(ds):
    pred, info, _ = train_and_predict("gru_res", HP, ds, replace(CFG, max_epochs=0), seed=0)
    assert info["epochs"] == 0
    assert np.allclose(pred, ds.y_prev)


def test_scale_and_unscale_are_inverses(ds):
    assert np.allclose(ds.unscale(ds.scale(ds.y_true)), ds.y_true)
    assert np.allclose(ds.scale(ds.y_true), ds.y, atol=1e-6)


def test_windows_never_exceed_the_history_every_sample_has(ds):
    with pytest.raises(ValueError):
        ds.windows(ds.part("train")[:1], MAX_WINDOW + 1)


def test_lgbm_is_seeded_and_beats_persistence_on_val(ds):
    hp = HP | {"lr": 0.05, "loss": "mse"}
    a, info, _ = train_and_predict("lgbm_res", hp, ds, CFG, seed=0)
    b, _, _ = train_and_predict("lgbm_res", hp, ds, CFG, seed=0)
    assert np.array_equal(a, b) and info["epochs"] > 0
    va = ds.part("val")
    rmse = regression_metrics(ds.y_true[va], a[va])["rmse"]
    assert rmse < regression_metrics(ds.y_true[va], ds.y_prev[va])["rmse"]
