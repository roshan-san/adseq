"""Leakage checks on the real Muscatine data (cite these in the paper's protocol section)."""

from pathlib import Path

import numpy as np
import pytest

from adseq.config import Config
from adseq.data import build_dataset, clean_data, make_features

CFG = Config()
if not Path(CFG.data_path).exists():
    pytest.skip("run `uv run adseq fetch` first", allow_module_level=True)


@pytest.fixture(scope="module")
def daily():
    return clean_data(CFG.data_path)


def test_features_do_not_use_the_future(daily):
    day = daily.index[700]
    before = make_features(daily).loc[day]
    changed = daily.copy()
    changed.loc[day:, "Biogas"] *= 10  # tamper with day t and later
    after = make_features(changed).loc[day]
    assert before.equals(after)


def test_split_is_chronological_and_test_is_held_out(daily):
    ds = build_dataset(daily, CFG)
    tr, va, te = (ds.dates[ds.part(p)] for p in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert te.min() >= np.datetime64(CFG.test_start)
    assert not np.isnan(ds.X_seq).any()


def test_every_target_day_has_a_full_window(daily):
    ds = build_dataset(daily, CFG)
    assert ds.X_seq.shape[1] == CFG.window
    assert (ds.X_seq[:, -1, :] == ds.X).all()  # last step of the window is day t


def test_scaling_is_fit_on_the_training_period_only(daily):
    ds = build_dataset(daily, CFG)
    tr = ds.part("train")
    assert ds.y[tr].min() == 0 and abs(ds.y[tr].max() - 1) < 1e-6
    assert np.allclose(ds.unscale(ds.y), ds.y_true, atol=1e-3)


def test_implausible_digester_temperatures_are_removed(daily):
    assert daily["Dig2-T_degF"].max() <= 105
    # the 121.5 / 126.1 F readings are dropped, then filled with the last good value (04-09)
    assert (
        daily.loc["2022-04-10":"2022-04-11", "Dig2-T_degF"]
        == daily.loc["2022-04-09", "Dig2-T_degF"]
    ).all()
