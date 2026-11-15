"""Paper Sec. III-IV-A: load data, build features, split by time, scale, make windows.

Dataset: Muscatine, IA WRRF, LABS-raw.csv (Schroer & Just 2023, doi:10.25820/data.006715).
Year/Month/Day, 21 operational columns, target Biogas = daily mean biogas flow (ft^3/min):
  feeds      V-TWAS_gal, V-PS_gal, V-HSW_gal (sludges, high-strength waste), FOG_gal
  digesters  Dig1/Dig2 -T_degF, -pH, -alk_mgL, -VFA_mgL, -FOS-TAC; SRT (days)
  feed lab   HSW-VS_percent, HSW-COD_mgL, TWAS-VS_percent, PS-VS_percent
  plant      Q-influent_MGD, BOD-influent_mgL
(The data dictionary's descriptions of TWAS-VS and PS-VS are swapped; trust the names.)
The authors' pre-cleaning: no rows 2021-04-06..06-27 (meter outage), no VFA before 2020-04-21.

Rules that keep the comparison fair and leakage-free (state these in the paper):
  1. Data is put on a calendar-daily index, so "lag 7" really is 7 days ago.
  2. Every target-derived input uses only days < t.
  3. Short gaps in inputs are forward-filled (past values only); labels are never filled.
  4. Scalers are fit on the training period only (standard; min-max for the base paper's FNN).
  5. All models are scored on exactly the same test days.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import Config

TARGET = "Biogas"  # daily mean biogas flow, ft^3/min

# Fixed by the base paper (Long et al., 2026, Table 1 / Sec. 2)
LAGS = (1, 2, 7, 15)  # target lags in days (lag_1..lag_4)
ROLL_WINDOWS = (5, 10, 15)  # rolling mean/std windows in days
FFILL_LIMIT = 15  # max days an *input* is carried forward over a gap; labels never are

# Physically implausible readings -> missing (then forward-filled like any gap). A 1,840 m^3
# mesophilic tank cannot swing 30 F in a day: Dig2-T reads 95.0, 121.5, 126.1, 92.1 on
# 2022-04-09..12. Cold spells down to 85 F (winter 2020/21) are plausible and kept.
VALID_RANGE = {"Dig1-T_degF": (80.0, 105.0), "Dig2-T_degF": (80.0, 105.0)}


def clean_data(path: Path) -> pd.DataFrame:

    # load the csv
    df = pd.read_csv(path)

    # build date column
    dates = pd.to_datetime(df[["Year", "Month", "Day"]])
    df = df.drop(columns=["Year", "Month", "Day"])

    # drop text columns
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    # set date as index
    df.index = pd.DatetimeIndex(dates)
    df = df[~df.index.duplicated()].sort_index()

    # remove impossible values
    for col, (lo, hi) in VALID_RANGE.items():
        if col in df:
            df[col] = df[col].where(df[col].between(lo, hi))

    # one row per every day,with NaN for missing vals
    df = df.asfreq("D")

    # forward-fill short gaps
    inputs = df.columns.drop(TARGET)
    df[inputs] = df[inputs].ffill(limit=FFILL_LIMIT)

    return df


def make_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:

    # existing features
    X = df.drop(columns=TARGET)

    # calender encoding
    t = df.index
    X["month_sin"] = np.sin(2 * np.pi * (t.month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (t.month - 1) / 12)
    X["doy_sin"] = np.sin(2 * np.pi * (t.day_of_year - 1) / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * (t.day_of_year - 1) / 365.25)


    # previous output as feature
    y_past = df[TARGET].ffill(limit=FFILL_LIMIT).shift(1)

    # lag and rolling features
    for i, k in enumerate(LAGS, start=1):
        X[f"lag_{i}"] = y_past.shift(k - 1)  # = y[t-k]
    for w in ROLL_WINDOWS:
        X[f"rolling_mean_{w}"] = y_past.rolling(w, min_periods=w // 2).mean()
        X[f"rolling_std_{w}"] = y_past.rolling(w, min_periods=w // 2).std()
    return X


@dataclass
class Dataset:
    target: str
    dates: pd.DatetimeIndex  # day t of each sample
    split: np.ndarray  # "train" / "val" / "test"
    X: np.ndarray  # (N, F)     features of day t          -> FNN input
    X_seq: np.ndarray  # (N, L, F') features of days t-L+1..t -> RNN/GRU/LSTM input
    y: np.ndarray  # (N,) scaled target: (y - y_offset) / y_scale
    y_true: np.ndarray  # (N,) target in original units
    y_prev: np.ndarray  # (N,) yesterday's target (persistence baseline)
    y_offset: float
    y_scale: float
    feature_names: list[str]

    def part(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.split == name)

    def unscale(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_scale + self.y_offset


def build_dataset(
    daily: pd.DataFrame,
    cfg: Config,
    scaling: Literal["standard", "minmax"] = "standard",
) -> Dataset:
    X = make_features(daily, cfg)
    y = daily[TARGET]

    # usable day t: label exists and the last L days have complete inputs
    row_ok = X.notna().all(axis=1)
    history_ok = row_ok.rolling(cfg.window).sum() == cfg.window
    dates = X.index[y.notna() & history_ok]

    # chronological split: train (first 80% before test_start) | val (last 20%) | test
    split = np.where(dates >= pd.Timestamp(cfg.test_start), "test", "train")
    earlier = np.flatnonzero(split == "train")
    n_val = round(len(earlier) * cfg.val_frac)
    split[earlier[len(earlier) - n_val :]] = "val"

    # scaling fitted on the training period only: x -> (x - offset) / scale
    train_dates = dates[split == "train"]
    fit, y_fit = X[: train_dates.max()], y[train_dates]
    if scaling == "standard":
        offset, scale = fit.mean(), fit.std()
        y_offset, y_scale = float(y_fit.mean()), float(y_fit.std())
    else:
        offset, scale = fit.min(), fit.max() - fit.min()
        y_offset, y_scale = float(y_fit.min()), float(y_fit.max() - y_fit.min())
    X = X.loc[:, scale > 0]  # drop columns that are constant during training
    Xs = (X - offset[X.columns]) / scale[X.columns]

    # per-day inputs for the sequence models
    if cfg.seq_inputs == "full":
        seq = Xs
    else:  # "raw": no hand-made lag features, the RNN must learn the history itself
        seq = Xs.drop(columns=[c for c in Xs if c.startswith(("lag_", "rolling_"))])
        seq["y_prev"] = Xs["lag_1"]
    seq_arr = seq.to_numpy(np.float32)
    pos = X.index.get_indexer(dates)
    X_seq = np.stack([seq_arr[p - cfg.window + 1 : p + 1] for p in pos])

    return Dataset(
        target=TARGET,
        dates=dates,
        split=split,
        X=Xs.loc[dates].to_numpy(np.float32),
        X_seq=X_seq,
        y=((y[dates] - y_offset) / y_scale).to_numpy(np.float32),
        y_true=y[dates].to_numpy(float),
        y_prev=X.loc[dates, "lag_1"].to_numpy(float),
        y_offset=y_offset,
        y_scale=y_scale,
        feature_names=list(Xs.columns),
    )
