"""Data: Schroer & Just's own processed SCADA data set, then splits and windows for our models.

Step 1, theirs: scripts/run_schroer.py runs their notebooks unmodified (doi:10.5281/zenodo.8360096)
on SCADA-raw.csv (Muscatine, IA WRRF, doi:10.25820/data.006715) ->
data/processed/SCADA-FINAL.csv: 491,761 minute rows, target "Forecast" = mean biogas
flow over the next 24 h (ft^3/min). We use the 36 inputs of their final MLP.

Step 2, ours:
  - test = their test set: the last 20 % of rows, every row scored (98,353 minutes)
  - validation = the last `val_frac` of their training rows; the 24 h before validation and
    before test are dropped from train / val, so no training label overlaps a later period
  - min-max scalers fit on training rows only
  - train / val: a fixed random 1/`stride` of the minutes (neighbours are near-duplicates)
  - each sample minute t gets a window of the features at t, t-60 min, ..., t-(L-1)*60 min
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

TARGET = "Forecast"  # mean biogas flow over the next 24 h, ft^3/min (their name)
PERSISTENCE = "Biogas_prev_hour_avg"  # mean of the previous clock hour (their baseline)
HORIZON = 1440  # minutes ahead the target reaches
STEP = 60  # minutes between window steps
WINDOWS = (12, 24, 48, 72)  # hours of history a recurrent model may be tuned to
MAX_WINDOW = max(WINDOWS)  # every sample has this much history, so any tuned window fits
PROCESSED = Path("data/processed/SCADA-FINAL.csv")  # their pretreatment notebook's output
MLP_DROPS = ["day_sin", "day_cos", "D2_TEMPERATURE", "Q-influent_MGD"]  # their SCADA-MLP, cell 0


def load_processed() -> pd.DataFrame:
    """Their model-ready data set, with the inputs of their final MLP."""
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} missing: run scripts/run_schroer.py (docstring)")
    return pd.read_csv(PROCESSED, index_col=0, parse_dates=True).drop(columns=MLP_DROPS)


def time_split(dates: pd.DatetimeIndex, val_frac: float) -> np.ndarray:
    """Their 80/20 test split (sklearn train_test_split, shuffle=False), our val split of the
    training rows, and 24-h "gap" rows before val and before test."""
    n = len(dates)
    n_train = n - math.ceil(0.2 * n)
    split = np.array(["train"] * n_train + ["test"] * (n - n_train), dtype=object)
    n_val = round(n_train * val_frac)
    split[n_train - n_val : n_train] = "val"
    gap = pd.Timedelta(minutes=HORIZON)
    test0, val0 = dates[n_train], dates[n_train - n_val]
    split[(split == "val") & (dates >= test0 - gap)] = "gap"
    split[(split == "train") & (dates >= val0 - gap)] = "gap"
    return split.astype(str)


@dataclass
class Dataset:
    dates: pd.DatetimeIndex  # sample minute t
    split: np.ndarray  # "train" / "val" / "test"
    grid: np.ndarray  # (minutes, F) scaled features on a regular minute grid
    pos: np.ndarray  # (N,) row of each sample in the grid
    y: np.ndarray  # (N,) scaled target: (y - y_offset) / y_scale
    y_true: np.ndarray  # (N,) their target, ft^3/min
    y_prev: np.ndarray  # (N,) their persistence baseline, ft^3/min
    y_offset: float
    y_scale: float
    n_features: int

    @property
    def X(self) -> np.ndarray:
        """(N, F) features of minute t -> ridge."""
        return self.grid[self.pos]

    def windows(self, idx: np.ndarray, window: int) -> np.ndarray:
        """(len(idx), L, F): features at t-(L-1)*60 min, ..., t-60 min, t."""
        if not 1 <= window <= MAX_WINDOW:  # longer would wrap negative indices to the test period
            raise ValueError(f"window {window} h outside 1..{MAX_WINDOW}")
        lags = STEP * np.arange(window - 1, -1, -1)
        return self.grid[self.pos[idx, None] - lags]

    def part(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.split == name)

    def scale(self, y: np.ndarray) -> np.ndarray:
        """ft^3/min -> the scaled units the models are trained in."""
        return (y - self.y_offset) / self.y_scale

    def unscale(self, y_scaled: np.ndarray) -> np.ndarray:
        """Scaled units -> ft^3/min."""
        return y_scaled * self.y_scale + self.y_offset


def build_dataset(df: pd.DataFrame, cfg: Config) -> Dataset:
    dates = pd.DatetimeIndex(df.index)
    split = time_split(dates, cfg.val_frac)
    features = df.columns.drop(TARGET)
    y_all = df[TARGET].to_numpy(float)

    # scalers fit on the training rows only
    train = df.loc[split == "train", features]
    lo, hi = train.min(), train.max()
    y_train = y_all[split == "train"]
    y_offset, y_scale = float(y_train.min()), float(y_train.max() - y_train.min())

    # regular minute grid so "60 min earlier" is one fixed offset; their dropped minutes are
    # carried forward from the last row (past values only)
    full = pd.date_range(dates[0], dates[-1], freq="min")
    grid = ((df[features] - lo) / (hi - lo).replace(0, 1)).reindex(full).ffill()
    pos_all = full.get_indexer(dates)

    # samples: every row of the test set; a fixed random 1/stride of the train / val minutes
    # (so every minute of the hour is seen); all with MAX_WINDOW hours of history
    sampled = np.random.default_rng(0).random(len(df)) < 1 / cfg.stride
    keep = (split == "test") | (np.isin(split, ("train", "val")) & sampled)
    keep &= pos_all >= STEP * (MAX_WINDOW - 1)
    if not keep[split == "test"].all():
        raise ValueError("a test row lacks a full window; lower MAX_WINDOW")

    y_true = y_all[keep]
    return Dataset(
        dates=dates[keep],
        split=split[keep],
        grid=grid.to_numpy(np.float32),
        pos=pos_all[keep],
        y=((y_true - y_offset) / y_scale).astype(np.float32),  # = Dataset.scale
        y_true=y_true,
        y_prev=df[PERSISTENCE].to_numpy(float)[keep],
        y_offset=y_offset,
        y_scale=y_scale,
        n_features=len(features),
    )
