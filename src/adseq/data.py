"""Data: Schroer & Just's own processed SCADA data set, then splits and windows for our models.

Step 1, theirs: scripts/run_schroer.py runs their notebooks unmodified (doi:10.5281/zenodo.8360096)
on SCADA-raw.csv (Muscatine, IA WRRF, doi:10.25820/data.006715) ->
data/processed/SCADA-FINAL.csv: 491,761 minute rows, target "Forecast" = mean biogas
flow over the next 24 h (ft^3/min). We use the 36 inputs of their final MLP.

Step 2, ours:
  - test = their test set: the last 20 % of rows, every row scored (98,353 minutes)
  - validation = the last VAL_FRAC of their training rows; the 24 h before validation and
    before test are dropped from train / val, so no training label overlaps a later period
  - min-max scalers fit on training rows only
  - every minute of train / val / test is a sample (except the first MAX_WINDOW - 1 hours,
    which lack a full window)
  - each sample minute t gets a window of the features at t, t-60 min, ..., t-(L-1)*60 min
  - the windowed models train and validate on every `stride`-th minute only (neighbouring
    minutes are near-duplicates); test is always scored on every minute
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed/SCADA-FINAL.csv")  # their pretreatment notebook's output
VAL_FRAC = 0.2  # the last 20 % of the training rows validate
TARGET = "Forecast"  # mean biogas flow over the next 24 h, ft^3/min (their name)
PERSISTENCE = "Biogas_prev_hour_avg"  # mean of the previous clock hour (their baseline)
HORIZON = 1440  # minutes ahead the target reaches
STEP = 60  # minutes between window steps
WINDOWS = (12, 24, 48, 72)  # hours of history a recurrent model may be tuned to
MAX_WINDOW = max(WINDOWS)  # every sample has this much history, so any tuned window fits
MLP_DROPS = ["day_sin", "day_cos", "D2_TEMPERATURE", "Q-influent_MGD"]  # their SCADA-MLP, cell 0
CHUNK = 4096  # samples per forward pass when predicting: all ~490k windows would not fit


def load_processed() -> pd.DataFrame:
    """Their model-ready data set, with the inputs of their final MLP."""
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} missing: run scripts/run_schroer.py (docstring)")
    return pd.read_csv(PROCESSED, index_col=0, parse_dates=True).drop(columns=MLP_DROPS)


def time_split(dates: pd.DatetimeIndex) -> np.ndarray:
    """Their 80/20 test split (sklearn train_test_split, shuffle=False), our val split of the
    training rows the same way, and 24-h "gap" rows before val and before test."""
    rows = np.arange(len(dates))
    train_rows, test_rows = train_test_split(rows, test_size=0.2, shuffle=False)
    train_rows, val_rows = train_test_split(train_rows, test_size=VAL_FRAC, shuffle=False)
    split = np.full(len(rows), "train", dtype="<U5")
    split[val_rows], split[test_rows] = "val", "test"
    gap = pd.Timedelta(minutes=HORIZON)
    test0, val0 = dates[test_rows[0]], dates[val_rows[0]]
    split[(split == "val") & (dates >= test0 - gap)] = "gap"
    split[(split == "train") & (dates >= val0 - gap)] = "gap"
    return split


@dataclass
class Dataset:
    """N samples (one per minute t) over a (minutes, F) feature grid."""

    dates: pd.DatetimeIndex  # (N,) sample minute t
    split: np.ndarray  # (N,) "train" / "val" / "test"
    grid: np.ndarray  # (minutes, F) scaled features on a regular minute grid
    pos: np.ndarray  # (N,) row of each sample in the grid
    y_true: np.ndarray  # (N,) their target, ft^3/min
    y_prev: np.ndarray  # (N,) their persistence baseline, ft^3/min
    y_offset: float  # target min-max scaler, fit on the training rows
    y_scale: float
    stride: int = 1  # part("train") / part("val") keep every stride-th minute
    y: np.ndarray = field(init=False)  # (N,) scaled target, what the models are trained on

    def __post_init__(self) -> None:
        self.y = self.scale(self.y_true).astype(np.float32)

    @property
    def n_features(self) -> int:
        return self.grid.shape[1]

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
        """Sample indices of one split. Train and val are thinned to every `stride`-th minute;
        test always keeps every minute."""
        rows = self.split == name
        if name != "test":
            rows &= self.pos % self.stride == 0
        return np.flatnonzero(rows)

    def base_level(self, residual: bool) -> np.ndarray:
        """What a model's output is added to (scaled units): persistence for the residual models,
        zero for the others."""
        return self.scale(self.y_prev) if residual else np.zeros_like(self.y)

    def scale(self, y: np.ndarray) -> np.ndarray:
        """ft^3/min -> the scaled units the models are trained in."""
        return (y - self.y_offset) / self.y_scale

    def unscale(self, y_scaled: np.ndarray) -> np.ndarray:
        """Scaled units -> ft^3/min."""
        return y_scaled * self.y_scale + self.y_offset


def chunks(idx: np.ndarray) -> list[np.ndarray]:
    """idx in pieces of about CHUNK samples."""
    return np.array_split(idx, max(1, len(idx) // CHUNK))


def build_dataset(df: pd.DataFrame, stride: int) -> Dataset:
    dates = pd.DatetimeIndex(df.index)
    split = time_split(dates)
    features = df.columns.drop(TARGET)
    y_all = df[TARGET].to_numpy(float)

    # scalers fit on the training rows only
    train = df.loc[split == "train", features]
    lo, hi = train.min(), train.max()
    y_train = y_all[split == "train"]
    y_offset, y_scale = float(y_train.min()), float(y_train.max() - y_train.min())

    # regular minute grid so "60 min earlier" is one fixed offset; their dropped minutes are
    # carried forward from the last row (past values only)
    grid = ((df[features] - lo) / (hi - lo).replace(0, 1)).asfreq("min").ffill()
    pos_all = grid.index.get_indexer(dates)

    # samples: every train / val / test minute (not the gaps) with MAX_WINDOW hours of history
    keep = (split != "gap") & (pos_all >= STEP * (MAX_WINDOW - 1))
    if not keep[split == "test"].all():
        raise ValueError("a test row lacks a full window; lower MAX_WINDOW")

    return Dataset(
        dates=dates[keep],
        split=split[keep],
        grid=grid.to_numpy(np.float32),
        pos=pos_all[keep],
        y_true=y_all[keep],
        y_prev=df[PERSISTENCE].to_numpy(float)[keep],
        y_offset=y_offset,
        y_scale=y_scale,
        stride=stride,
    )
