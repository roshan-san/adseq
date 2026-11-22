from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed/SCADA-FINAL.csv")
VAL_FRAC = 0.2
TARGET = "Forecast"
PERSISTENCE = "Biogas_prev_hour_avg"
HORIZON = 1440
STEP = 60
WINDOWS = (12, 24, 48, 72)
MAX_WINDOW = max(WINDOWS)
MLP_DROPS = ["day_sin", "day_cos", "D2_TEMPERATURE", "Q-influent_MGD"]
STATIC = ("prev", "hour_")
CHUNK = 4096


def load_processed() -> pd.DataFrame:
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} missing: run `uv run prepare_data.py`")
    return pd.read_csv(PROCESSED, index_col=0, parse_dates=True).drop(columns=MLP_DROPS)


def time_split(dates: pd.DatetimeIndex) -> np.ndarray:
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
    dates: pd.DatetimeIndex
    split: np.ndarray
    grid: np.ndarray
    hourly: np.ndarray
    static: np.ndarray
    pos: np.ndarray
    y_true: np.ndarray
    y_prev: np.ndarray
    y_offset: float
    y_scale: float
    stride: int = 1
    y: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.y = self.scale(self.y_true).astype(np.float32)

    @property
    def n_features(self) -> int:
        return self.grid.shape[1]

    @property
    def n_seq(self) -> int:
        return self.hourly.shape[1]

    @property
    def n_static(self) -> int:
        return self.static.shape[1]

    @property
    def X(self) -> np.ndarray:
        return self.grid[self.pos]

    def windows(self, idx: np.ndarray, window: int) -> np.ndarray:
        if not 1 <= window <= MAX_WINDOW:
            raise ValueError(f"window {window} h outside 1..{MAX_WINDOW}")
        lags = STEP * np.arange(window - 1, -1, -1)
        return self.hourly[self.pos[idx, None] - lags]

    def part(self, name: str) -> np.ndarray:
        rows = self.split == name
        if name != "test":
            rows &= self.pos % self.stride == 0
        return np.flatnonzero(rows)

    def pretest(self) -> np.ndarray:
        return np.concatenate([self.part("train"), self.part("val")])

    def base_level(self, residual: bool) -> np.ndarray:
        return self.scale(self.y_prev) if residual else np.zeros_like(self.y)

    def scale(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_offset) / self.y_scale

    def unscale(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_scale + self.y_offset


def chunks(idx: np.ndarray) -> list[np.ndarray]:
    return np.array_split(idx, max(1, len(idx) // CHUNK))


def build_dataset(df: pd.DataFrame, stride: int) -> Dataset:
    dates = pd.DatetimeIndex(df.index)
    split = time_split(dates)
    features = df.columns.drop(TARGET)
    y_all = df[TARGET].to_numpy(float)

    train = df.loc[split == "train", features]
    lo, hi = train.min(), train.max()
    y_train = y_all[split == "train"]
    y_offset, y_scale = float(y_train.min()), float(y_train.max() - y_train.min())

    grid = ((df[features] - lo) / (hi - lo).replace(0, 1)).asfreq("min").ffill()
    pos_all = grid.index.get_indexer(dates)
    static = np.array([any(k in c for k in STATIC) for c in features])
    hourly = grid.loc[:, ~static].rolling(STEP, min_periods=1).mean()
    grid_np = grid.to_numpy(np.float32)

    keep = (split != "gap") & (pos_all >= STEP * (MAX_WINDOW - 1))
    if not keep[split == "test"].all():
        raise ValueError("a test row lacks a full window; lower MAX_WINDOW")

    return Dataset(
        dates=dates[keep],
        split=split[keep],
        grid=grid_np,
        hourly=hourly.to_numpy(np.float32),
        static=grid_np[pos_all[keep]][:, static],
        pos=pos_all[keep],
        y_true=y_all[keep],
        y_prev=df[PERSISTENCE].to_numpy(float)[keep],
        y_offset=y_offset,
        y_scale=y_scale,
        stride=stride,
    )
