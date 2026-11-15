"""Every experiment setting, in one flat dataclass. tyro turns each field into a CLI flag
(`--window 30`) and the docstring under it into the help text. See docs/paper_map.md.

Fixed choices copied from the base paper (lags, rolling windows, ...) are constants in data.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

ModelName = Literal["persistence", "fnn_paper", "fnn_tuned", "window_fnn", "rnn", "gru", "lstm"]


@dataclass
class Config:
    name: str = "main"
    models: tuple[ModelName, ...] = get_args(ModelName)

    data_path: Path = Path("data/raw/muscatine_labs.csv")
    test_start: str = "2022-04-01"
    val_frac: float = 0.2
    window: int = 15
    seq_inputs: Literal["full", "raw"] = "full"

    n_trials: int = 40
    max_epochs: int = 1000
    patience: int = 60
    seeds: tuple[int, ...] = tuple(range(10))
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    @property
    def run_dir(self) -> Path:
        d = Path("runs") / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
