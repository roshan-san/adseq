from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, get_args

RUNS = Path("runs")


class Training(NamedTuple):
    stride: int
    max_epochs: int
    patience: int


TUNE = Training(stride=10, max_epochs=200, patience=20)
EVAL = Training(stride=1, max_epochs=200, patience=20)
EPOCH_WINDOWS = 30_000
PROTOCOL = 2


ModelName = Literal[
    "sj_mlp",
    "persistence",
    "ridge",
    "lgbm",
    "lgbm_res",
    "rnn",
    "gru",
    "lstm",
    "rnn_res",
    "gru_res",
    "lstm_res",
]

UNTUNED = ("persistence", "ridge", "sj_mlp")


@dataclass
class Config:
    name: str = "main"
    models: tuple[ModelName, ...] = get_args(ModelName)
    n_trials: int = 30
    seeds: tuple[int, ...] = tuple(range(1, 6))
    trial_seeds: tuple[int, ...] = (0, 1, 2)
    max_epochs: int | None = None

    def training(self, base: Training) -> Training:
        return base if self.max_epochs is None else base._replace(max_epochs=self.max_epochs)

    @property
    def run_dir(self) -> Path:
        d = RUNS / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
