from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, get_args

RUNS = Path("runs")


class Training(NamedTuple):
    stride: int
    max_epochs: int
    patience: int


TUNE = Training(stride=10, max_epochs=200, patience=20)
EVAL = Training(stride=1, max_epochs=50, patience=10)


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

UNTUNED = ("persistence", "ridge", "sj_mlp")  # nothing to tune
ONE_SEED = ("persistence", "ridge")  # deterministic: one seed


@dataclass
class Config:
    name: str = "main"
    models: tuple[ModelName, ...] = get_args(ModelName)
    n_trials: int = 50
    seeds: tuple[int, ...] = tuple(range(1, 6))

    @property
    def run_dir(self) -> Path:
        d = RUNS / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
