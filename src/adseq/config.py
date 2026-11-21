"""Experiment settings. tyro turns each Config field into a CLI flag (`--n-trials 50`) and the
docstring under it into the help text; everything else is fixed here."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, get_args

RUNS = Path("runs")  # every output; paths are relative to the repo root

ModelName = Literal[
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
    "sj_mlp",  # Schroer & Just's MLP: imported by evaluate (not trained), skipped by tune
]


class Training(NamedTuple):
    stride: int  # train / validate on every stride-th minute (test: every minute; ridge: always)
    max_epochs: int  # recurrent models only
    patience: int  # epochs without a better val RMSE before stopping


# Tuning uses every 10th minute: neighbouring minutes are near-duplicates, so it finds the same
# kind of models several times faster. Its epochs are 10x shorter, so it gets more of them and a
# longer patience (at stride 1, 53-87 % of the epochs came after the best one).
TUNE = Training(stride=10, max_epochs=200, patience=20)
EVAL = Training(stride=1, max_epochs=50, patience=10)  # what the reported numbers come from


@dataclass
class Config:
    name: str = "main"
    """Experiment name: runs/<name>/ and the MLflow experiment."""
    models: tuple[ModelName, ...] = get_args(ModelName)
    """Models to tune / evaluate."""
    n_trials: int = 50
    """Optuna trials per tuned model (completed or pruned)."""
    seeds: tuple[int, ...] = tuple(range(1, 6))
    """Seeds each model is evaluated with."""

    @property
    def run_dir(self) -> Path:
        d = RUNS / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
