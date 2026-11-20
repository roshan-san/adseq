"""Every experiment setting, in one flat dataclass. tyro turns each field into a CLI flag
(`--n-trials 50`) and the docstring under it into the help text."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

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
]


@dataclass
class Config:
    name: str = "main"
    """Experiment name: outputs go to runs/<name>/, and it names the MLflow experiment."""
    models: tuple[ModelName, ...] = get_args(ModelName)
    """Models to tune / evaluate."""
    val_frac: float = 0.2
    """Fraction of the training rows (the last ones) used for validation."""

    n_trials: int = 50
    """Optuna trials per tuned model (completed or pruned)."""
    max_epochs: int = 50
    """Upper limit on training epochs for a recurrent model."""
    patience: int = 10
    """Stop after this many epochs without a better validation RMSE."""
    seeds: tuple[int, ...] = tuple(range(1, 6))
    """Seeds each model is evaluated with."""
    mlflow_uri: str = "sqlite:///runs/mlflow.db"
    """MLflow tracking URI."""

    @property
    def run_dir(self) -> Path:
        d = Path("runs") / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
