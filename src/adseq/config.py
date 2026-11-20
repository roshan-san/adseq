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
    "sj_mlp",  # Schroer & Just's MLP: imported by evaluate (not trained), skipped by tune
]


@dataclass
class Config:
    name: str = "main"
    """Experiment name: outputs go to runs/<name>/, and it names the MLflow experiment."""
    models: tuple[ModelName, ...] = get_args(ModelName)
    """Models to tune / evaluate."""
    val_frac: float = 0.2
    """Fraction of the training rows (the last ones) used for validation."""
    stride: int = 1
    """Evaluation: train and validate on every stride-th minute (test: always every minute).
    Ridge always fits on every minute, as in Schroer & Just's notebook."""

    n_trials: int = 50
    """Optuna trials per tuned model (completed or pruned)."""
    max_epochs: int = 50
    """Evaluation: upper limit on training epochs for a recurrent model."""
    patience: int = 10
    """Evaluation: stop after this many epochs without a better validation RMSE."""
    tune_stride: int = 10
    """Tuning: --stride while tuning. Neighbouring minutes are near-duplicates, so every 10th
    minute finds the same kind of models several times faster."""
    tune_max_epochs: int = 200
    """Tuning: --max-epochs while tuning (an epoch is 10x shorter at stride 10)."""
    tune_patience: int = 20
    """Tuning: --patience while tuning. At stride 1, 53-87 % of the epochs came after the best
    one (archived W&B run); a shorter epoch needs more of them to see a real plateau."""
    seeds: tuple[int, ...] = tuple(range(1, 6))
    """Seeds each model is evaluated with."""
    mlflow_uri: str = "sqlite:///runs/mlflow.db"
    """MLflow tracking URI."""

    @property
    def run_dir(self) -> Path:
        d = Path("runs") / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
