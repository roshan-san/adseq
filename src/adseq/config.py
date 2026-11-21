"""Every experiment setting. tyro turns each field into a CLI flag (`--n-trials 50`,
`--tune.max-epochs 200`) and the docstring under it into the help text."""

from dataclasses import dataclass, field
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
class Training:
    """How a model is trained: once with these settings while tuning, once while evaluating."""

    stride: int
    """Train and validate on every stride-th minute (test: always every minute). Ridge always
    fits on every minute, as in Schroer & Just's notebook."""
    max_epochs: int
    """Upper limit on training epochs for a recurrent model."""
    patience: int
    """Stop after this many epochs without a better validation RMSE."""


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
    seeds: tuple[int, ...] = tuple(range(1, 6))
    """Seeds each model is evaluated with."""
    mlflow_uri: str = "sqlite:///runs/mlflow.db"
    """MLflow tracking URI."""
    tune: Training = field(default_factory=lambda: Training(stride=10, max_epochs=200, patience=20))
    """Tuning. Neighbouring minutes are near-duplicates, so every 10th minute finds the same
    kind of models several times faster; an epoch is then 10x shorter, so more of them are
    allowed. At stride 1, 53-87 % of the epochs came after the best one (archived W&B run): a
    shorter epoch needs a longer patience to see a real plateau."""
    eval: Training = field(default_factory=lambda: Training(stride=1, max_epochs=50, patience=10))
    """Evaluation: every model with every seed, using its best hyper-parameters."""

    @property
    def run_dir(self) -> Path:
        d = Path("runs") / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
