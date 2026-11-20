"""Every experiment setting, in one flat dataclass. tyro turns each field into a CLI flag
(`--stride 30`) and the docstring under it into the help text."""

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
    models: tuple[ModelName, ...] = get_args(ModelName)

    data_path: Path = Path("data/raw/muscatine_scada.csv")
    """SCADA-raw.csv; scripts/run_schroer.py turns it into their processed data set."""
    val_frac: float = 0.2
    stride: int = 60
    """Train / val use a random 1/stride of their minutes (every test minute is always scored)."""

    n_trials: int = 20
    max_epochs: int = 300
    patience: int = 20
    seeds: tuple[int, ...] = tuple(range(5))
    mlflow_uri: str = "sqlite:///runs/mlflow.db"
    """MLflow tracking store, one for every --name (each is an experiment). Browse it with
    `uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db`."""

    @property
    def run_dir(self) -> Path:
        d = Path("runs") / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
