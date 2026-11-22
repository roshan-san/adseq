from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, get_args

RUNS = Path("runs")


class Training(NamedTuple):
    stride: int
    max_epochs: int
    patience: int


# the same epoch length (EPOCH_WINDOWS) and limits in both, so an epoch count found in one carries
# over; only how many training minutes there are to draw from differs
TUNE = Training(stride=10, max_epochs=200, patience=20)
EVAL = Training(stride=1, max_epochs=200, patience=20)
EPOCH_WINDOWS = 30_000  # training windows per recurrent epoch: ~all of train at stride 10
PROTOCOL = 2  # bumped when a change makes earlier Optuna studies incomparable


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


@dataclass
class Config:
    name: str = "main"
    models: tuple[ModelName, ...] = get_args(ModelName)
    n_trials: int = 30
    seeds: tuple[int, ...] = tuple(range(1, 6))
    trial_seeds: tuple[int, ...] = (0, 1, 2)
    """Seeds each Optuna trial trains with; the trial's score is their mean val RMSE."""
    max_epochs: int | None = None
    """Caps the recurrent models' epochs in tune and evaluate, for a quick test run; default: TUNE's
    and EVAL's limits."""

    def training(self, base: Training) -> Training:
        """TUNE or EVAL with --max-epochs applied."""
        return base if self.max_epochs is None else base._replace(max_epochs=self.max_epochs)

    @property
    def run_dir(self) -> Path:
        d = RUNS / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
