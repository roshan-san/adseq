"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

import os
from typing import Annotated

import mlflow
import tyro
from mlflow.lightgbm import autolog as autolog_lightgbm

from .config import EVAL, RUNS, TUNE, UNTUNED, Config
from .data import build_dataset, load_processed
from .evaluate import evaluate
from .report import make_report
from .tune import tune

# flags read --n-trials instead of --cfg.n-trials
Cfg = Annotated[Config, tyro.conf.arg(name="")]
MLFLOW_URI = "sqlite:///runs/mlflow.db"


def start_tracking(cfg: Config) -> None:
    RUNS.mkdir(exist_ok=True)  # MLFLOW_URI's sqlite file lives there
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # no bars in the logs
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg.name)
    autolog_lightgbm(log_models=False, log_datasets=False, silent=True)


def tune_cmd(cfg: Cfg) -> None:
    """Optuna search for each model (persistence, ridge and sj_mlp have nothing to tune). Re-running
    resumes it."""
    start_tracking(cfg)
    ds = build_dataset(load_processed(), TUNE.stride)  # built once, shared by every model
    for name in cfg.models:
        if name not in UNTUNED:
            tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    """Train each model with every seed using its best hyper-parameters; score every split."""
    start_tracking(cfg)
    ds = build_dataset(load_processed(), EVAL.stride)
    for name in cfg.models:
        evaluate(cfg, ds, name)


def report_cmd(cfg: Cfg) -> None:
    """Results table and figure, read from MLflow (the newest finished evaluations)."""
    start_tracking(cfg)
    make_report(cfg)


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {"tune": tune_cmd, "evaluate": evaluate_cmd, "report": report_cmd}
    )
