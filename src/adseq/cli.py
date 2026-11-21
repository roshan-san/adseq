"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

import os
from dataclasses import replace
from typing import Annotated

import mlflow
import tyro
from mlflow.lightgbm import autolog as autolog_lightgbm

from .config import EVAL, RUNS, TUNE, UNTUNED, Config
from .data import build_dataset, load_processed
from .report import make_report
from .run import evaluate, tune

# flags read --n-trials instead of --cfg.n-trials
Cfg = Annotated[Config, tyro.conf.arg(name="")]
MLFLOW_URI = "sqlite:///runs/mlflow.db"


def start_tracking(cfg: Config) -> None:
    """Log to MLFLOW_URI, in the experiment named --name; LightGBM logs its own curves."""
    RUNS.mkdir(exist_ok=True)  # MLFLOW_URI's sqlite file lives there
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # no bars in the logs
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg.name)
    autolog_lightgbm(log_models=False, log_datasets=False, silent=True)


def run_cmd(cfg: Cfg) -> None:
    """For each model: tune it (TUNE settings; persistence and ridge have nothing to tune), then
    evaluate it with every seed (EVAL settings). Re-running resumes tuning where it stopped."""
    start_tracking(cfg)
    tune_ds = build_dataset(load_processed(), TUNE.stride)  # built once, shared by every model
    eval_ds = replace(tune_ds, stride=EVAL.stride)  # the same samples, every minute
    for name in cfg.models:
        if name not in UNTUNED:
            tune(cfg, tune_ds, name)
        evaluate(cfg, eval_ds, name)


def report_cmd(cfg: Cfg) -> None:
    """Results table (incl. Schroer & Just's own MLP) and figure."""
    start_tracking(cfg)
    make_report(cfg)


def main() -> None:
    tyro.extras.subcommand_cli_from_dict({"run": run_cmd, "report": report_cmd})
