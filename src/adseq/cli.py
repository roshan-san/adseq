"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

import os
from typing import Annotated

import mlflow
import tyro
from mlflow.lightgbm import autolog as autolog_lightgbm

from .config import RUNS, Config
from .data import build_dataset, fetch, load_processed
from .models import MODELS
from .report import make_report
from .run import evaluate, tune

# flags read --n-trials instead of --cfg.n-trials
Cfg = Annotated[Config, tyro.conf.arg(name="")]


def start_tracking(cfg: Config) -> None:
    """Log to cfg.mlflow_uri, in the experiment named --name; LightGBM logs its own curves."""
    RUNS.mkdir(exist_ok=True)  # the default mlflow_uri's sqlite file lives there
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # no bars in the logs
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.name)
    autolog_lightgbm(log_models=False, log_datasets=False, silent=True)


def fetch_cmd(cfg: Cfg) -> None:
    """Download SCADA-raw.csv (85 MB) and Schroer & Just's code, skipping whatever is already
    there. Then run scripts/run_schroer.py (see its docstring) to build
    data/processed/SCADA-FINAL.csv; the raw file is not downloaded once that exists."""
    fetch()


def tune_cmd(cfg: Cfg) -> None:
    """Optuna search for every tuned model (same trial budget for all), with the --tune.*
    stride and epoch limits."""
    start_tracking(cfg)
    # built once, shared by every model
    ds = build_dataset(load_processed(), cfg.val_frac, cfg.tune.stride)
    for name in cfg.models:
        if MODELS[name].tuned:
            tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    """Train each model with every seed using its best hyper-parameters; score on the test period."""
    start_tracking(cfg)
    # built once, shared by every model
    ds = build_dataset(load_processed(), cfg.val_frac, cfg.eval.stride)
    for name in cfg.models:
        evaluate(cfg, ds, name)


def report_cmd(cfg: Cfg) -> None:
    """Results table (incl. Schroer & Just's own MLP) and figure."""
    start_tracking(cfg)
    make_report(cfg)


def run_all_cmd(cfg: Cfg) -> None:
    """tune -> evaluate -> report."""
    tune_cmd(cfg)
    evaluate_cmd(cfg)
    report_cmd(cfg)


def main() -> None:
    commands = {
        "fetch": fetch_cmd,
        "tune": tune_cmd,
        "evaluate": evaluate_cmd,
        "report": report_cmd,
        "run-all": run_all_cmd,
    }
    tyro.extras.subcommand_cli_from_dict(commands)
