"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

import os
import subprocess
import sys
from typing import Annotated

import mlflow
import tyro
from mlflow.lightgbm import autolog as autolog_lightgbm

from .config import EVAL, RUNS, TUNE, UNTUNED, Config, ModelName
from .data import Dataset, build_dataset, load_processed
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


def load(stride: int) -> Dataset:
    print(f"[data] loading the data, a window every {stride} minute(s)")
    return build_dataset(load_processed(), stride)


def tune_cmd(cfg: Cfg) -> None:
    """Optuna search for each model (persistence, ridge and sj_mlp have nothing to tune). Re-running
    resumes it."""
    models: list[ModelName] = [m for m in cfg.models if m not in UNTUNED]
    print(f"[tune] experiment '{cfg.name}': {' '.join(models)}, {cfg.n_trials} trials each")
    start_tracking(cfg)
    ds = load(TUNE.stride)  # built once, shared by every model
    for name in models:
        tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    """Train each model with every seed using its best hyper-parameters; score every split."""
    print(f"[eval] experiment '{cfg.name}': {' '.join(cfg.models)}")
    start_tracking(cfg)
    ds = load(EVAL.stride)
    for name in cfg.models:
        evaluate(cfg, ds, name)


def report_cmd(cfg: Cfg) -> None:
    """Results table and figure, read from MLflow (the newest finished evaluations)."""
    start_tracking(cfg)
    make_report(cfg)


def ui_cmd(port: int = 5000) -> None:
    """MLflow in the browser, over every experiment; Ctrl-C closes it. Training does not need it:
    results are recorded either way."""
    RUNS.mkdir(exist_ok=True)
    print(
        f"[ui] MLflow at http://127.0.0.1:{port} (the first start can take ~30 s); Ctrl-C closes it"
    )
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        MLFLOW_URI,
        "--port",
        str(port),
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("[ui] closed")
    except subprocess.CalledProcessError:
        sys.exit(
            f"[ui] MLflow did not start; if port {port} is taken: uv run adseq ui --port {port + 1}"
        )


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # progress shows at once in Git Bash too
    tyro.extras.subcommand_cli_from_dict(
        {"tune": tune_cmd, "evaluate": evaluate_cmd, "report": report_cmd, "ui": ui_cmd}
    )
