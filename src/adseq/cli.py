import io
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

Cfg = Annotated[Config, tyro.conf.arg(name="")]
MLFLOW_URI = "sqlite:///runs/mlflow.db"


def start_tracking(cfg: Config) -> None:
    RUNS.mkdir(exist_ok=True)
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg.name)
    autolog_lightgbm(log_models=False, log_datasets=False, silent=True)


def load(stride: int) -> Dataset:
    print(f"[data] loading the data, a window every {stride} minute(s)")
    return build_dataset(load_processed(), stride)


def tune_cmd(cfg: Cfg) -> None:
    models: list[ModelName] = [m for m in cfg.models if m not in UNTUNED]
    print(f"[tune] experiment '{cfg.name}': {' '.join(models)}, {cfg.n_trials} trials each")
    start_tracking(cfg)
    ds = load(TUNE.stride)
    for name in models:
        tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    print(f"[eval] experiment '{cfg.name}': {' '.join(cfg.models)}")
    start_tracking(cfg)
    ds = load(EVAL.stride)
    for name in cfg.models:
        evaluate(cfg, ds, name)


def report_cmd(cfg: Cfg) -> None:
    start_tracking(cfg)
    make_report(cfg)


def ui_cmd(port: int = 5000) -> None:
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
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    tyro.extras.subcommand_cli_from_dict(
        {"tune": tune_cmd, "evaluate": evaluate_cmd, "report": report_cmd, "ui": ui_cmd}
    )
