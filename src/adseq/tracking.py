"""MLflow: the only store of results, and the only module that knows how they are laid out.

One experiment per --name, its artifacts in runs/mlartifacts/<name>/ (not ./mlruns).
"""

import contextlib
import time
from pathlib import Path

import mlflow
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException

from .config import Config

BATCH_LIMIT = 1000  # MLflow's maximum number of metrics per log_batch call


def start_tracking(cfg: Config) -> None:
    """Log to cfg.mlflow_uri, in the experiment named --name (created with its artifact folder
    on first use)."""
    Path("runs").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    if mlflow.get_experiment_by_name(cfg.name) is None:
        artifacts = Path("runs/mlartifacts", cfg.name).resolve().as_uri()
        # another process (scripts/run_parallel.sh) may create it first
        with contextlib.suppress(MlflowException):
            mlflow.create_experiment(cfg.name, artifact_location=artifacts)
    mlflow.set_experiment(cfg.name)


def log_info(info: dict) -> None:
    """Training info to the active run: numbers as metrics, strings (the device) as tags."""
    mlflow.log_metrics({k: v for k, v in info.items() if not isinstance(v, str)})
    mlflow.set_tags({k: v for k, v in info.items() if isinstance(v, str)})


def log_lgbm_curve(train_loss: list[float], val_rmse: list[float]) -> None:
    """LightGBM's learning curve to the active run, one step per boosting round, logged once
    after training: thousands of rounds, so sent in batches."""
    t = int(time.time() * 1000)
    metrics = []
    for step, (loss, rmse) in enumerate(zip(train_loss, val_rmse, strict=True), start=1):
        metrics.append(Metric("epoch_train_loss", float(loss), t, step))
        metrics.append(Metric("epoch_val_rmse", float(rmse), t, step))
    run_id = mlflow.active_run().info.run_id
    client = mlflow.MlflowClient()
    for i in range(0, len(metrics), BATCH_LIMIT):
        client.log_batch(run_id, metrics=metrics[i : i + BATCH_LIMIT])
