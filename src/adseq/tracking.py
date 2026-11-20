"""MLflow logging, shared by tune and evaluate.

One experiment per --name. Every tuning trial and every evaluation seed is one run, tagged
job ("tune" / "evaluate"), model, and trial or seed: group or filter by these tags in the
MLflow UI. A run holds the hyper-parameters, the learning curve (epoch_* metrics) and the
final metrics. The files under runs/<name>/ remain the source of truth for the report.
"""

import time
from pathlib import Path

import mlflow
from mlflow.entities import Metric

from .config import Config
from .fit import Curve

BATCH_LIMIT = 1000  # MLflow's maximum number of metrics per log_batch call


def start_tracking(cfg: Config) -> None:
    """Log to cfg.mlflow_uri, in the experiment named --name."""
    Path("runs").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.name)


def log_run(params: dict, info: dict, curve: Curve, scores: dict | None = None) -> None:
    """One trial / seed to the active run: params, learning curve, training info (numbers as
    metrics, strings such as the device as tags) and, if given, {split: metrics} as
    <split>_<metric>."""
    mlflow.log_params(params)
    log_curve(curve)
    numbers = {k: v for k, v in info.items() if not isinstance(v, str)}
    strings = {k: v for k, v in info.items() if isinstance(v, str)}
    split_metrics = {f"{s}_{k}": v for s, m in (scores or {}).items() for k, v in m.items()}
    mlflow.log_metrics(numbers | split_metrics)
    mlflow.set_tags(strings)


def log_curve(curve: Curve) -> None:
    """Learning curve to the active run, as epoch_train_loss / epoch_val_rmse per step (epoch_*
    so it can't clash with the final metrics). Sent in batches: LightGBM curves can be
    thousands of rounds long."""
    t = int(time.time() * 1000)
    metrics = []
    for epoch, train_loss, val_rmse in curve:
        metrics.append(Metric("epoch_train_loss", float(train_loss), t, epoch))
        metrics.append(Metric("epoch_val_rmse", float(val_rmse), t, epoch))
    run_id = mlflow.active_run().info.run_id
    client = mlflow.MlflowClient()
    for i in range(0, len(metrics), BATCH_LIMIT):
        client.log_batch(run_id, metrics=metrics[i : i + BATCH_LIMIT])
