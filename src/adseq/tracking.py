"""MLflow: the only store of results, and the only module that knows how they are laid out.

One experiment per --name, its artifacts in runs/mlartifacts/<name>/ (not ./mlruns):

  tune/<model>     tags kind=tune, model; best_val_rmse (scaled); artifact best_params.json.
                   One per model, reopened on every `adseq tune`, like its Optuna study.
    trial-<n>      tags trial, pruned; params = hyper-parameters; learning curve; fit info
  eval/<model>     tags kind=eval, model; a new one per `adseq evaluate`

Only parents carry the kind tag, so every lookup filters on it.
"""

import contextlib
import time
from pathlib import Path

import mlflow
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException

from .config import Config
from .models import UNTUNED_MODELS

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


def tune_parent_id(cfg: Config, model: str) -> str | None:
    """The model's tune/<model> run: the oldest one, so a stray duplicate never wins."""
    runs = mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string=f"tags.kind = 'tune' AND tags.model = '{model}'",
        order_by=["attributes.start_time ASC"],
    )
    return None if runs.empty else runs.run_id.iloc[0]


def tune_run(cfg: Config, model: str) -> mlflow.ActiveRun:
    """Reopen the model's tune/<model> run, or start it: one per model, as its Optuna study."""
    run_id = tune_parent_id(cfg, model)
    if run_id is not None:
        return mlflow.start_run(run_id=run_id)
    return mlflow.start_run(run_name=f"tune/{model}", tags={"kind": "tune", "model": model})


def eval_run(model: str) -> mlflow.ActiveRun:
    """A new eval/<model> run. It ends FINISHED only if every seed did, so a crashed
    evaluation is never reported."""
    return mlflow.start_run(run_name=f"eval/{model}", tags={"kind": "eval", "model": model})


def child_run(name: str, **tags: str) -> mlflow.ActiveRun:
    """A trial or seed run under the active parent."""
    return mlflow.start_run(run_name=name, nested=True, tags=tags)


def best_params(cfg: Config, model: str) -> dict:
    """The best trial's hyper-parameters, from tune/<model>'s best_params.json (an artifact, not
    params: params can't change, the best trial does when tuning goes on). {} for the untuned
    baselines, persistence and ridge."""
    if model in UNTUNED_MODELS:
        return {}
    missing = LookupError(
        f"no tuned hyper-parameters for {model} in experiment {cfg.name}: "
        f"run `adseq tune --name {cfg.name}` first"
    )
    run_id = tune_parent_id(cfg, model)
    if run_id is None:
        raise missing
    try:  # a tune run that crashed before its first finish has no best_params.json yet
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/best_params.json")
    except (MlflowException, OSError) as e:
        raise missing from e
