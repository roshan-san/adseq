"""MLflow: the only store of results, and the only module that knows how they are laid out.

One experiment per --name, its artifacts in runs/mlartifacts/<name>/ (not ./mlruns):

  tune/<model>     tags kind=tune, model; best_val_rmse (scaled); artifact best_params.json.
                   One per model, reopened on every `adseq tune`, like its Optuna study.
    trial-<n>      tags trial, pruned; params = hyper-parameters; learning curve; fit info
  eval/<model>     tags kind=eval, model; a new one per `adseq evaluate`; the report uses the
                   newest FINISHED one per model. test_<metric>_mean / _std, n_seeds;
                   artifact preds.npz (dates, split, y_true, preds: seeds x samples)
    seed-<s>       params = seed + hyper-parameters; learning curve; fit info;
                   <split>_<metric> for train / val / test

Only parents carry the kind tag, so every lookup filters on it.
"""

import os
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts, load_dict
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException

from .config import Config
from .data import Dataset
from .models import UNTUNED_MODELS

BATCH_LIMIT = 1000  # MLflow's maximum number of metrics per log_batch call


def start_tracking(cfg: Config) -> None:
    """Log to cfg.mlflow_uri, in the experiment named --name (created with its artifact folder
    on first use)."""
    Path("runs").mkdir(exist_ok=True)
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # no bars in the logs
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    try:
        artifacts = Path("runs/mlartifacts", cfg.name).resolve().as_uri()
        mlflow.create_experiment(cfg.name, artifact_location=artifacts)
    except MlflowException as e:
        # it exists (maybe another process of scripts/run_parallel.sh made it first); any other
        # error stops here, else set_experiment would create it with artifacts in ./mlruns
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
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
    run = mlflow.active_run()
    assert run is not None  # fit is always called inside a trial or seed run
    for i in range(0, len(metrics), BATCH_LIMIT):
        mlflow.MlflowClient().log_batch(run.info.run_id, metrics=metrics[i : i + BATCH_LIMIT])


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
    run_id = tune_parent_id(cfg, model)  # None: start it
    tags = {"kind": "tune", "model": model}
    return mlflow.start_run(run_id, run_name=f"tune/{model}", tags=tags)


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
    run_id = tune_parent_id(cfg, model)
    try:  # fails for no tune run (runs:/None/...) or one that crashed before its first finish
        return load_dict(f"runs:/{run_id}/best_params.json")
    except MlflowException as e:
        raise LookupError(
            f"no tuned hyper-parameters for {model} in experiment {cfg.name}: "
            f"run `adseq tune --name {cfg.name}` first"
        ) from e


def log_test_summary(tests: list[dict[str, float]]) -> None:
    """Mean and std over seeds of each test metric to the active (eval) run, as
    test_<metric>_mean / _std, and n_seeds. Sample std (NaN for one seed)."""
    df = pd.DataFrame(tests)
    mean = {f"test_{m}_mean": float(v) for m, v in df.mean().items()}
    std = {f"test_{m}_std": float(v) for m, v in df.std().items()}
    mlflow.log_metrics(mean | std | {"n_seeds": float(len(df))})


def log_preds(ds: Dataset, preds: np.ndarray) -> None:
    """Every seed's predictions (seeds x samples, ft^3/min) to the active (eval) run as
    preds.npz, with the sample dates, splits and labels, so the figure needs nothing else."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d, "preds.npz")
        np.savez_compressed(
            path,
            dates=ds.dates.strftime("%Y-%m-%d %H:%M").to_numpy(str),
            split=ds.split,
            y_true=ds.y_true,
            preds=preds,
        )
        mlflow.log_artifact(str(path))


def load_preds(run_id: str) -> dict:
    """An eval run's preds.npz: dates, split, y_true, preds (seeds x samples)."""
    with tempfile.TemporaryDirectory() as d:
        path = download_artifacts(run_id=run_id, artifact_path="preds.npz", dst_path=d)
        with np.load(path) as f:
            return dict(f)


def latest_evals(cfg: Config) -> pd.DataFrame:
    """The newest FINISHED eval/<model> run of each model: what the report shows."""
    runs = mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string="tags.kind = 'eval' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    return runs.drop_duplicates("tags.model") if not runs.empty else runs
