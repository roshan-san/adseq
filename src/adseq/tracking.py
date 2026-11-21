"""MLflow: the only store of results, and the only module that knows how they are laid out
(writers and readers in pairs). LightGBM's learning curve comes from mlflow.lightgbm.autolog.

One experiment per --name, artifacts in MLflow's default ./mlruns/:

  tune/<model>     best_val_rmse (scaled); artifact best_params.json. A new one per
                   `adseq tune`: the Optuna study carries the trials across resumes, and
                   evaluate uses the newest FINISHED one.
    trial-<n>      tags trial, pruned; params = hyper-parameters; learning curve; fit info
  eval/<model>     a new one per `adseq evaluate`; the report uses the newest FINISHED one per
                   model. test_<metric>_mean / _std, n_seeds; artifact preds.npz (dates, split,
                   y_true, preds: seeds x samples)
    seed-<s>       params = seed + hyper-parameters; learning curve; fit info;
                   <split>_<metric> for train / val / test

Every lookup is by run name.
"""

import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts, load_dict
from mlflow.lightgbm import autolog as autolog_lightgbm

from .config import RUNS, Config, ModelName
from .data import Dataset
from .models import MODELS


def start_tracking(cfg: Config) -> None:
    """Log to cfg.mlflow_uri, in the experiment named --name; LightGBM logs its own curves."""
    RUNS.mkdir(exist_ok=True)  # the default mlflow_uri's sqlite file lives there
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # no bars in the logs
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.name)
    autolog_lightgbm(log_models=False, log_datasets=False, silent=True)


def log_info(info: dict) -> None:
    """Training info to the active run: numbers as metrics, strings (the device) as tags."""
    mlflow.log_metrics({k: v for k, v in info.items() if not isinstance(v, str)})
    mlflow.set_tags({k: v for k, v in info.items() if isinstance(v, str)})


def finished_runs(cfg: Config, run_name: str) -> pd.DataFrame:
    """The FINISHED runs named run_name, newest first: a crashed run is never used."""
    return mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string=f"tags.mlflow.runName = '{run_name}' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )


def tune_run(model: str) -> mlflow.ActiveRun:
    return mlflow.start_run(run_name=f"tune/{model}")


def eval_run(model: str) -> mlflow.ActiveRun:
    """It ends FINISHED only if every seed did, so a crashed evaluation is never reported."""
    return mlflow.start_run(run_name=f"eval/{model}")


def child_run(name: str, **tags: str) -> mlflow.ActiveRun:
    """A trial or seed run under the active parent."""
    return mlflow.start_run(run_name=name, nested=True, tags=tags)


def best_params(cfg: Config, model: ModelName) -> dict:
    """The best trial's hyper-parameters, from tune/<model>'s best_params.json (an artifact, not
    params: params can't change, the best trial does when tuning goes on). {} for the untuned
    baselines, persistence and ridge."""
    if not MODELS[model].tuned:
        return {}
    runs = finished_runs(cfg, f"tune/{model}")
    if runs.empty:
        raise LookupError(
            f"no tuned hyper-parameters for {model} in experiment {cfg.name}: "
            f"run `adseq tune --name {cfg.name}` first"
        )
    return load_dict(f"runs:/{runs.run_id.iloc[0]}/best_params.json")


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
    """The newest FINISHED eval/<model> run of each model, with a model column: what the report
    shows."""
    runs = mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string="tags.mlflow.runName LIKE 'eval/%' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return runs
    runs["model"] = runs["tags.mlflow.runName"].str.removeprefix("eval/")
    return runs.drop_duplicates("model")
