"""Train each model with every seed using its best hyper-parameters, and score every split.
MLflow: an eval/<model> run (test_<metric>_mean/_std over seeds, preds.npz) with a seed-<s> child
per seed (hyper-parameters, learning curve, every split's metrics). Lookups go by run name,
FINISHED runs only, newest first.
"""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import load_dict
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from .config import EVAL, ONE_SEED, UNTUNED, Config, ModelName
from .data import Dataset
from .models import fit, log_info, predict

SPLITS = ("train", "val", "test")

# --- MLflow -------------------------------------------------------------------------------------


def finished_runs(cfg: Config, run_name: str) -> pd.DataFrame:
    """Newest first; a crashed run is never used."""
    return mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string=f"tags.mlflow.runName = '{run_name}' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )


def best_params(cfg: Config, model: ModelName) -> dict:
    """An artifact, not params: they would come back as strings."""
    if model in UNTUNED:
        return {}
    runs = finished_runs(cfg, f"tune/{model}")
    if runs.empty:
        raise LookupError(
            f"no tuned hyper-parameters for {model} in experiment {cfg.name}: "
            f"run `adseq tune --name {cfg.name}` first"
        )
    return load_dict(f"runs:/{runs.run_id.iloc[0]}/best_params.json")


def log_test_summary(tests: list[dict[str, float]]) -> None:
    """Mean and sample std over seeds (NaN for one seed)."""
    df = pd.DataFrame(tests)
    mean = {f"test_{m}_mean": float(v) for m, v in df.mean().items()}
    std = {f"test_{m}_std": float(v) for m, v in df.std().items()}
    mlflow.log_metrics(mean | std | {"n_seeds": float(len(df))})


def log_preds(ds: Dataset, preds: np.ndarray) -> None:
    """Every seed's predictions with the dates, splits and labels: the report needs nothing else."""
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


# --- metrics ------------------------------------------------------------------------------------


def regression_metrics(y: np.ndarray, p: np.ndarray, k: int = 1) -> dict[str, float]:
    """Schroer & Just's metrics; k = input features, for the adjusted R2."""
    n = len(y)
    rmse = float(root_mean_squared_error(y, p))
    mape = float(mean_absolute_percentage_error(y, p))  # a fraction
    r2 = float(r2_score(y, p, force_finite=False))  # NaN, not a made-up 0/1, for a constant y
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, p)),
        "mape": 100 * mape,  # %
        "r2": r2,
        "adj_r2": adj_r2,
        # Schroer & Just's testing error index: MAPE^2 x RMSE / adj R2 (MAPE as a fraction)
        "teei": mape**2 * rmse / adj_r2 if adj_r2 > 0 else float("nan"),
    }


# --- evaluation ---------------------------------------------------------------------------------


def evaluate(cfg: Config, ds: Dataset, name: ModelName) -> None:
    """Every seed (one for ONE_SEED models: the same result for any seed). The run ends
    FINISHED only if every seed did, so a crashed evaluation is never reported."""
    hp = best_params(cfg, name)
    seeds = cfg.seeds[:1] if name in ONE_SEED else cfg.seeds
    n_inputs = 1 if name == "persistence" else ds.n_features  # for the adjusted R2

    preds, tests = [], []
    with mlflow.start_run(run_name=f"eval/{name}"):
        for seed in seeds:
            with mlflow.start_run(run_name=f"seed-{seed}", nested=True, tags={"seed": str(seed)}):
                fitted, info = fit(name, hp, ds, EVAL, seed)
                pred = predict(fitted, name, hp, ds)
                scores = {
                    s: regression_metrics(ds.y_true[ds.part(s)], pred[ds.part(s)], n_inputs)
                    for s in SPLITS
                }
                mlflow.log_params({"seed": seed, **hp})
                log_info(info)
                mlflow.log_metrics({f"{s}_{k}": v for s, m in scores.items() for k, v in m.items()})
            preds.append(pred)
            tests.append(t := scores["test"])
            print(f"[eval] {name:11s} seed={seed}  test RMSE={t['rmse']:.4g}  R2={t['r2']:.3f}")
        log_test_summary(tests)
        log_preds(ds, np.array(preds))
