"""The two jobs, one after the other for each model: tune (Optuna, the same trial budget for
every model, resumable from runs/<name>/optuna/<model>.log), then evaluate with every seed.

MLflow, one experiment per --name (lookups by run name, FINISHED runs only, newest first):
  tune/<model>   best_params.json, best_val_rmse;    trial-<n> children: hp, curve, fit info
  eval/<model>   test_<metric>_mean/_std, preds.npz; seed-<s> children: hp, curve, every split
"""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.artifacts import load_dict
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial, TrialState
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from .config import EVAL, TUNE, UNTUNED, Config, ModelName
from .data import Dataset
from .models import fit, predict, space

# --- MLflow -------------------------------------------------------------------------------------


def log_info(info: dict) -> None:
    """Numbers as metrics, strings (the device) as tags."""
    mlflow.log_metrics({k: v for k, v in info.items() if not isinstance(v, str)})
    mlflow.set_tags({k: v for k, v in info.items() if isinstance(v, str)})


def finished_runs(cfg: Config, run_name: str) -> pd.DataFrame:
    """Newest first; a crashed run is never used."""
    return mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string=f"tags.mlflow.runName = '{run_name}' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )


def child_run(name: str, **tags: str) -> mlflow.ActiveRun:
    return mlflow.start_run(run_name=name, nested=True, tags=tags)


def best_params(cfg: Config, model: ModelName) -> dict:
    """An artifact, not params: they would come back as strings."""
    if model in UNTUNED:
        return {}
    runs = finished_runs(cfg, f"tune/{model}")
    if runs.empty:
        raise LookupError(
            f"no tuned hyper-parameters for {model} in experiment {cfg.name}: "
            f"run `adseq run --name {cfg.name}` first"
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


# --- tuning -------------------------------------------------------------------------------------

PRUNE_STARTUP = 5  # finished trials before any pruning (Optuna's MedianPruner default)
# passes over the training data before a recurrent trial may be pruned: slow starters need them
# (the tuned gru_res is still 5.8 % above its final val RMSE after 2 passes, 1.9 % after 3)
PRUNE_WARMUP_PASSES = 3
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)  # what counts towards --n-trials


def open_study(cfg: Config, name: ModelName) -> optuna.Study:
    study_file = cfg.run_dir / "optuna" / f"{name}.log"
    study_file.parent.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=JournalStorage(JournalFileBackend(str(study_file))),
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_STARTUP, n_warmup_steps=PRUNE_WARMUP_PASSES * TUNE.stride
        ),
    )
    # seed 0 replays an uninterrupted run exactly; after a resume the seed moves on, else the
    # sampler's random first trials would repeat earlier ones
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    return study


def run_trial(trial: Trial, name: ModelName, ds: Dataset, cfg: Config) -> float:
    """Its best val RMSE (scaled), or TrialPruned."""
    hp = space(trial, name)
    pruned = False

    def on_step(epoch: int, best_so_far: float) -> bool:
        nonlocal pruned
        trial.report(best_so_far, epoch)
        pruned = trial.should_prune()
        return pruned

    with child_run(f"trial-{trial.number}", trial=str(trial.number)):
        _, info = fit(name, hp, ds, TUNE, seed=0, on_step=on_step)
        mlflow.log_params(hp)
        log_info(info)
        mlflow.set_tag("pruned", str(pruned).lower())
    if pruned:
        raise optuna.TrialPruned()
    return info["val_rmse_scaled"]


def tune(cfg: Config, ds: Dataset, name: ModelName) -> dict:
    """Up to --n-trials finished trials (completed or pruned), seed 0 each."""
    study = open_study(cfg, name)
    remaining = cfg.n_trials - len(study.get_trials(deepcopy=False, states=FINISHED))
    with mlflow.start_run(run_name=f"tune/{name}"):
        if remaining > 0:
            # a trial that raises stops the study with its traceback; re-run to continue
            study.optimize(lambda trial: run_trial(trial, name, ds, cfg), n_trials=remaining)
        best, val = study.best_params, study.best_value
        mlflow.log_dict(best, "best_params.json")
        mlflow.log_metric("best_val_rmse", val)  # scaled units, as the objective
    print(f"[tune] {name}: best val RMSE = {val * ds.y_scale:.2f} (scaled {val:.4f})  {best}")
    return best


# --- evaluation ---------------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def evaluate(cfg: Config, ds: Dataset, name: ModelName) -> None:
    """Every seed (one for persistence and ridge: the same result for any seed). The run ends
    FINISHED only if every seed did, so a crashed evaluation is never reported."""
    hp = best_params(cfg, name)
    seeds = cfg.seeds[:1] if name in UNTUNED else cfg.seeds
    n_inputs = 1 if name == "persistence" else ds.n_features  # for the adjusted R2

    preds, tests = [], []
    with mlflow.start_run(run_name=f"eval/{name}"):
        for seed in seeds:
            with child_run(f"seed-{seed}", seed=str(seed)):
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
