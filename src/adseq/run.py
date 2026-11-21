"""The two jobs, run one after the other for each model: tune, then evaluate.

Tuning: an equal-budget hyper-parameter search with Optuna (TPE sampler, fixed seed). Every tuned
model gets the same number of finished trials (--n-trials, completed or pruned), each trained
with seed 0 and scored on the same validation samples. Each model's study is saved in
runs/<name>/optuna/<model>.log, so re-running continues it: finished trials are kept, and a trial
cut off by a crash stays "running" and does not count. Run one process per model at a time.

Pruning (MedianPruner, recurrent models only): after the first PRUNE_STARTUP finished trials, a
trial stops early when its best val RMSE so far is worse than the median of earlier trials at the
same epoch, checked every epoch once it has made PRUNE_WARMUP_PASSES passes over the training data
(that many epochs x TUNE.stride, so the warmup is the same amount of training at any stride).
Its curve up to that point is kept; it never becomes the best trial. LightGBM trials are not
pruned: early stopping already ends a poor one. The objective `val_rmse` is in scaled units
(target min-max scaled on the training period); the `[tune]` line also prints it in ft^3/min.

Evaluation: each model is trained with every seed using its best hyper-parameters (once for the
deterministic baselines) and scored on every split.

MLflow is the only store of results; one experiment per --name, artifacts in MLflow's default
./mlruns/:

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


# --- metrics ------------------------------------------------------------------------------------


def regression_metrics(y: np.ndarray, p: np.ndarray, k: int = 1) -> dict[str, float]:
    """scikit-learn's RMSE, MAE, MAPE and R2, plus the adjusted R2 (k = number of input features
    per sample) and Schroer & Just's TeEI, which no library has."""
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
# passes over the training data before a trial may be pruned: slow starters (low learning rate)
# need it; the tuned gru_res is still 5.8 % above its final val RMSE after 2 passes, 1.9 % after 3
PRUNE_WARMUP_PASSES = 3
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)  # what counts towards --n-trials


def open_study(cfg: Config, name: ModelName) -> optuna.Study:
    """The model's study, new or resumed from its journal file."""
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
    # seed 0 for a new study, so a run without interruptions can be replayed exactly; after a
    # resume the seed moves on, else the sampler's random first trials would repeat earlier ones
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    return study


def run_trial(trial: Trial, name: ModelName, ds: Dataset, cfg: Config) -> float:
    """Train one suggested configuration; its best val RMSE (or TrialPruned)."""
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
    """Run the model's study up to --n-trials finished trials; log and return the best
    hyper-parameters (best_params.json on tune/<model>, typed, overwritten as tuning goes on)."""
    study = open_study(cfg, name)
    remaining = cfg.n_trials - len(study.get_trials(deepcopy=False, states=FINISHED))
    with tune_run(name):
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
    """Fit the model with its best hyper-parameters once per seed (once for the deterministic
    baselines); log its metrics and predictions."""
    hp = best_params(cfg, name)
    seeds = cfg.seeds[:1] if name in UNTUNED else cfg.seeds
    n_inputs = 1 if name == "persistence" else ds.n_features  # for the adjusted R2

    preds, tests = [], []
    with eval_run(name):
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
