import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from .config import EVAL, UNTUNED, Config, ModelName
from .data import Dataset
from .models import fit, predict, refit
from .tune import n_done, open_study

SPLITS = ("train", "val", "test")


def best_params(cfg: Config, ds: Dataset, model: ModelName) -> dict:
    seeds = " ".join(map(str, cfg.seeds))
    if model in UNTUNED:
        print(f"[eval] {model}: nothing to tune; seeds {seeds}")
        return {}
    study = open_study(cfg, model)
    if (done := n_done(study)) < cfg.n_trials:
        raise LookupError(
            f"{model}: {done}/{cfg.n_trials} trials tuned in {cfg.name}: run "
            f"`adseq tune --name {cfg.name} --models {model} --n-trials {cfg.n_trials}` first"
        )
    best = study.best_trial
    print(
        f"[eval] {model}: best of {done} trials is trial {best.number} "
        f"(val RMSE {study.best_value * ds.y_scale:.2f} ft^3/min); seeds {seeds}"
    )
    return best.params


def log_summary(split: str, per_seed: list[dict[str, float]]) -> None:
    df = pd.DataFrame(per_seed)
    mean = {f"{split}_{m}_mean": float(v) for m, v in df.mean().items()}
    std = {f"{split}_{m}_std": float(v) for m, v in df.std().items()}
    mlflow.log_metrics(mean | std)


def log_preds(ds: Dataset, fits: np.ndarray, preds: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d, "preds.npz")
        np.savez_compressed(
            path,
            dates=ds.dates.strftime("%Y-%m-%d %H:%M").to_numpy(str),
            split=ds.split,
            y_true=ds.y_true,
            fits=fits,
            preds=preds,
        )
        mlflow.log_artifact(str(path))


def regression_metrics(y: np.ndarray, p: np.ndarray, k: int = 1) -> dict[str, float]:
    n = len(y)
    rmse = float(root_mean_squared_error(y, p))
    mape = float(mean_absolute_percentage_error(y, p))
    r2 = float(r2_score(y, p, force_finite=False))
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, p)),
        "mape": 100 * mape,
        "r2": r2,
        "adj_r2": adj_r2,
        "teei": mape**2 * rmse / adj_r2 if adj_r2 > 0 else float("nan"),
    }


def evaluate(cfg: Config, ds: Dataset, name: ModelName) -> None:
    hp = best_params(cfg, ds, name)
    n_inputs = 1 if name == "persistence" else ds.n_features
    train = cfg.training(EVAL)
    va, te = ds.part("val"), ds.part("test")

    def metrics(p: np.ndarray, rows: np.ndarray) -> dict[str, float]:
        return regression_metrics(ds.y_true[rows], p[rows], n_inputs)

    fits, preds, vals, tests = [], [], [], []
    with mlflow.start_run(run_name=f"eval/{name}"):
        for seed in cfg.seeds:
            with mlflow.start_run(run_name=f"seed-{seed}", nested=True, tags={"seed": str(seed)}):
                fitted, info = fit(name, hp, ds, train, seed)
                fit_pred = predict(fitted, name, hp, ds)
                with mlflow.start_run(run_name="refit", nested=True):
                    final, refit_info = refit(name, hp, ds, train, seed, info)
                    mlflow.log_metrics(refit_info)
                pred = predict(final, name, hp, ds)
                scores = {
                    "train": metrics(fit_pred, ds.part("train")),
                    "val": metrics(fit_pred, va),
                    "test": metrics(pred, te),
                }
                mlflow.log_params({"seed": seed, **hp})
                mlflow.log_metrics(info)
                mlflow.log_metrics({f"{s}_{k}": v for s, m in scores.items() for k, v in m.items()})
            fits.append(fit_pred)
            preds.append(pred)
            vals.append(v := scores["val"])
            tests.append(t := scores["test"])
            print(
                f"[eval] {name} seed {seed}: val RMSE {v['rmse']:.2f}; "
                f"test RMSE {t['rmse']:.2f}, R2 {t['r2']:.3f}"
            )
        log_summary("val", vals)
        log_summary("test", tests)
        mlflow.log_metric("n_seeds", float(len(cfg.seeds)))

        def mean_std(scores: list[dict[str, float]]) -> str:
            rmse = pd.Series([s["rmse"] for s in scores])
            return f"{rmse.mean():.2f}" + (f" ± {rmse.std():.2f}" if len(rmse) > 1 else "")

        print(
            f"[eval] {name}: done, val RMSE {mean_std(vals)}, test RMSE {mean_std(tests)} ft^3/min"
        )
        log_preds(ds, np.array(fits), np.array(preds))
