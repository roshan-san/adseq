"""Train a model with several seeds and score it on every split (Schroer & Just's MLP is
imported instead: schroer.py).

Each call is a new eval/<model> MLflow run with one seed-<s> child per seed (learning curve,
train/val/test metrics). The parent gets the test metrics' mean and std over seeds and every
seed's predictions (preds.npz); the report uses the newest FINISHED one per model, so a re-run
replaces the reported numbers and a crashed one never does.
"""

import mlflow
import numpy as np

from .config import Config, ModelName
from .data import Dataset
from .metrics import regression_metrics
from .models import MODELS, fit, predict
from .schroer import import_sj_mlp
from .tracking import best_params, child_run, eval_run, log_info, log_preds, log_test_summary

SPLITS = ("train", "val", "test")


def evaluate(cfg: Config, ds: Dataset, name: ModelName) -> None:
    """Fit the model with its best hyper-parameters once per seed (once for the deterministic
    baselines); log its metrics and predictions."""
    if MODELS[name].family == "external":
        return import_sj_mlp(ds)
    hp = best_params(cfg, name)
    seeds = cfg.seeds if MODELS[name].seeded else cfg.seeds[:1]
    n_inputs = 1 if name == "persistence" else ds.n_features  # for the adjusted R2

    preds, tests = [], []
    with eval_run(name):
        for seed in seeds:
            with child_run(f"seed-{seed}", seed=str(seed)):
                fitted, info = fit(name, hp, ds, cfg.eval, seed)
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
