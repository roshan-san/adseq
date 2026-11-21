"""Train a model with several seeds and score it on every split; import Schroer & Just's MLP.

Each call is a new eval/<model> MLflow run with one seed-<s> child per seed (learning curve,
train/val/test metrics). The parent gets the test metrics' mean and std over seeds and every
seed's predictions (preds.npz); the report uses the newest FINISHED one per model, so a re-run
replaces the reported numbers and a crashed one never does.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from .config import Config
from .data import Dataset
from .fit import fit, predict
from .metrics import regression_metrics
from .models import UNTUNED_MODELS
from .tracking import best_params, child_run, eval_run, log_info, log_preds, log_test_summary

SPLITS = ("train", "val", "test")
SJ_PREDS = Path("runs/schroer_repro/schroer_mlp_test.csv")  # from scripts/run_schroer.py
SJ_INPUTS = 36  # inputs of their MLP, for its adjusted R2


def evaluate(cfg: Config, ds: Dataset, name: str) -> None:
    """Fit the model with its best hyper-parameters once per seed (once for the deterministic
    baselines); log its metrics and predictions."""
    if name == "sj_mlp":
        return import_sj_mlp(ds)
    hp = best_params(cfg, name)
    seeds = cfg.seeds[:1] if name in UNTUNED_MODELS else cfg.seeds
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


def import_sj_mlp(ds: Dataset) -> None:
    """Schroer & Just's MLP test predictions (their notebook's, via scripts/run_schroer.py) as an
    eval/sj_mlp run laid out like ours: scored on our test samples, NaN elsewhere (their MLP is
    only scored on test). Checks first that their test minutes and labels are ours."""
    if not SJ_PREDS.exists():
        print(f"[eval] sj_mlp: {SJ_PREDS} missing, skipped (run scripts/run_schroer.py)")
        return
    sj = pd.read_csv(SJ_PREDS, index_col=0, parse_dates=True)
    te = ds.split == "test"
    if not ds.dates[te].equals(sj.index) or not np.allclose(sj["y_true"], ds.y_true[te]):
        raise ValueError("their MLP was scored on different test minutes or labels")
    pred = np.full(len(te), np.nan)
    pred[te] = sj["pred"].to_numpy()
    t = regression_metrics(ds.y_true[te], pred[te], SJ_INPUTS)
    with eval_run("sj_mlp"):
        log_test_summary([t])
        log_preds(ds, pred[None])
    print(f"[eval] sj_mlp      (their code)  test RMSE={t['rmse']:.4g}  R2={t['r2']:.3f}")
