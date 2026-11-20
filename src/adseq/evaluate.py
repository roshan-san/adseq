"""Train a model with several seeds and score it on every split.

Each seed is an MLflow run (tags job=evaluate, model, seed) with the learning curve and the
train/val/test metrics.

Outputs, one file per model (so models can be evaluated in parallel terminals):
  runs/<name>/metrics/<model>.csv   one row per (seed, split)
  runs/<name>/preds/<model>.npz     every seed's predictions, for the report
"""

import mlflow
import numpy as np
import pandas as pd

from .config import Config
from .data import Dataset
from .fit import fit, predict
from .metrics import regression_metrics
from .models import UNTRAINED_MODELS
from .tracking import log_run
from .tune import load_hp

SPLITS = ("train", "val", "test")


def evaluate(cfg: Config, ds: Dataset, name: str) -> None:
    """Fit the model with its best hyper-parameters once per seed (once for the deterministic
    baselines); save its metrics and predictions."""
    hp = load_hp(cfg, name)
    seeds = cfg.seeds[:1] if name in UNTRAINED_MODELS else cfg.seeds
    n_inputs = 1 if name == "persistence" else ds.n_features  # for the adjusted R2

    preds, rows = [], []
    for seed in seeds:
        tags = {"job": "evaluate", "model": name, "seed": str(seed)}
        with mlflow.start_run(run_name=f"{name}-s{seed}", tags=tags):
            fitted, info, curve = fit(name, hp, ds, cfg, seed)
            pred = predict(fitted, name, hp, ds)
            scores = {
                s: regression_metrics(ds.y_true[ds.part(s)], pred[ds.part(s)], n_inputs)
                for s in SPLITS
            }
            log_run({"seed": seed, **hp}, info, curve, scores)

        preds.append(pred)
        rows += [{"model": name, "seed": seed, "split": s, **m, **info} for s, m in scores.items()]
        test = scores["test"]
        print(f"[eval] {name:11s} seed={seed}  test RMSE={test['rmse']:.4g}  R2={test['r2']:.3f}")

    for d in ("preds", "metrics"):
        (cfg.run_dir / d).mkdir(exist_ok=True)
    np.savez(
        cfg.run_dir / "preds" / f"{name}.npz",
        dates=ds.dates.strftime("%Y-%m-%d %H:%M").to_numpy(str),
        split=ds.split,
        y_true=ds.y_true,
        preds=np.array(preds),  # (seeds, samples)
    )
    pd.DataFrame(rows).to_csv(cfg.run_dir / "metrics" / f"{name}.csv", index=False)
