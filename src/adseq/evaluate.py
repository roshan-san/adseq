"""Train every model with several seeds and score it on the test period.

Each (model, seed) is a Weights & Biases run in project "adseq", grouped by --name:
hyper-parameters, the learning curve and the train/val/test metrics. Online by default;
--wandb-mode offline keeps them in runs/<name>/wandb/ for a later `wandb sync`.

Outputs, one file per model (so models can be evaluated in parallel terminals):
  runs/<name>/metrics/<model>.csv   one row per (seed, split)
  runs/<name>/preds/<model>.npz     every seed's predictions, for the report's DM tests
"""

import numpy as np
import pandas as pd
import wandb

from .config import WANDB_PROJECT, Config
from .data import Dataset, build_dataset
from .metrics import regression_metrics
from .models import UNTRAINED_MODELS
from .train import log_curve, train_and_predict
from .tune import load_hp

SPLITS = ("train", "val", "test")


def evaluate(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    for d in ("preds", "metrics"):
        (cfg.run_dir / d).mkdir(exist_ok=True)
    ds = build_dataset(df, cfg)
    return pd.concat([evaluate_model(cfg, ds, name) for name in cfg.models], ignore_index=True)


def evaluate_model(cfg: Config, ds: Dataset, name: str) -> pd.DataFrame:
    hp = load_hp(cfg, name)
    seeds = (0,) if name in UNTRAINED_MODELS else cfg.seeds  # deterministic: one run
    n_inputs = 1 if name == "persistence" else ds.n_features  # for the adjusted R2
    preds, rows = [], []
    for seed in seeds:
        with wandb.init(
            project=WANDB_PROJECT,
            group=cfg.name,
            job_type="evaluate",
            name=f"{name}-s{seed}",
            config={"model": name, "seed": seed, **hp},
            mode=cfg.wandb_mode,
            dir=cfg.run_dir,
        ) as run:
            pred, info, curve = train_and_predict(name, hp, ds, cfg, seed)
            log_curve(run, curve)
            scores = {
                split: regression_metrics(ds.y_true[ds.part(split)], pred[ds.part(split)], n_inputs)
                for split in SPLITS
            }
            flat = {
                f"{split}_{metric}": v for split, m in scores.items() for metric, v in m.items()
            }
            run.summary.update(flat | info)
        preds.append(pred)
        rows += [{"model": name, "seed": seed, "split": s, **m, **info} for s, m in scores.items()]
        test = scores["test"]
        print(f"[eval] {name:11s} seed={seed}  test RMSE={test['rmse']:.4g}  R2={test['r2']:.3f}")

    np.savez(
        cfg.run_dir / "preds" / f"{name}.npz",
        dates=ds.dates.strftime("%Y-%m-%d %H:%M").to_numpy(str),
        split=ds.split,
        y_true=ds.y_true,
        preds=np.array(preds),  # (seeds, samples)
    )
    out = pd.DataFrame(rows)
    out.to_csv(cfg.run_dir / "metrics" / f"{name}.csv", index=False)
    return out
