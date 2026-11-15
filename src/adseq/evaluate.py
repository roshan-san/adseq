"""Paper Sec. IV-D / V: train every model with several seeds and score it on the test period.

Each (model, seed) is a Weights & Biases run in project "adseq", grouped by --name:
hyper-parameters, the learning curve and the train/val/test metrics. Online by default;
--wandb-mode offline keeps them in runs/<name>/wandb/ for a later `wandb sync`.
"""

import os

import numpy as np
import pandas as pd
import wandb

from .config import Config
from .data import TARGET, build_dataset
from .metrics import regression_metrics
from .train import train_and_predict
from .tune import load_hp


def evaluate(cfg: Config, daily: pd.DataFrame) -> pd.DataFrame:
    os.environ.setdefault("WANDB_SILENT", "true")
    rows = []
    (cfg.run_dir / "preds").mkdir(exist_ok=True)
    target = TARGET
    datasets = {s: build_dataset(daily, cfg, s) for s in ("standard", "minmax")}
    for name in cfg.models:
        hp = load_hp(cfg, target, name)
        ds = datasets[hp.get("scaling", "standard")]  # same days and features, other scaler
        seeds = (0,) if name == "persistence" else cfg.seeds  # persistence is deterministic
        preds = []
        for seed in seeds:
            with wandb.init(
                project="adseq",
                group=cfg.name,
                job_type="evaluate",
                name=f"{target}-{name}-s{seed}",
                config={"target": target, "model": name, "seed": seed, **hp},
                mode=cfg.wandb_mode,
                dir=cfg.run_dir,
            ) as run:
                # learning curve in scaled units; epoch_* so it can't clash with final metrics
                def log_curve(epoch, train_mse, val_rmse, run=run):
                    run.log({"epoch_train_mse": train_mse, "epoch_val_rmse": val_rmse}, step=epoch)

                pred, info = train_and_predict(name, hp, ds, cfg, seed, on_epoch=log_curve)
                scores = {
                    split: regression_metrics(ds.y_true[ds.part(split)], pred[ds.part(split)])
                    for split in ("train", "val", "test")
                }
                run.summary.update(
                    {f"{split}_{k}": v for split, m in scores.items() for k, v in m.items()} | info
                )
            preds.append(pred)
            for split, m in scores.items():
                rows.append(
                    {"target": target, "model": name, "seed": seed, "split": split, **m, **info}
                )
            print(
                f"[eval] {target:9s} {name:11s} seed={seed}  "
                f"test RMSE={scores['test']['rmse']:.4g}  R2={scores['test']['r2']:.3f}"
            )

        # keep every prediction so the report can average seeds and run DM tests
        np.savez(
            cfg.run_dir / "preds" / f"{target}__{name}.npz",
            dates=ds.dates.strftime("%Y-%m-%d").to_numpy(str),
            split=ds.split,
            y_true=ds.y_true,
            preds=np.array(preds),
        )

    # merge: replace only the (target, model) pairs just run, keep the rest of metrics.csv
    df = pd.DataFrame(rows)
    path = cfg.run_dir / "metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        rerun = old.set_index(["target", "model"]).index.isin(
            df.set_index(["target", "model"]).index
        )
        df = pd.concat([old[~rerun], df], ignore_index=True)
    df.to_csv(path, index=False)
    return df
