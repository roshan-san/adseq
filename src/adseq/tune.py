"""Paper Sec. IV-C: equal-budget hyper-parameter search with W&B Sweeps (Bayesian optimisation).

Every tuned model gets the same number of sweep runs (--n-trials), each trained with seed 0 and
scored on the same validation days. Every trial is a W&B run (job_type "tune"), so the whole
search can be inspected online. The sweep id is saved in runs/<name>/sweeps/, so re-running
continues the same sweep instead of starting a new one. Sweeps need --wandb-mode online.
"""

import json
import os
from pathlib import Path

import wandb

from .config import Config
from .data import Dataset
from .models import PAPER_HP
from .train import train_and_predict

PROJECT = "adseq"
TUNED_MODELS = ("fnn_tuned", "window_fnn", "rnn", "gru", "lstm")
FNN_MODELS = ("fnn_tuned", "window_fnn")


def sweep_config(cfg: Config, target: str, name: str) -> dict:
    """Search space (paper Table: hyper-parameter ranges). Same shared ranges for every model."""
    params = {
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64, 128]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
    }
    if name in FNN_MODELS:
        params["n_layers"] = {"distribution": "int_uniform", "min": 1, "max": 3}
        params["width"] = {"values": [32, 64, 128, 256]}
    else:
        params["hidden_size"] = {"values": [16, 32, 64, 128]}
        params["num_layers"] = {"distribution": "int_uniform", "min": 1, "max": 2}
    return {
        "name": f"{cfg.name}/{target}/{name}",
        "method": "bayes",
        "metric": {"name": "val_rmse", "goal": "minimize"},
        "parameters": params,
    }


def to_hp(c: dict, name: str) -> dict:
    """Sweep config -> hyper-parameters that make_model / train_and_predict understand."""
    hp = {k: c[k] for k in ("lr", "weight_decay", "batch_size", "dropout")}
    if name in FNN_MODELS:
        hp["hidden"] = [
            c["width"] // 2**i for i in range(c["n_layers"])
        ]  # e.g. 128-64, like the paper
    else:
        hp["hidden_size"], hp["num_layers"] = c["hidden_size"], c["num_layers"]
    return hp


def hp_file(cfg: Config, target: str, name: str) -> Path:
    return cfg.run_dir / "best_hp" / f"{target}__{name}.json"


def load_hp(cfg: Config, target: str, name: str) -> dict:
    """Best sweep run for tuned models; fnn_paper uses the base paper's fixed HP."""
    if name == "persistence":
        return {}
    if name == "fnn_paper":
        return dict(PAPER_HP)
    path = hp_file(cfg, target, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `adseq tune --name {cfg.name}` first")
    return json.loads(path.read_text())


def tune(cfg: Config, ds: Dataset, name: str) -> dict:
    if cfg.wandb_mode != "online":
        raise SystemExit("W&B Sweeps need --wandb-mode online (run `uv run wandb login` once).")
    os.environ.setdefault("WANDB_SILENT", "true")
    target = ds.target
    api = wandb.Api()

    # one sweep per (target, model); its id is saved so a re-run continues it
    id_file = cfg.run_dir / "sweeps" / f"{target}__{name}.txt"
    if id_file.exists():
        sweep_path = id_file.read_text().strip()
    else:
        sweep_id = wandb.sweep(sweep_config(cfg, target, name), project=PROJECT)
        sweep_path = f"{api.default_entity}/{PROJECT}/{sweep_id}"
        id_file.parent.mkdir(exist_ok=True)
        id_file.write_text(sweep_path)

    done = sum(r.state == "finished" for r in api.sweep(sweep_path).runs)
    number = iter(range(done + 1, 10_000))  # readable run names: Biogas-gru-tune-07

    def trial() -> None:
        with wandb.init(
            name=f"{target}-{name}-tune-{next(number):02d}",
            group=f"{cfg.name}-tune",
            job_type="tune",
            tags=[target, name],
            dir=cfg.run_dir,
        ) as run:
            hp = to_hp(dict(run.config), name)

            def log_curve(epoch, train_mse, val_rmse):
                run.log({"epoch_train_mse": train_mse, "epoch_val_rmse": val_rmse}, step=epoch)

            _, info = train_and_predict(name, hp, ds, cfg, seed=0, on_epoch=log_curve)
            run.summary["val_rmse"] = info["val_rmse_scaled"]  # best epoch, the sweep's metric
            run.summary.update({"epochs": info["epochs"], "n_params": info["n_params"]})

    if cfg.n_trials > done:
        wandb.agent(
            sweep_path.split("/")[-1],
            function=trial,
            entity=sweep_path.split("/")[0],
            project=PROJECT,
            count=cfg.n_trials - done,
        )

    best_run = wandb.Api().sweep(sweep_path).best_run()  # fresh Api: no cached run list
    best = to_hp(best_run.config, name)
    path = hp_file(cfg, target, name)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(best, indent=2))
    print(
        f"[tune] {target} {name}: best val RMSE (scaled) = {best_run.summary['val_rmse']:.4f}  {best}"
    )
    print(
        f"       sweep: https://wandb.ai/{sweep_path.replace(f'/{PROJECT}/', f'/{PROJECT}/sweeps/')}"
    )
    return best
