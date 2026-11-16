"""Paper Sec. IV-C: equal-budget hyper-parameter search with W&B Sweeps (Bayesian optimisation).

Every tuned model gets the same number of sweep runs (--n-trials), each trained with seed 0 and
scored on the same validation days. Every trial is a W&B run (job_type "tune"), so the whole
search can be inspected online. The sweep id is saved in runs/<name>/sweeps/, so re-running
continues the same sweep instead of starting a new one. Sweeps need --wandb-mode online.

The sweep metric `val_rmse` is in *scaled* units: fine for ranking runs within one sweep, but
not comparable across scalings (min-max values look ~6x smaller). To compare runs/main with
runs/minmax use the original-unit value that the `[tune]` line prints.
"""

import json
from pathlib import Path

import wandb

from .config import WANDB_PROJECT, Config
from .data import TARGET, Dataset
from .models import PAPER_HP, RECURRENT_MODELS
from .train import train_and_predict


def sweep_config(cfg: Config, name: str) -> dict:
    """Search space (paper Table: hyper-parameter ranges). Same shared ranges for every model."""
    params = {
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64, 128]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
    }
    if name in RECURRENT_MODELS:
        params["hidden_size"] = {"values": [16, 32, 64, 128]}
        params["num_layers"] = {"distribution": "int_uniform", "min": 1, "max": 2}
    else:  # fnn_tuned, window_fnn
        params["n_layers"] = {"distribution": "int_uniform", "min": 1, "max": 3}
        params["width"] = {"values": [32, 64, 128, 256]}
    return {
        "name": f"{cfg.name}/{TARGET}/{name}",
        "method": "bayes",
        "metric": {"name": "val_rmse", "goal": "minimize"},
        "parameters": params,
    }


def hp_file(cfg: Config, name: str) -> Path:
    return cfg.run_dir / "best_hp" / f"{TARGET}__{name}.json"


def load_hp(cfg: Config, name: str) -> dict:
    """Best sweep run for tuned models; fnn_paper uses the base paper's fixed HP."""
    if name == "persistence":
        return {}
    if name == "fnn_paper":
        return dict(PAPER_HP)
    path = hp_file(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `adseq tune --name {cfg.name}` first")
    return json.loads(path.read_text())


def tune(cfg: Config, ds: Dataset, name: str) -> dict:
    if cfg.wandb_mode != "online":
        raise SystemExit("W&B Sweeps need --wandb-mode online (run `uv run wandb login` once).")
    api = wandb.Api()

    # one sweep per model; its id is saved so a re-run continues it
    id_file = cfg.run_dir / "sweeps" / f"{TARGET}__{name}.txt"
    if id_file.exists():
        sweep_path = id_file.read_text().strip()
    else:
        sweep_id = wandb.sweep(sweep_config(cfg, name), project=WANDB_PROJECT)
        sweep_path = f"{api.default_entity}/{WANDB_PROJECT}/{sweep_id}"
        id_file.parent.mkdir(exist_ok=True)
        id_file.write_text(sweep_path)
    entity, _, sweep_id = sweep_path.split("/")

    done = sum(r.state == "finished" for r in api.sweep(sweep_path).runs)

    def trial() -> None:
        with wandb.init(
            group=f"{cfg.name}-tune",
            job_type="tune",
            tags=[TARGET, name],
            dir=cfg.run_dir,
        ) as run:
            hp = dict(run.config)  # the sweep's parameters are exactly the model's hyper-parameters
            _, info, curve = train_and_predict(name, hp, ds, cfg, seed=0)
            for epoch, train_mse, val_rmse in curve:
                run.log({"epoch_train_mse": train_mse, "epoch_val_rmse": val_rmse}, step=epoch)
            run.summary["val_rmse"] = info["val_rmse_scaled"]  # best epoch, the sweep's metric
            run.summary.update({"epochs": info["epochs"], "n_params": info["n_params"]})

    if cfg.n_trials > done:
        wandb.agent(
            sweep_id,
            function=trial,
            entity=entity,
            project=WANDB_PROJECT,
            count=cfg.n_trials - done,
        )

    sweep = wandb.Api().sweep(sweep_path)  # fresh Api: no cached run list
    best_run = sweep.best_run()
    # the scaler the HP were tuned with travels with them, so evaluate can't mix them up
    best = dict(best_run.config) | {"scaling": cfg.scaling}
    path = hp_file(cfg, name)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(best, indent=2))
    val = best_run.summary["val_rmse"]
    print(
        f"[tune] {TARGET} {name}: best val RMSE = {val * ds.y_scale:.2f} "
        f"(scaled {val:.4f}, {cfg.scaling})  {best}"
    )
    print(f"       sweep: {sweep.url}")
    return best
