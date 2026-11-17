"""Equal-budget hyper-parameter search with W&B Sweeps (Bayesian optimisation).

Every tuned model gets the same number of sweep runs (--n-trials), each trained with seed 0 and
scored on the same validation samples. Every trial is a W&B run (job_type "tune"), so the whole
search can be inspected online. The sweep id is saved in runs/<name>/sweeps/, so re-running
continues the same sweep instead of starting a new one. Sweeps need --wandb-mode online.

The sweep metric `val_rmse` is in scaled units (target min-max scaled on the training period);
the `[tune]` line also prints it in ft^3/min.
"""

import json
from pathlib import Path

import wandb

from .config import WANDB_PROJECT, Config
from .data import WINDOWS, Dataset
from .models import UNTRAINED_MODELS
from .train import fit_recurrent, log_curve


def sweep_config(cfg: Config, name: str) -> dict:
    """Search space: the same ranges for every recurrent model."""
    params = {
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-2},
        "batch_size": {"values": [64, 128, 256]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
        "loss": {"values": ["mse", "l1"]},
        "hidden_size": {"values": [16, 32, 64, 128]},
        "num_layers": {"distribution": "int_uniform", "min": 1, "max": 2},
        "window": {"values": list(WINDOWS)},
    }
    return {
        "name": f"{cfg.name}/{name}",
        "method": "bayes",
        "metric": {"name": "val_rmse", "goal": "minimize"},
        "parameters": params,
    }


def hp_file(cfg: Config, name: str) -> Path:
    return cfg.run_dir / "best_hp" / f"{name}.json"


def load_hp(cfg: Config, name: str) -> dict:
    """Best sweep run's hyper-parameters ({} for the untrained baselines)."""
    if name in UNTRAINED_MODELS:
        return {}
    path = hp_file(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `adseq tune --name {cfg.name}` first")
    return json.loads(path.read_text())


def tune(cfg: Config, ds: Dataset, name: str) -> dict:
    if cfg.wandb_mode != "online":
        raise SystemExit("W&B Sweeps need --wandb-mode online (run `uv run wandb login` once).")
    api = wandb.Api()

    # one sweep per model; its id is saved so a re-run continues it
    id_file = cfg.run_dir / "sweeps" / f"{name}.txt"
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
            tags=[name],
            dir=cfg.run_dir,
        ) as run:
            hp = dict(run.config)  # the sweep's parameters are exactly the model's hyper-parameters
            _, info, curve = fit_recurrent(name, hp, ds, cfg, seed=0)  # no test predictions
            log_curve(run, curve)
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
    # wandb.agent swallows trial exceptions, so check that at least one trial finished
    if best_run is None or "val_rmse" not in best_run.summary:
        raise SystemExit(f"[tune] {name}: no trial finished; see the failed runs at {sweep.url}")
    best = dict(best_run.config)
    path = hp_file(cfg, name)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(best, indent=2))
    val = best_run.summary["val_rmse"]
    print(f"[tune] {name}: best val RMSE = {val * ds.y_scale:.2f} (scaled {val:.4f})  {best}")
    print(f"       sweep: {sweep.url}")
    return best
