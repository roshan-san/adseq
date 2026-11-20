"""Equal-budget hyper-parameter search with Optuna (TPE sampler, fixed seed).

Every tuned model gets the same number of finished trials (--n-trials), each trained with seed 0
and scored on the same validation samples. Each model's study is saved in
runs/<name>/optuna/<model>.log, so re-running continues it: finished trials are kept, and a trial
cut off by a crash is marked failed and run again. Run one process per model at a time.
Every trial is also an MLflow run (tag job=tune) with its learning curve.

The objective `val_rmse` is in scaled units (target min-max scaled on the training period);
the `[tune]` line also prints it in ft^3/min.
"""

import json
from pathlib import Path

import mlflow
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial, TrialState

from .config import Config
from .data import WINDOWS, Dataset
from .models import TREE_MODELS, UNTRAINED_MODELS
from .train import fit_tuned, log_curve


def suggest(trial: Trial, name: str) -> dict:
    """Search space: the same ranges for every recurrent model; LightGBM's own for the trees
    (same loss and window choices)."""
    if name in TREE_MODELS:
        return {
            "lr": trial.suggest_float("lr", 5e-3, 0.2, log=True),
            "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127]),
            "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [5, 10, 20, 50, 100]),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "loss": trial.suggest_categorical("loss", ["mse", "l1"]),
            "window": trial.suggest_categorical("window", list(WINDOWS)),
        }
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "loss": trial.suggest_categorical("loss", ["mse", "l1"]),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "window": trial.suggest_categorical("window", list(WINDOWS)),
    }


def hp_file(cfg: Config, name: str) -> Path:
    return cfg.run_dir / "best_hp" / f"{name}.json"


def load_hp(cfg: Config, name: str) -> dict:
    """Best trial's hyper-parameters ({} for the untrained baselines)."""
    if name in UNTRAINED_MODELS:
        return {}
    path = hp_file(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `adseq tune --name {cfg.name}` first")
    return json.loads(path.read_text())


def tune(cfg: Config, ds: Dataset, name: str) -> dict:
    study_file = cfg.run_dir / "optuna" / f"{name}.log"
    study_file.parent.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(study_file)))
    study = optuna.create_study(
        study_name=name, storage=storage, direction="minimize", load_if_exists=True
    )
    # seed 0 for a new study, so a run without interruptions can be replayed exactly; after a
    # resume the seed moves on, else the sampler's random first trials would repeat earlier ones
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    # a trial still "running" was cut off by a crash (one process per model): fail it, run anew
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        storage.set_trial_state_values(t._trial_id, TrialState.FAIL)

    def objective(trial: Trial) -> float:
        hp = suggest(trial, name)
        tags = {"job": "tune", "model": name, "trial": str(trial.number)}
        with mlflow.start_run(run_name=f"{name}-t{trial.number}", tags=tags):
            mlflow.log_params(hp)
            info, curve = fit_tuned(name, hp, ds, cfg, seed=0)  # no test predictions
            log_curve(curve)
            mlflow.log_metrics({k: info[k] for k in ("val_rmse_scaled", "epochs", "n_params")})
            mlflow.set_tag("device", info["device"])
        return info["val_rmse_scaled"]  # best epoch

    done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if cfg.n_trials > done:
        # a trial that raises stops the study with its traceback; re-run to continue
        study.optimize(objective, n_trials=cfg.n_trials - done)

    best = study.best_params
    path = hp_file(cfg, name)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(best, indent=2))
    val = study.best_value
    print(f"[tune] {name}: best val RMSE = {val * ds.y_scale:.2f} (scaled {val:.4f})  {best}")
    return best
