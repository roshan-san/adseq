"""Optuna search per model, the same trial budget for every model, resumable from
runs/<name>/optuna/<model>.log. MLflow: a tune/<model> run (best_params.json, best_val_rmse)
with a trial-<n> child per trial (hyper-parameters, learning curve, fit info, tag pruned).
"""

import mlflow
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial, TrialState

from .config import TUNE, Config, ModelName
from .data import Dataset
from .models import fit, log_info, space

PRUNE_STARTUP = 5  # finished trials before any pruning (Optuna's MedianPruner default)
# passes over the training data before a recurrent trial may be pruned: slow starters need them
# (the tuned gru_res is still 5.8 % above its final val RMSE after 2 passes, 1.9 % after 3)
PRUNE_WARMUP_PASSES = 3
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)  # what counts towards --n-trials


def open_study(cfg: Config, name: ModelName) -> optuna.Study:
    study_file = cfg.run_dir / "optuna" / f"{name}.log"
    study_file.parent.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=JournalStorage(JournalFileBackend(str(study_file))),
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_STARTUP, n_warmup_steps=PRUNE_WARMUP_PASSES * TUNE.stride
        ),
    )
    # seed 0 replays an uninterrupted run exactly; after a resume the seed moves on, else the
    # sampler's random first trials would repeat earlier ones
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    return study


def run_trial(trial: Trial, name: ModelName, ds: Dataset, cfg: Config) -> float:
    """Its best val RMSE (scaled), or TrialPruned."""
    hp = space(trial, name)
    pruned = False

    def on_step(epoch: int, best_so_far: float) -> bool:
        nonlocal pruned
        trial.report(best_so_far, epoch)
        pruned = trial.should_prune()
        return pruned

    run_name, tags = f"trial-{trial.number}", {"trial": str(trial.number)}
    with mlflow.start_run(run_name=run_name, nested=True, tags=tags):
        _, info = fit(name, hp, ds, TUNE, seed=0, on_step=on_step)
        mlflow.log_params(hp)
        log_info(info)
        mlflow.set_tag("pruned", str(pruned).lower())
    if pruned:
        raise optuna.TrialPruned()
    return info["val_rmse_scaled"]


def tune(cfg: Config, ds: Dataset, name: ModelName) -> dict:
    """Up to --n-trials finished trials (completed or pruned), seed 0 each."""
    study = open_study(cfg, name)
    remaining = cfg.n_trials - len(study.get_trials(deepcopy=False, states=FINISHED))
    with mlflow.start_run(run_name=f"tune/{name}"):
        if remaining > 0:
            # a trial that raises stops the study with its traceback; re-run to continue
            study.optimize(lambda trial: run_trial(trial, name, ds, cfg), n_trials=remaining)
        best, val = study.best_params, study.best_value
        mlflow.log_dict(best, "best_params.json")
        mlflow.log_metric("best_val_rmse", val)  # scaled units, as the objective
    print(f"[tune] {name}: best val RMSE = {val * ds.y_scale:.2f} (scaled {val:.4f})  {best}")
    return best
