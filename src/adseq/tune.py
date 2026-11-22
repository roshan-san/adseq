"""Optuna search per model, the same number of completed trials for every model, resumable from
runs/<name>/optuna-<model>.db (a file per model, so models can be tuned in parallel), which is
also where evaluate reads the best hyper-parameters. MLflow: a tune/<model> run (the best
hyper-parameters as params, best_val_rmse in ft^3/min and best_val_rmse_scaled) with a trial-<n>
child per trial (hyper-parameters, learning curve, fit info).
"""

import mlflow
import optuna
from optuna.trial import FrozenTrial, Trial, TrialState

from .config import TUNE, Config, ModelName
from .data import Dataset
from .models import fit, space

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)  # our own progress lines instead; failures still show


def open_study(cfg: Config, name: ModelName) -> optuna.Study:
    """Created empty if new."""
    return optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{(cfg.run_dir / f'optuna-{name}.db').as_posix()}",  # / on Windows too
        direction="minimize",
        load_if_exists=True,
    )


def n_done(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


def run_trial(trial: Trial, name: ModelName, ds: Dataset) -> float:
    """Its best val RMSE (scaled)."""
    hp = space(trial, name)
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        _, info = fit(name, hp, ds, TUNE, seed=0)
        mlflow.log_params(hp)
        mlflow.log_metrics(info)
    return info["val_rmse_scaled"]


def tune(cfg: Config, ds: Dataset, name: ModelName) -> dict:
    """Up to --n-trials completed trials, seed 0 each."""
    study = open_study(cfg, name)
    done = n_done(study)

    def progress(study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state == TrialState.COMPLETE and trial.value is not None:
            minutes = trial.duration.total_seconds() / 60 if trial.duration else float("nan")
            print(
                f"[tune] {name} trial {n_done(study)}/{cfg.n_trials}: "
                f"val RMSE {trial.value * ds.y_scale:.2f}, "
                f"best {study.best_value * ds.y_scale:.2f} ft^3/min ({minutes:.1f} min)"
            )

    if done < cfg.n_trials:
        print(
            f"[tune] {name}: {done}/{cfg.n_trials} trials done, running {cfg.n_trials - done} more"
        )
    else:
        print(f"[tune] {name}: already tuned ({done}/{cfg.n_trials} trials)")
    with mlflow.start_run(run_name=f"tune/{name}"):
        if done < cfg.n_trials:
            # a trial that raises stops the study with its traceback; re-run to continue
            study.optimize(
                lambda trial: run_trial(trial, name, ds),
                n_trials=cfg.n_trials - done,
                callbacks=[progress],
            )
        best, val = study.best_params, study.best_value
        mlflow.log_params(best)  # to look at; evaluate reads them, typed, from the study
        mlflow.log_metrics({"best_val_rmse_scaled": val, "best_val_rmse": val * ds.y_scale})
    print(f"[tune] {name}: done, best val RMSE {val * ds.y_scale:.2f} ft^3/min  {best}")
    return best
