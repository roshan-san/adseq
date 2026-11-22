"""Optuna search per model, the same number of completed trials for every model, resumable from
runs/<name>/optuna-<model>.db (a file per model, so models can be tuned in parallel), which is
also where evaluate reads the best hyper-parameters. A trial trains once per --trial-seeds seed
and scores their mean val RMSE, so one lucky seed does not pick the hyper-parameters. MLflow: a
tune/<model> run (the best hyper-parameters as params, best_val_rmse in ft^3/min and
best_val_rmse_scaled) with a trial-<n> child per trial (hyper-parameters, mean and std of the
seeds' val RMSE), and a seed-<s> child of that per seed (learning curve, fit info).
"""

import mlflow
import numpy as np
import optuna
from optuna.trial import FrozenTrial, Trial, TrialState

from .config import PROTOCOL, TUNE, Config, ModelName, Training
from .data import Dataset
from .models import fit, space

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)  # our own progress lines instead; failures still show


def open_study(cfg: Config, name: ModelName) -> optuna.Study:
    """Created empty if new, stamped with PROTOCOL and the trial seeds; TPE sampler, seeded. One tuned under another
    PROTOCOL is refused: its trials are not comparable with this code's."""
    path = cfg.run_dir / f"optuna-{name}.db"
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{path.as_posix()}",  # / on Windows too
        direction="minimize",
        load_if_exists=True,
    )
    # seeded, so a search is reproducible; by the trials already there, so a resumed search does
    # not redraw the first trials' random hyper-parameters
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    stamp = {"protocol": PROTOCOL, "trial_seeds": list(cfg.trial_seeds)}
    if not study.trials:
        for k, v in stamp.items():
            study.set_user_attr(k, v)
    elif study.user_attrs.get("protocol") != PROTOCOL:
        raise SystemExit(
            f"{path} was tuned under protocol {study.user_attrs.get('protocol', 1)}, "
            f"this code is protocol {PROTOCOL}: tune again under a new --name"
        )
    return study


def n_done(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


def run_trial(
    trial: Trial, name: ModelName, ds: Dataset, train: Training, seeds: tuple[int, ...]
) -> float:
    """The mean over seeds of each one's best val RMSE (scaled)."""
    hp = space(trial, name)
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(hp)
        rmses = []
        for seed in seeds:
            with mlflow.start_run(run_name=f"seed-{seed}", nested=True, tags={"seed": str(seed)}):
                _, info = fit(name, hp, ds, train, seed)
                mlflow.log_metrics(info)
            rmses.append(info["val_rmse_scaled"])
        mean = float(np.mean(rmses))
        mlflow.log_metrics({"val_rmse_scaled": mean, "val_rmse_scaled_std": float(np.std(rmses))})
    return mean


def tune(cfg: Config, ds: Dataset, name: ModelName) -> dict:
    """Up to --n-trials completed trials, --trial-seeds seeds each."""
    study = open_study(cfg, name)
    done = n_done(study)
    if done and (seeds := study.user_attrs.get("trial_seeds")) != list(cfg.trial_seeds):
        raise SystemExit(f"{name} was tuned with --trial-seeds {seeds}: continue with the same")

    def progress(study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state == TrialState.COMPLETE and trial.value is not None:
            minutes = trial.duration.total_seconds() / 60 if trial.duration else float("nan")
            print(
                f"[tune] {name} trial {n_done(study)}/{cfg.n_trials}: "
                f"val RMSE {trial.value * ds.y_scale:.2f} (mean of {len(cfg.trial_seeds)} seeds), "
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
                lambda trial: run_trial(trial, name, ds, cfg.training(TUNE), cfg.trial_seeds),
                n_trials=cfg.n_trials - done,
                callbacks=[progress],
            )
        best, val = study.best_params, study.best_value
        mlflow.log_params(best)  # to look at; evaluate reads them, typed, from the study
        mlflow.log_metrics({"best_val_rmse_scaled": val, "best_val_rmse": val * ds.y_scale})
    print(f"[tune] {name}: done, best val RMSE {val * ds.y_scale:.2f} ft^3/min  {best}")
    return best
