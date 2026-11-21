"""Equal-budget hyper-parameter search with Optuna (TPE sampler, fixed seed).

Every tuned model gets the same number of finished trials (--n-trials, completed or pruned), each
trained with seed 0 and scored on the same validation samples. Each model's study is saved in
runs/<name>/optuna/<model>.log, so re-running continues it: finished trials are kept, and a trial
cut off by a crash stays "running" and does not count. Run one process per model at a time.
Every trial is an MLflow run (tags trial, pruned) with its learning curve, nested under the
model's tune/<model> run, which is reopened on a resume like the study and holds the best
hyper-parameters (best_params.json) and best_val_rmse.

Pruning (MedianPruner, recurrent models only): after the first PRUNE_STARTUP finished trials, a
trial stops early when its best val RMSE so far is worse than the median of earlier trials at the
same epoch, checked every epoch once it has made PRUNE_WARMUP_PASSES passes over the training data
(that many epochs x --stride, so the warmup is the same amount of training at any stride). Its
curve up to that point is kept; it never becomes the best trial. LightGBM trials are not pruned: early stopping already ends a poor
one.

The objective `val_rmse` is in scaled units (target min-max scaled on the training period);
the `[tune]` line also prints it in ft^3/min.
"""

import mlflow
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial, TrialState

from .config import Config
from .data import WINDOWS, Dataset
from .fit import fit
from .models import TREE_MODELS
from .tracking import child_run, log_info, tune_run

PRUNE_STARTUP = 5  # finished trials before any pruning (Optuna's MedianPruner default)
# passes over the training data before a trial may be pruned: slow starters (low learning rate)
# need it; the tuned gru_res is still 5.8 % above its final val RMSE after 2 passes, 1.9 % after 3
PRUNE_WARMUP_PASSES = 3
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)  # what counts towards --n-trials


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


def open_study(cfg: Config, name: str) -> optuna.Study:
    """The model's study, new or resumed from its journal file."""
    study_file = cfg.run_dir / "optuna" / f"{name}.log"
    study_file.parent.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=JournalStorage(JournalFileBackend(str(study_file))),
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_STARTUP, n_warmup_steps=PRUNE_WARMUP_PASSES * cfg.tune.stride
        ),
    )
    # seed 0 for a new study, so a run without interruptions can be replayed exactly; after a
    # resume the seed moves on, else the sampler's random first trials would repeat earlier ones
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    return study


def run_trial(trial: Trial, name: str, ds: Dataset, cfg: Config) -> float:
    """Train one suggested configuration; its best val RMSE (or TrialPruned)."""
    hp = suggest(trial, name)
    pruned = False

    def on_step(epoch: int, best_so_far: float) -> bool:
        nonlocal pruned
        trial.report(best_so_far, epoch)
        pruned = trial.should_prune()
        return pruned

    with child_run(f"trial-{trial.number}", trial=str(trial.number)):
        _, info = fit(name, hp, ds, cfg.tune, seed=0, on_step=on_step)
        mlflow.log_params(hp)
        log_info(info)
        mlflow.set_tag("pruned", str(pruned).lower())
    if pruned:
        raise optuna.TrialPruned()
    return info["val_rmse_scaled"]


def tune(cfg: Config, ds: Dataset, name: str) -> dict:
    """Run the model's study up to --n-trials finished trials; log and return the best
    hyper-parameters (best_params.json on tune/<model>, typed, overwritten as tuning goes on)."""
    study = open_study(cfg, name)
    remaining = cfg.n_trials - len(study.get_trials(deepcopy=False, states=FINISHED))
    with tune_run(cfg, name):
        if remaining > 0:
            # a trial that raises stops the study with its traceback; re-run to continue
            study.optimize(lambda trial: run_trial(trial, name, ds, cfg), n_trials=remaining)
        best, val = study.best_params, study.best_value
        mlflow.log_dict(best, "best_params.json")
        mlflow.log_metric("best_val_rmse", val)  # scaled units, as the objective
    print(f"[tune] {name}: best val RMSE = {val * ds.y_scale:.2f} (scaled {val:.4f})  {best}")
    return best
