"""LightGBM on the same window the recurrent models see, flattened (one column per hour and
feature): lgbm, and lgbm_res predicting the change from persistence.

Boosted with the same loss choice as the recurrent models and early stopping on the same
validation RMSE (one "epoch" = one tree). Always on the CPU. Its learning curve and booster
params are logged by mlflow.lightgbm.autolog (turned on in tracking.start_tracking).
"""

import lightgbm as lgb
import numpy as np
from optuna.trial import Trial

from ..data import WINDOWS, Dataset, chunks

LGBM_MAX_ROUNDS = 5000
LGBM_PATIENCE = 100  # boosting rounds without a better val RMSE


def space(trial: Trial) -> dict:
    """Search space: LightGBM's own, with the same loss and window choices as the recurrent
    models."""
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


def tree_input(ds: Dataset, idx: np.ndarray, window: int) -> np.ndarray:
    """(len(idx), L * F): the recurrent models' windows flattened, one column per (hour,
    feature)."""
    return ds.windows(idx, window).reshape(len(idx), -1)


def fit(residual: bool, hp: dict, ds: Dataset, seed: int) -> tuple[lgb.Booster, dict]:
    """Boost on the training samples; keep the trees up to the best validation round. Not
    pruned while tuning: early stopping already ends a poor trial."""
    train, val = ds.part("train"), ds.part("val")
    y = ds.y - ds.base_level(residual)
    loss = "l1" if hp["loss"] == "l1" else "l2"
    params = {
        "objective": loss,
        "metric": ["rmse", loss],  # rmse: early stopping (first); loss: the train curve
        "learning_rate": hp["lr"],
        "num_leaves": hp["num_leaves"],
        "min_data_in_leaf": hp["min_data_in_leaf"],
        "feature_fraction": hp["feature_fraction"],
        "bagging_fraction": hp["bagging_fraction"],
        "bagging_freq": 1,
        "lambda_l2": hp["lambda_l2"],
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    d_train = lgb.Dataset(tree_input(ds, train, hp["window"]), y[train], free_raw_data=True)
    d_val = d_train.create_valid(tree_input(ds, val, hp["window"]), y[val])
    evals: dict = {}
    booster = lgb.train(
        params,
        d_train,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[d_train, d_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(LGBM_PATIENCE, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evals),
        ],
    )

    best = booster.best_iteration
    val_rmse = evals["val"]["rmse"]
    trees = booster.dump_model(num_iteration=best)["tree_info"]
    info = {
        "epochs": best,
        "val_rmse_scaled": val_rmse[best - 1],
        "n_params": sum(t["num_leaves"] for t in trees),  # one fitted value per leaf
        "device": "cpu",
    }
    return booster, info


def predict(booster: lgb.Booster, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample, CHUNK samples at a time."""
    out = np.concatenate(
        [
            np.asarray(
                booster.predict(
                    tree_input(ds, c, hp["window"]), num_iteration=booster.best_iteration
                )
            )
            for c in chunks(np.arange(len(ds.y)))
        ]
    )
    return ds.unscale(out + ds.base_level(residual))
