"""Fit any model, then predict every sample.

fit(name, ...) -> (fitted, info) and predict(fitted, name, ...) -> ft^3/min are the
only entry points; each model family has its own section below. fit is always called inside an
MLflow run (a tuning trial or an evaluation seed) and logs the learning curve into it as it
trains (epoch_train_loss / epoch_val_rmse, scaled units), so it can be watched in the UI.

Persistence has nothing to fit and ridge no training loop. Every recurrent model shares one
procedure: AdamW (weight decay applied to the weights directly, not through the gradient), MSE
or L1 loss (a tuned choice), mini-batches, gradient clipping, early stopping on validation RMSE;
the weights from the best validation epoch are kept. LightGBM is boosted with the same loss
choice and early stopping on the same validation RMSE (one "epoch" = one tree).

The recurrent models train on the GPU when there is one (install the CUDA build of PyTorch with
`uv sync --extra cu132`), else on the CPU; the code is the same. LightGBM always uses the CPU.
"""

import copy
from collections.abc import Callable

import lightgbm as lgb
import mlflow
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import Config
from .data import Dataset
from .models import TREE_MODELS, RecurrentNet, is_residual, make_model
from .tracking import log_lgbm_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # recurrent models
CHUNK = 4096  # samples per forward pass when predicting: all ~490k windows would not fit
RIDGE_CV_GAP = 300  # their notebook: rows left out between each CV fold's train and val rows
LGBM_MAX_ROUNDS = 5000
LGBM_PATIENCE = 100  # boosting rounds without a better val RMSE

# called after every epoch with (epoch, best val RMSE so far); returning True stops it (pruned)
OnStep = Callable[[int, float], bool]
Fitted = RecurrentNet | lgb.Booster | Pipeline | None  # None: persistence (nothing is fitted)


def fit(
    name: str, hp: dict, ds: Dataset, cfg: Config, seed: int, on_step: OnStep | None = None
) -> tuple[Fitted, dict]:
    """The fitted model and training info. on_step (recurrent models only) can
    stop the training early: Optuna pruning."""
    if name == "persistence":
        return None, {"epochs": 0, "n_params": 0}
    if name == "ridge":
        return fit_ridge(ds)
    if name in TREE_MODELS:
        return fit_lgbm(name, hp, ds, seed)
    return fit_recurrent(name, hp, ds, cfg, seed, on_step)


def predict(fitted: Fitted, name: str, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample."""
    if fitted is None:
        return ds.y_prev
    if isinstance(fitted, Pipeline):
        return ds.unscale(fitted.predict(ds.X.astype(float)))
    out = np.concatenate([predict_chunk(fitted, hp, ds, c) for c in chunks(np.arange(len(ds.y)))])
    return ds.unscale(out + base_level(name, ds))


def predict_chunk(fitted: RecurrentNet | lgb.Booster, hp: dict, ds: Dataset, idx) -> np.ndarray:
    """A windowed model's raw output (scaled units, before adding base_level) for samples idx."""
    if isinstance(fitted, lgb.Booster):
        return fitted.predict(
            tree_input(ds, idx, hp["window"]), num_iteration=fitted.best_iteration
        )
    fitted.eval()
    with torch.no_grad():
        return fitted(net_input(ds, idx, hp["window"])).cpu().numpy().astype(float)


def chunks(idx: np.ndarray) -> list[np.ndarray]:
    """idx in pieces of about CHUNK samples."""
    return np.array_split(idx, max(1, len(idx) // CHUNK))


def base_level(name: str, ds: Dataset) -> np.ndarray:
    """What the model's output is added to (scaled units): persistence for the residual models,
    zero for the others."""
    return ds.scale(ds.y_prev) if is_residual(name) else np.zeros_like(ds.y)


def net_input(ds: Dataset, idx: np.ndarray, window: int) -> torch.Tensor:
    """(len(idx), L, F) windows on DEVICE: a recurrent model's input."""
    return torch.from_numpy(ds.windows(idx, window)).to(DEVICE)


def tree_input(ds: Dataset, idx: np.ndarray, window: int) -> np.ndarray:
    """(len(idx), L * F): the same windows flattened, one column per (hour, feature)."""
    return ds.windows(idx, window).reshape(len(idx), -1)


# --- ridge ------------------------------------------------------------------------------------


def fit_ridge(ds: Dataset) -> tuple[Pipeline, dict]:
    """Their linear model (SCADA-linear-regression.ipynb): StandardScaler + RidgeCV with its
    default penalties (0.1, 1, 10); the one with the best R2 over 5 TimeSeriesSplit folds is
    refit on all rows. Fit on every pre-test row (train + val, as theirs saw their whole training
    set), so no val split is needed; every minute, whatever the stride. float64, as the lag
    features are collinear."""
    rows = np.flatnonzero(ds.split != "test")
    model = make_pipeline(StandardScaler(), RidgeCV(cv=TimeSeriesSplit(gap=RIDGE_CV_GAP)))
    model.fit(ds.X[rows].astype(float), ds.y[rows].astype(float))
    ridge = model[-1]
    info = {"epochs": 0, "alpha": float(ridge.alpha_), "n_params": ridge.coef_.size + 1}
    return model, info


# --- LightGBM ---------------------------------------------------------------------------------


def fit_lgbm(name: str, hp: dict, ds: Dataset, seed: int) -> tuple[lgb.Booster, dict]:
    """Boost on the training samples; keep the trees up to the best validation round. Not
    pruned while tuning: early stopping already ends a poor trial."""
    train, val = ds.part("train"), ds.part("val")
    y = ds.y - base_level(name, ds)
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
    train_loss, val_rmse = evals["train"][loss], evals["val"]["rmse"]
    log_lgbm_curve(train_loss, val_rmse)
    trees = booster.dump_model(num_iteration=best)["tree_info"]
    info = {
        "epochs": best,
        "val_rmse_scaled": val_rmse[best - 1],
        "n_params": sum(t["num_leaves"] for t in trees),  # one fitted value per leaf
        "device": "cpu",
    }
    return booster, info


# --- recurrent --------------------------------------------------------------------------------


def fit_recurrent(
    name: str, hp: dict, ds: Dataset, cfg: Config, seed: int, on_step: OnStep | None = None
) -> tuple[RecurrentNet, dict]:
    """Train on the training samples; return the best-validation-epoch model. After every
    epoch on_step (if given) sees the best val RMSE so far and may stop the training."""
    train, val = ds.part("train"), ds.part("val")
    y = torch.from_numpy((ds.y - base_level(name, ds)).astype(np.float32))

    torch.manual_seed(seed)  # also seeds CUDA; the weights are initialised on the CPU either way
    model = make_model(name, hp, ds.n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.L1Loss() if hp["loss"] == "l1" else nn.MSELoss()
    val_chunks = chunks(val)  # the ~77k validation windows would take ~0.8 GB at once

    def val_rmse() -> float:
        model.eval()
        sq = 0.0
        with torch.no_grad():
            for c in val_chunks:
                pred = model(net_input(ds, c, hp["window"]))
                sq += (pred - y[c].to(DEVICE)).pow(2).sum().item()
        return (sq / len(val)) ** 0.5

    def train_epoch() -> float:
        """One pass over the shuffled training samples; the mean training loss."""
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        loss_sum = 0.0
        for i in range(0, len(order), hp["batch_size"]):
            batch = order[i : i + hp["batch_size"]]
            loss = loss_fn(model(net_input(ds, batch, hp["window"])), y[batch].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(batch)
        return loss_sum / len(order)

    # epoch 0 = untrained weights, so there is always a state to restore
    best_rmse, best_state, best_epoch = val_rmse(), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_epoch()
        rmse = val_rmse()
        mlflow.log_metrics({"epoch_train_loss": train_loss, "epoch_val_rmse": rmse}, step=epoch)
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = rmse, copy.deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= cfg.patience or (on_step and on_step(epoch, best_rmse)):
            break

    model.load_state_dict(best_state)
    info = {
        "epochs": best_epoch,
        "val_rmse_scaled": best_rmse,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": DEVICE.type,
    }
    return model, info
