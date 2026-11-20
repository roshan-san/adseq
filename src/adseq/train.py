"""Fit any model and predict every sample.

Persistence and ridge have no training loop. Every recurrent model shares one procedure:
Adam, MSE or L1 loss (a tuned choice), mini-batches, gradient clipping, early stopping on
validation RMSE; the weights from the best validation epoch are kept. LightGBM is boosted with
the same loss choice and early stopping on the same validation RMSE (one "epoch" = one tree).

The recurrent models train on the GPU when there is one (install the CUDA build of PyTorch with
`uv sync --extra cu132`), else on the CPU; the code is the same. LightGBM always uses the CPU.
"""

import copy
import time

import lightgbm as lgb
import mlflow
import numpy as np
import torch
from mlflow.entities import Metric
from torch import nn

from .config import Config
from .data import Dataset
from .models import (
    LGBM_MAX_ROUNDS,
    LGBM_PATIENCE,
    TREE_MODELS,
    RecurrentNet,
    fit_ridge,
    is_residual,
    make_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # recurrent models
Curve = list[tuple[int, float, float]]  # (epoch, mean train loss, val RMSE), scaled units


def train_and_predict(
    name: str, hp: dict, ds: Dataset, cfg: Config, seed: int
) -> tuple[np.ndarray, dict, Curve]:
    """Returns predictions (ft^3/min) for every sample, training info, and the learning curve."""
    if name == "persistence":
        return ds.y_prev, {"epochs": 0, "n_params": 0}, []
    if name == "ridge":
        return predict_ridge(ds)
    if name in TREE_MODELS:
        booster, info, curve = fit_lgbm(name, hp, ds, seed)
        return predict_lgbm(booster, name, hp, ds), info, curve
    model, info, curve = fit_recurrent(name, hp, ds, cfg, seed)
    return predict_recurrent(model, name, hp, ds), info, curve


def predict_ridge(ds: Dataset) -> tuple[np.ndarray, dict, Curve]:
    """Fit on train, penalty chosen on val."""
    train, val = ds.part("train"), ds.part("val")
    X = ds.X
    w, b, alpha, rmse = fit_ridge(X[train], ds.y[train], X[val], ds.y[val])
    info = {"epochs": 0, "val_rmse_scaled": rmse, "alpha": alpha, "n_params": w.size + 1}
    return ds.unscale(X @ w + b), info, []


def fit_tuned(name: str, hp: dict, ds: Dataset, cfg: Config, seed: int) -> tuple[dict, Curve]:
    """Fit a tuned model (no test predictions): training info and learning curve, for tuning."""
    if name in TREE_MODELS:
        _, info, curve = fit_lgbm(name, hp, ds, seed)
    else:
        _, info, curve = fit_recurrent(name, hp, ds, cfg, seed)
    return info, curve


def base_level(name: str, ds: Dataset) -> np.ndarray:
    """What the model's output is added to (scaled units): persistence for the residual models,
    zero for the others."""
    return ds.scale(ds.y_prev) if is_residual(name) else np.zeros_like(ds.y)


def flat_windows(ds: Dataset, idx: np.ndarray, window: int) -> np.ndarray:
    """(len(idx), L * F): the recurrent models' input window, one column per (hour, feature)."""
    return ds.windows(idx, window).reshape(len(idx), -1)


def fit_lgbm(name: str, hp: dict, ds: Dataset, seed: int) -> tuple[lgb.Booster, dict, Curve]:
    """Boost on the training samples; keep the trees up to the best validation round."""
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
    d_train = lgb.Dataset(flat_windows(ds, train, hp["window"]), y[train], free_raw_data=True)
    d_val = d_train.create_valid(flat_windows(ds, val, hp["window"]), y[val])
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
    curve = [(i + 1, evals["train"][loss][i], val_rmse[i]) for i in range(len(val_rmse))]
    trees = booster.dump_model(num_iteration=best)["tree_info"]
    info = {
        "epochs": best,
        "val_rmse_scaled": val_rmse[best - 1],
        "n_params": sum(t["num_leaves"] for t in trees),  # one fitted value per leaf
        "device": "cpu",
    }
    return booster, info, curve


def predict_lgbm(booster: lgb.Booster, name: str, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample, in chunks like the recurrent models."""
    chunks = np.array_split(np.arange(len(ds.y)), max(1, len(ds.y) // 4096))
    out = np.concatenate(
        [
            booster.predict(flat_windows(ds, c, hp["window"]), num_iteration=booster.best_iteration)
            for c in chunks
        ]
    )
    return ds.unscale(out + base_level(name, ds))


def fit_recurrent(
    name: str, hp: dict, ds: Dataset, cfg: Config, seed: int
) -> tuple[RecurrentNet, dict, Curve]:
    """Train on the training samples; return the best-validation-epoch model."""
    train, val = ds.part("train"), ds.part("val")
    y = torch.from_numpy((ds.y - base_level(name, ds)).astype(np.float32))

    def inputs(idx: np.ndarray) -> torch.Tensor:  # window: hours, tuned
        return torch.from_numpy(ds.windows(idx, hp["window"])).to(DEVICE)

    torch.manual_seed(seed)  # also seeds CUDA; the weights are initialised on the CPU either way
    model = make_model(name, hp, ds.n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.L1Loss() if hp["loss"] == "l1" else nn.MSELoss()
    # validation in chunks: with every training minute (--stride 1) the ~77k validation windows
    # would take ~0.8 GB at once
    val_chunks = np.array_split(val, max(1, len(val) // 4096))

    def val_rmse() -> float:
        model.eval()
        sq = 0.0
        with torch.no_grad():
            for c in val_chunks:
                sq += (model(inputs(c)) - y[c].to(DEVICE)).pow(2).sum().item()
        return (sq / len(val)) ** 0.5

    # epoch 0 = untrained weights, so there is always a state to restore
    best_rmse, best_state, best_epoch = val_rmse(), copy.deepcopy(model.state_dict()), 0
    curve = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        loss_sum = 0.0
        for i in range(0, len(order), hp["batch_size"]):
            batch = order[i : i + hp["batch_size"]]
            loss = loss_fn(model(inputs(batch)), y[batch].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(batch)

        rmse = val_rmse()
        curve.append((epoch, loss_sum / len(order), rmse))
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = rmse, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    info = {
        "epochs": best_epoch,
        "val_rmse_scaled": best_rmse,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": DEVICE.type,
    }
    return model, info, curve


def predict_recurrent(model: RecurrentNet, name: str, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample."""
    model.eval()
    with torch.no_grad():  # in chunks: ~100k samples x window would not fit in memory at once
        chunks = np.array_split(np.arange(len(ds.y)), max(1, len(ds.y) // 4096))
        out = np.concatenate(
            [
                model(torch.from_numpy(ds.windows(c, hp["window"])).to(DEVICE)).cpu().numpy()
                for c in chunks
            ]
        )
    return ds.unscale(out.astype(float) + base_level(name, ds))


def log_curve(curve: Curve) -> None:
    """Learning curve to the active MLflow run; epoch_* so it can't clash with the final
    metrics. Sent in batches: LightGBM curves can be thousands of rounds long."""
    t = int(time.time() * 1000)
    metrics = [
        Metric(key, float(v), t, epoch)
        for epoch, train_loss, val_rmse in curve
        for key, v in (("epoch_train_loss", train_loss), ("epoch_val_rmse", val_rmse))
    ]
    run_id = mlflow.active_run().info.run_id
    client = mlflow.MlflowClient()
    for i in range(0, len(metrics), 1000):  # MLflow's per-batch limit
        client.log_batch(run_id, metrics=metrics[i : i + 1000])
