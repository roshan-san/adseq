"""Fit any model and predict every sample.

Persistence and ridge have no training loop. Every recurrent model shares one procedure:
Adam, MSE or L1 loss (a tuned choice), mini-batches, gradient clipping, early stopping on
validation RMSE; the weights from the best validation epoch are kept.
"""

import copy

import numpy as np
import torch
from torch import nn

from .config import Config
from .data import Dataset
from .models import RESIDUAL_MODELS, fit_ridge, make_model

Curve = list[tuple[int, float, float]]  # (epoch, mean train loss, val RMSE), scaled units


def train_and_predict(
    name: str, hp: dict, ds: Dataset, cfg: Config, seed: int, predict: bool = True
) -> tuple[np.ndarray | None, dict, Curve]:
    """Returns predictions (ft^3/min) for every sample, training info, and the learning curve.
    predict=False (tuning) skips the prediction pass and returns None for it."""
    if name == "persistence":
        return ds.y_prev, {"epochs": 0, "n_params": 0}, []
    if name == "ridge":
        return predict_ridge(ds)
    return train_recurrent(name, hp, ds, cfg, seed, predict)


def predict_ridge(ds: Dataset) -> tuple[np.ndarray, dict, Curve]:
    """Fit on train, penalty chosen on val."""
    train, val = ds.part("train"), ds.part("val")
    X = ds.X
    w, b, alpha, rmse = fit_ridge(X[train], ds.y[train], X[val], ds.y[val])
    info = {"epochs": 0, "val_rmse_scaled": rmse, "alpha": alpha, "n_params": w.size + 1}
    return ds.unscale(X @ w + b), info, []


def train_recurrent(
    name: str, hp: dict, ds: Dataset, cfg: Config, seed: int, predict: bool
) -> tuple[np.ndarray | None, dict, Curve]:
    train, val = ds.part("train"), ds.part("val")

    # residual models learn the change from persistence, in the same scaled units
    base = ds.scale(ds.y_prev) if name in RESIDUAL_MODELS else np.zeros_like(ds.y)
    y = torch.from_numpy((ds.y - base).astype(np.float32))

    def inputs(idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(ds.windows(idx, hp["window"]))  # window: hours, tuned

    torch.manual_seed(seed)
    model = make_model(name, hp, ds.n_features)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.L1Loss() if hp["loss"] == "l1" else nn.MSELoss()
    X_val = inputs(val)

    def val_rmse() -> float:
        model.eval()
        with torch.no_grad():
            return (model(X_val) - y[val]).pow(2).mean().sqrt().item()

    # epoch 0 = untrained weights, so there is always a state to restore
    best_rmse, best_state, best_epoch = val_rmse(), copy.deepcopy(model.state_dict()), 0
    curve = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        loss_sum = 0.0
        for i in range(0, len(order), hp["batch_size"]):
            batch = order[i : i + hp["batch_size"]]
            loss = loss_fn(model(inputs(batch)), y[batch])
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
    }
    if not predict:
        return None, info, curve
    model.eval()
    with torch.no_grad():  # in chunks: ~100k samples x window would not fit in memory at once
        chunks = np.array_split(np.arange(len(ds.y)), max(1, len(ds.y) // 4096))
        out = np.concatenate([model(inputs(c)).numpy() for c in chunks])
    return ds.unscale(out.astype(float) + base), info, curve


def log_curve(run, curve: Curve) -> None:
    """Learning curve to W&B; epoch_* so it can't clash with the final metrics."""
    for epoch, train_loss, val_rmse in curve:
        run.log({"epoch_train_loss": train_loss, "epoch_val_rmse": val_rmse}, step=epoch)
