"""Paper Sec. IV-C: one training procedure shared by every neural model.

Adam + MSE loss, mini-batches, gradient clipping, early stopping on validation RMSE,
and the weights from the best validation epoch are kept.
"""

import copy

import numpy as np
import torch
from torch import nn

from .config import Config
from .data import Dataset
from .models import WINDOW_MODELS, make_model


def train_and_predict(
    name: str,
    hp: dict,
    ds: Dataset,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, dict, list[tuple[int, float, float]]]:
    """Returns predictions (original units) for every sample, training info, and the learning
    curve: one (epoch, mean train MSE, val RMSE) per epoch, both in scaled units."""
    if name == "persistence":  # tomorrow = today
        return ds.y_prev, {"epochs": 0, "n_params": 0}, []

    torch.manual_seed(seed)
    X = torch.from_numpy(ds.X_seq if name in WINDOW_MODELS else ds.X)
    y = torch.from_numpy(ds.y)
    train, val = ds.part("train"), ds.part("val")

    model = make_model(name, hp, X.shape[-1], ds.X_seq.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    def val_rmse_now() -> float:
        model.eval()
        with torch.no_grad():
            return (model(X[val]) - y[val]).pow(2).mean().sqrt().item()

    # epoch 0 = untrained weights, so there is always a state to restore
    best_rmse, best_state, best_epoch = val_rmse_now(), copy.deepcopy(model.state_dict()), 0
    curve = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        sse = 0.0  # summed squared error over the epoch, for the learning curve
        for i in range(0, len(order), hp["batch_size"]):
            batch = order[i : i + hp["batch_size"]]
            loss = loss_fn(model(X[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sse += loss.item() * len(batch)

        val_rmse = val_rmse_now()
        curve.append((epoch, sse / len(order), val_rmse))
        if val_rmse < best_rmse:
            best_rmse, best_state, best_epoch = val_rmse, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = ds.unscale(model(X).numpy().astype(float))
    info = {
        "epochs": best_epoch,
        "val_rmse_scaled": best_rmse,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return pred, info, curve
