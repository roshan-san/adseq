"""The recurrent models: rnn / gru / lstm and their *_res versions.

Every recurrent model shares one procedure: AdamW (weight decay applied to the weights directly,
not through the gradient), MSE or L1 loss (a tuned choice), mini-batches, gradient clipping,
early stopping on validation RMSE; the weights from the best validation epoch are kept. The
learning curve (epoch_train_loss / epoch_val_rmse, scaled units) goes into the active MLflow run
after every epoch, so it can be watched in the UI.

They train on the GPU when there is one (install the CUDA build of PyTorch with
`uv sync --extra cu132`), else on the CPU; the code is the same.
"""

import copy
from collections.abc import Callable

import mlflow
import numpy as np
import torch
from optuna.trial import Trial
from torch import nn

from ..config import Training
from ..data import WINDOWS, Dataset, chunks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}  # cell name -> torch layer

# called after every epoch with (epoch, best val RMSE so far); returning True stops it (pruned)
OnStep = Callable[[int, float], bool]


class RecurrentNet(nn.Module):
    """Reads a (batch, L, F) window hour by hour, oldest first; its hidden state after the
    last hour goes through dropout and a linear head to one number per window."""

    def __init__(self, cell: str, n_in: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = CELLS[cell](
            n_in,
            hidden_size,
            num_layers,
            batch_first=True,  # input is (batch, L, F), not (L, batch, F)
            # torch applies this dropout only between stacked layers, so 1 layer must get 0
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)  # before the head: applies for any number of layers
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (batch, L, hidden): the hidden state after each hour
        return self.head(self.dropout(out[:, -1])).squeeze(-1)  # last hour only -> (batch,)


def make_model(cell: str, residual: bool, hp: dict, n_features: int) -> RecurrentNet:
    """The network: "gru" and "gru_res" are the same GRU network, they differ only in what they
    are trained to predict (Dataset.base_level). A residual model's head starts at zero, so the
    untrained model predicts exactly persistence; the head learns from the first step and the
    rest of the network from the second."""
    model = RecurrentNet(cell, n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    if residual:
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    return model


def space(trial: Trial) -> dict:
    """Search space: the same ranges for every recurrent model."""
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


def net_input(ds: Dataset, idx: np.ndarray, window: int) -> torch.Tensor:
    """(len(idx), L, F) windows on DEVICE: a recurrent model's input."""
    return torch.from_numpy(ds.windows(idx, window)).to(DEVICE)


def fit(
    cell: str,
    residual: bool,
    hp: dict,
    ds: Dataset,
    train: Training,
    seed: int,
    on_step: OnStep | None,
) -> tuple[RecurrentNet, dict]:
    """Train on the training samples; return the best-validation-epoch model. After every
    epoch on_step (if given) sees the best val RMSE so far and may stop the training."""
    train_idx, val_idx = ds.part("train"), ds.part("val")
    y = torch.from_numpy((ds.y - ds.base_level(residual)).astype(np.float32))

    torch.manual_seed(seed)  # also seeds CUDA; the weights are initialised on the CPU either way
    model = make_model(cell, residual, hp, ds.n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.L1Loss() if hp["loss"] == "l1" else nn.MSELoss()
    val_chunks = chunks(val_idx)  # the ~77k validation windows would take ~0.8 GB at once

    def val_rmse() -> float:
        model.eval()
        sq = 0.0
        with torch.no_grad():
            for c in val_chunks:
                pred = model(net_input(ds, c, hp["window"]))
                sq += (pred - y[c].to(DEVICE)).pow(2).sum().item()
        return (sq / len(val_idx)) ** 0.5

    def train_epoch() -> float:
        """One pass over the shuffled training samples; the mean training loss."""
        model.train()
        order = train_idx[torch.randperm(len(train_idx)).numpy()]
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
    for epoch in range(1, train.max_epochs + 1):
        train_loss = train_epoch()
        rmse = val_rmse()
        mlflow.log_metrics({"epoch_train_loss": train_loss, "epoch_val_rmse": rmse}, step=epoch)
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = rmse, copy.deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= train.patience or (on_step and on_step(epoch, best_rmse)):
            break

    model.load_state_dict(best_state)
    info = {
        "epochs": best_epoch,
        "val_rmse_scaled": best_rmse,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": DEVICE.type,
    }
    return model, info


def predict(net: RecurrentNet, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample, CHUNK samples at a time."""
    net.eval()
    with torch.no_grad():
        out = np.concatenate(
            [
                net(net_input(ds, c, hp["window"])).cpu().numpy().astype(float)
                for c in chunks(np.arange(len(ds.y)))
            ]
        )
    return ds.unscale(out + ds.base_level(residual))
