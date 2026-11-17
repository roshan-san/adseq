"""Models.

persistence   next-24-h mean = mean of the previous clock hour (Schroer & Just's baseline)
ridge         linear model on the minute-t features, penalty chosen on validation
(Schroer & Just's own MLP is run from their notebooks: scripts/run_schroer.py)
rnn/gru/lstm  recurrent encoder over the last L hours -> last hidden state -> linear
*_res         proposed: the same recurrent encoder predicts the *change* from the last hour's
              biogas and adds it to persistence, so it only has to learn a correction
"""

import numpy as np
import torch
from torch import nn

RESIDUAL_MODELS = ("rnn_res", "gru_res", "lstm_res")  # proposed
RECURRENT_MODELS = ("rnn", "gru", "lstm", *RESIDUAL_MODELS)
UNTRAINED_MODELS = ("persistence", "ridge")  # deterministic: one run, no sweep
RIDGE_ALPHAS = np.logspace(-4, 3, 50)
CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RecurrentNet(nn.Module):
    """Recurrent encoder over a (batch, L, F) window -> last hidden state -> one number."""

    def __init__(self, cell: str, n_in: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = CELLS[cell](
            n_in,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # torch only drops between layers
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # out: (batch, L, hidden)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


def make_model(name: str, hp: dict, n_features: int) -> RecurrentNet:
    """Net for a model name ("gru" and "gru_res" are both GRUs). A residual model's output layer starts at zero, so the
    untrained model predicts exactly persistence and training only learns the correction."""
    model = RecurrentNet(
        name.removesuffix("_res"), n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"]
    )
    if name in RESIDUAL_MODELS:
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    return model


def fit_ridge(
    X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Closed-form ridge with an unpenalised intercept, in float64 (the lag features are
    collinear); the penalty with the lowest validation RMSE wins.
    Returns (weights, bias, alpha, val RMSE)."""
    X, y, X_val, y_val = (np.asarray(a, float) for a in (X, y, X_val, y_val))
    x_mean, y_mean = X.mean(0), y.mean()
    Xc = X - x_mean
    gram, xty = Xc.T @ Xc, Xc.T @ (y - y_mean)

    def fit(alpha: float) -> tuple[np.ndarray, float, float, float]:
        w = np.linalg.solve(gram + alpha * np.eye(len(gram)), xty)
        b = float(y_mean - x_mean @ w)
        rmse = float(np.sqrt(np.mean((X_val @ w + b - y_val) ** 2)))
        return w, b, alpha, rmse

    return min((fit(float(a)) for a in RIDGE_ALPHAS), key=lambda f: f[3])
