"""Models.

persistence   next-24-h mean = mean of the previous clock hour (Schroer & Just's baseline)
ridge         linear model on the minute-t features, as in Schroer & Just's linear notebook:
              standardised, penalty chosen by TimeSeriesSplit CV on every pre-test row
lgbm          LightGBM on the same window the recurrent models see, flattened (one column per
              feature and hour)
lgbm_res      LightGBM predicting the change from persistence, like the *_res nets (ablation:
              is the gain the residual target or the recurrent encoder?)
(Schroer & Just's own MLP is run from their notebooks: scripts/run_schroer.py)
rnn/gru/lstm  recurrent encoder over the last L hours -> last hidden state -> linear
*_res         proposed: the same recurrent encoder predicts the *change* from the last hour's
              biogas and adds it to persistence, so it only has to learn a correction

How each one is fitted: fit.py.
"""

import torch
from torch import nn

RECURRENT_MODELS = ("rnn", "gru", "lstm", "rnn_res", "gru_res", "lstm_res")  # *_res: proposed
TREE_MODELS = ("lgbm", "lgbm_res")
TUNED_MODELS = (*RECURRENT_MODELS, *TREE_MODELS)  # Optuna study each
UNTRAINED_MODELS = ("persistence", "ridge")  # deterministic: one run, no tuning
CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


def is_residual(name: str) -> bool:
    """Predicts the change from persistence (recurrent or tree)."""
    return name.endswith("_res")


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
    """The net for a model name ("gru" and "gru_res" are both GRUs). A residual model's output
    layer starts at zero, so untrained it predicts exactly persistence and training only learns
    the correction."""
    cell = name.removesuffix("_res")
    model = RecurrentNet(cell, n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    if is_residual(name):
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    return model
