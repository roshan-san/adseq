"""The models: which ones exist, and the recurrent network. How each one is fitted: fit.py.

persistence   next-24-h mean = mean of the previous clock hour (Schroer & Just's baseline)
ridge         linear model on the 36 features of minute t, as in Schroer & Just's linear
              notebook: standardised, penalty chosen by TimeSeriesSplit CV on train + val
lgbm          LightGBM on the same window the recurrent models see, flattened (one column per
              hour and feature)
lgbm_res      LightGBM predicting the change from persistence (ablation: does the gain come from
              the residual target or from the recurrent network?)
rnn/gru/lstm  recurrent network over the last L hours -> its last hidden state -> one number
*_res         proposed: the same network, but it predicts the *change* from persistence, which
              is then added back, so it only has to learn a correction

Schroer & Just's own MLP is not here: it is run from their notebooks (scripts/run_schroer.py).
"""

import torch
from torch import nn

RECURRENT_MODELS = ("rnn", "gru", "lstm", "rnn_res", "gru_res", "lstm_res")  # *_res: proposed
TREE_MODELS = ("lgbm", "lgbm_res")
TUNED_MODELS = (*RECURRENT_MODELS, *TREE_MODELS)  # each gets an Optuna study
# no hyperparameters to tune and the same result for every seed: fitted once, with one seed
UNTUNED_MODELS = ("persistence", "ridge")
CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}  # cell name -> torch layer


def is_residual(name: str) -> bool:
    """True for the models that predict the change from persistence: the *_res nets and
    lgbm_res."""
    return name.endswith("_res")


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


def make_model(name: str, hp: dict, n_features: int) -> RecurrentNet:
    """The network for a model name: "gru" and "gru_res" are the same GRU network, they differ
    only in what they are trained to predict (fit.base_level). A residual model's head starts
    at zero, so the untrained model predicts exactly persistence; the head learns from the
    first step and the rest of the network from the second."""
    cell = name.removesuffix("_res")
    model = RecurrentNet(cell, n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    if is_residual(name):
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    return model
