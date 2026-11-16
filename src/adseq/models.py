"""Paper Sec. IV-B: models.

fnn_paper   base paper's network, fixed: 128-64-1 ReLU, Adam lr=1e-3, min-max inputs (Long et al., 2026)
fnn_tuned   same idea, hyper-parameters tuned with the same budget as the RNNs
window_fnn  FNN on the flattened L-day window (ablation: same inputs as RNNs, no recurrence)
rnn/gru/lstm  recurrent encoder over the L-day window -> last hidden state -> linear
"""

import torch
from torch import nn

# "scaling": the base paper min-max scales to [0, 1]; every other model uses --scaling
PAPER_HP = {
    "width": 128,  # 128-64-1
    "n_layers": 2,
    "dropout": 0.0,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 32,
    "scaling": "minmax",
}
RECURRENT_MODELS = ("rnn", "gru", "lstm")
WINDOW_MODELS = ("window_fnn", *RECURRENT_MODELS)  # use the (N, L, F) input
TUNED_MODELS = ("fnn_tuned", *WINDOW_MODELS)  # searched with W&B Sweeps


class FNN(nn.Module):
    """n_layers hidden layers, each half as wide as the one before: width=128, n_layers=2 -> 128-64."""

    def __init__(self, n_in: int, width: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for h in (width // 2**i for i in range(n_layers)):
            layers += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(dropout)]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)  # flatten(1) handles window inputs


class RecurrentNet(nn.Module):
    def __init__(self, cell: str, n_in: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        rnn_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_class(
            n_in,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # out: (batch, L, hidden)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


def make_model(name: str, hp: dict, n_features: int, window: int) -> nn.Module:
    if name in RECURRENT_MODELS:
        return RecurrentNet(name, n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    n_in = n_features * window if name == "window_fnn" else n_features
    return FNN(n_in, hp["width"], hp["n_layers"], hp["dropout"])
