"""The models, by name: persistence, ridge and their MLP (Schroer & Just's models), LightGBM on the
flattened window, and rnn / gru / lstm over the last L hours. *_res models predict the change
from persistence, which is then added back (the proposed Res-RNN / GRU / LSTM; lgbm_res is the
ablation: residual target without the recurrent network)."""

import copy
from collections.abc import Callable

import lightgbm as lgb
import mlflow
import numpy as np
import torch
from optuna.trial import Trial
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import ModelName, Training
from .data import WINDOWS, Dataset, chunks

# --- ridge ---------------------------------------------------------------------------------

RIDGE_CV_GAP = 300  # their notebook: rows left out between each CV fold's train and val rows


def fit_ridge(ds: Dataset) -> tuple[Pipeline, dict]:
    """As their SCADA-linear-regression notebook: RidgeCV over 5 TimeSeriesSplit folds, on every
    pre-test minute (they used their whole training set). float64: the lag features are collinear."""
    rows = np.flatnonzero(ds.split != "test")
    model = make_pipeline(StandardScaler(), RidgeCV(cv=TimeSeriesSplit(gap=RIDGE_CV_GAP)))
    model.fit(ds.X[rows].astype(float), ds.y[rows].astype(float))
    ridge = model.named_steps["ridgecv"]
    info = {"epochs": 0, "alpha": float(ridge.alpha_), "n_params": ridge.coef_.size + 1}
    return model, info


# --- Schroer & Just's MLP ------------------------------------------------------------------


def sj_mlp_pipeline(seed: int) -> Pipeline:
    """Their final SCADA MLP, settings verbatim (SCADA/SCADA-MLP.ipynb, cell 2)."""
    mlp = MLPRegressor(
        hidden_layer_sizes=(20,),
        solver="sgd",
        learning_rate="invscaling",
        alpha=155,
        max_iter=200,
        early_stopping=True,  # its own random 10 % of the rows, as theirs
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), mlp)


def fit_sj_mlp(ds: Dataset, seed: int) -> tuple[Pipeline, dict]:
    """On every pre-test minute, as ridge (their whole training set), against the raw target in
    ft^3/min: alpha=155 was tuned for that scale."""
    rows = np.flatnonzero(ds.split != "test")
    model = sj_mlp_pipeline(seed)
    model.fit(ds.X[rows].astype(float), ds.y_true[rows].astype(float))
    mlp = model.named_steps["mlpregressor"]
    n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    return model, {"epochs": mlp.n_iter_, "n_params": n_params}


# --- LightGBM ------------------------------------------------------------------------------

LGBM_MAX_ROUNDS = 5000
LGBM_PATIENCE = 100  # boosting rounds without a better val RMSE


def space_lgbm(trial: Trial) -> dict:
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
    """The recurrent models' windows, flattened: one column per (hour, feature)."""
    return ds.windows(idx, window).reshape(len(idx), -1)


def fit_lgbm(residual: bool, hp: dict, ds: Dataset, seed: int) -> tuple[lgb.Booster, dict]:
    """Not pruned while tuning: early stopping already ends a poor trial. mlflow.lightgbm.autolog
    logs its learning curve."""
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
    trees = booster.dump_model(num_iteration=best)["tree_info"]
    info = {
        "epochs": best,
        "val_rmse_scaled": evals["val"]["rmse"][best - 1],
        "n_params": sum(t["num_leaves"] for t in trees),  # one fitted value per leaf
        "device": "cpu",
    }
    return booster, info


def predict_lgbm(booster: lgb.Booster, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
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


# --- recurrent -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

# called after every epoch with (epoch, best val RMSE so far); returning True stops it (pruned)
OnStep = Callable[[int, float], bool]


class RecurrentNet(nn.Module):
    """A (batch, L, F) window, oldest hour first -> the last hidden state -> one number."""

    def __init__(self, cell: str, n_in: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # torch's dropout only acts between stacked layers, so 1 layer must get 0
        self.rnn = CELLS[cell](
            n_in,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)  # before the head, for any number of layers
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


def make_model(cell: str, residual: bool, hp: dict, n_features: int) -> RecurrentNet:
    """A residual model's head starts at zero, so the untrained model predicts exactly
    persistence."""
    model = RecurrentNet(cell, n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    if residual:
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    return model


def space_recurrent(trial: Trial) -> dict:
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
    return torch.from_numpy(ds.windows(idx, window)).to(DEVICE)


def fit_recurrent(
    cell: str,
    residual: bool,
    hp: dict,
    ds: Dataset,
    train: Training,
    seed: int,
    on_step: OnStep | None,
) -> tuple[RecurrentNet, dict]:
    """AdamW, gradient clipping, early stopping on val RMSE; keeps the best epoch's weights.
    Logs the learning curve (scaled units) to the active MLflow run after every epoch."""
    train_idx, val_idx = ds.part("train"), ds.part("val")
    y = torch.from_numpy((ds.y - ds.base_level(residual)).astype(np.float32))

    torch.manual_seed(seed)  # also seeds CUDA
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


def predict_recurrent(net: RecurrentNet, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        out = np.concatenate(
            [
                net(net_input(ds, c, hp["window"])).cpu().numpy().astype(float)
                for c in chunks(np.arange(len(ds.y)))
            ]
        )
    return ds.unscale(out + ds.base_level(residual))


# --- any model, by name --------------------------------------------------------------------

Fitted = RecurrentNet | lgb.Booster | Pipeline | None  # None: persistence (nothing is fitted)


def fit(
    name: ModelName,
    hp: dict,
    ds: Dataset,
    train: Training,
    seed: int,
    on_step: OnStep | None = None,
) -> tuple[Fitted, dict]:
    """on_step (recurrent models only) can stop the training early: Optuna pruning."""
    residual = name.endswith("_res")
    if name == "persistence":
        return None, {"epochs": 0, "n_params": 0}
    if name == "ridge":
        return fit_ridge(ds)
    if name == "sj_mlp":
        return fit_sj_mlp(ds, seed)
    if name.startswith("lgbm"):
        return fit_lgbm(residual, hp, ds, seed)
    return fit_recurrent(name.removesuffix("_res"), residual, hp, ds, train, seed, on_step)


def log_info(info: dict) -> None:
    """fit's info: numbers as metrics, strings (the device) as tags."""
    mlflow.log_metrics({k: v for k, v in info.items() if not isinstance(v, str)})
    mlflow.set_tags({k: v for k, v in info.items() if isinstance(v, str)})


def predict(fitted: Fitted, name: ModelName, hp: dict, ds: Dataset) -> np.ndarray:
    """ft^3/min, for every sample."""
    residual = name.endswith("_res")
    if fitted is None:
        return ds.y_prev
    if name == "sj_mlp" and isinstance(fitted, Pipeline):  # trained on the raw target
        return np.asarray(fitted.predict(ds.X.astype(float)))
    if isinstance(fitted, Pipeline):
        return ds.unscale(np.asarray(fitted.predict(ds.X.astype(float))))
    if isinstance(fitted, lgb.Booster):
        return predict_lgbm(fitted, residual, hp, ds)
    return predict_recurrent(fitted, residual, hp, ds)


def space(trial: Trial, name: ModelName) -> dict:
    return space_lgbm(trial) if name.startswith("lgbm") else space_recurrent(trial)
