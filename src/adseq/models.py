"""The models, by name: persistence, ridge and their MLP (Schroer & Just's models), LightGBM on the
flattened window, and rnn / gru / lstm over the last L hours. *_res models predict the change
from persistence, which is then added back (the proposed Res-RNN / GRU / LSTM; lgbm_res is the
ablation: residual target without the recurrent network)."""

import copy

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
    pre-test minute (they used their whole training set), against the raw target in ft^3/min.
    float64: the lag features are collinear."""
    rows = np.flatnonzero(ds.split != "test")
    model = make_pipeline(StandardScaler(), RidgeCV(cv=TimeSeriesSplit(gap=RIDGE_CV_GAP)))
    model.fit(ds.X[rows].astype(float), ds.y_true[rows])
    ridge = model.named_steps["ridgecv"]
    info = {"alpha": float(ridge.alpha_), "n_params": ridge.coef_.size + 1}
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
    return model, {"epochs_run": mlp.n_iter_, "n_params": n_params}


# --- LightGBM ------------------------------------------------------------------------------

LGBM_MAX_ROUNDS = 5000
LGBM_PATIENCE = 100  # boosting rounds without a better val RMSE


def space_lgbm(trial: Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127]),
        "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [5, 10, 20, 50, 100]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "window": trial.suggest_categorical("window", list(WINDOWS)),
    }


def tree_input(ds: Dataset, idx: np.ndarray, window: int) -> np.ndarray:
    """The recurrent models' windows, flattened: one column per (hour, feature)."""
    return ds.windows(idx, window).reshape(len(idx), -1)


def fit_lgbm(residual: bool, hp: dict, ds: Dataset, seed: int) -> tuple[lgb.Booster, dict]:
    """Squared loss, early stopping on val RMSE; mlflow.lightgbm.autolog logs the val curve."""
    train, val = ds.part("train"), ds.part("val")
    y = ds.y - ds.base_level(residual)
    params = {k: v for k, v in hp.items() if k != "window"} | {  # hp: LightGBM's own names
        "objective": "l2",
        "metric": "rmse",
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    d_train = lgb.Dataset(tree_input(ds, train, hp["window"]), y[train], free_raw_data=True)
    d_val = d_train.create_valid(tree_input(ds, val, hp["window"]), y[val])
    booster = lgb.train(
        params,
        d_train,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[d_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False)],
    )

    best = booster.best_iteration
    trees = booster.dump_model(num_iteration=best)["tree_info"]
    info = {
        "best_round": best,
        "val_rmse_scaled": booster.best_score["val"]["rmse"],
        "n_params": sum(t["num_leaves"] for t in trees),  # one fitted value per leaf
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


class RecurrentNet(nn.Module):
    """A (batch, L, F) window, oldest hour first -> the last hidden state -> one number. A residual
    model's head starts at zero, so the untrained model predicts exactly persistence."""

    def __init__(self, cell: str, residual: bool, hp: dict, n_in: int):
        super().__init__()
        # torch's dropout only acts between stacked layers, so 1 layer must get 0
        self.rnn = CELLS[cell](
            n_in,
            hp["hidden_size"],
            hp["num_layers"],
            batch_first=True,
            dropout=hp["dropout"] if hp["num_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(hp["dropout"])  # before the head, for any number of layers
        self.head = nn.Linear(hp["hidden_size"], 1)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


def space_recurrent(trial: Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
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
) -> tuple[RecurrentNet, dict]:
    """AdamW on squared loss, gradient clipping, early stopping on val RMSE; keeps the best epoch's
    weights.
    Logs the learning curve (scaled units) to the active MLflow run after every epoch."""
    train_idx, val_idx = ds.part("train"), ds.part("val")
    y = torch.from_numpy((ds.y - ds.base_level(residual)).astype(np.float32))

    torch.manual_seed(seed)  # also seeds CUDA
    model = RecurrentNet(cell, residual, hp, ds.n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()
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
        mlflow.log_metrics(
            {"epoch_train_mse_scaled": train_loss, "epoch_val_rmse_scaled": rmse}, step=epoch
        )
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = rmse, copy.deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= train.patience:
            break

    model.load_state_dict(best_state)
    info = {
        "best_epoch": best_epoch,
        "val_rmse_scaled": best_rmse,
        "n_params": sum(p.numel() for p in model.parameters()),
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
) -> tuple[Fitted, dict]:
    residual = name.endswith("_res")
    if name == "persistence":
        return None, {"n_params": 0}
    if name == "ridge":
        return fit_ridge(ds)
    if name == "sj_mlp":
        return fit_sj_mlp(ds, seed)
    if name.startswith("lgbm"):
        return fit_lgbm(residual, hp, ds, seed)
    return fit_recurrent(name.removesuffix("_res"), residual, hp, ds, train, seed)


def predict(fitted: Fitted, name: ModelName, hp: dict, ds: Dataset) -> np.ndarray:
    """ft^3/min, for every sample."""
    residual = name.endswith("_res")
    if fitted is None:
        return ds.y_prev
    if isinstance(fitted, Pipeline):  # ridge, sj_mlp: trained on the raw target
        return np.asarray(fitted.predict(ds.X.astype(float)))
    if isinstance(fitted, lgb.Booster):
        return predict_lgbm(fitted, residual, hp, ds)
    return predict_recurrent(fitted, residual, hp, ds)


def space(trial: Trial, name: ModelName) -> dict:
    return space_lgbm(trial) if name.startswith("lgbm") else space_recurrent(trial)
