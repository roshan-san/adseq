"""The models: one row each in MODELS, and fit / predict / space for any of them.

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
sj_mlp        Schroer & Just's own MLP: run from their notebooks (scripts/run_schroer.py), its
              test predictions imported by run.evaluate

fit is always called inside an MLflow run (a tuning trial or an evaluation seed), which gets the
learning curve: logged by the recurrent loop after every epoch (epoch_train_loss /
epoch_val_rmse, scaled units), and by mlflow.lightgbm.autolog for LightGBM.

Every recurrent model shares one procedure: AdamW (weight decay applied to the weights directly,
not through the gradient), MSE or L1 loss (a tuned choice), mini-batches, gradient clipping,
early stopping on validation RMSE; the weights from the best validation epoch are kept. They
train on the GPU when there is one (`uv sync --extra cu132`), else on the CPU. LightGBM is
boosted with the same loss choice and early stopping on the same validation RMSE (one "epoch" =
one tree), always on the CPU.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import lightgbm as lgb
import mlflow
import numpy as np
import torch
from optuna.trial import Trial
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import ModelName, Training
from .data import WINDOWS, Dataset, chunks

# --- table ---------------------------------------------------------------------------------

Family = Literal["persistence", "ridge", "lgbm", "recurrent", "external"]


@dataclass(frozen=True)
class Spec:
    label: str  # table / legend label
    family: Family
    residual: bool = False  # predicts the change from persistence

    @property
    def tuned(self) -> bool:
        """Gets an Optuna study."""
        return self.family in ("lgbm", "recurrent")

    @property
    def seeded(self) -> bool:
        """Evaluated with every seed; the others give the same result for any seed, so one."""
        return self.tuned


MODELS: dict[ModelName, Spec] = {  # in table order
    "sj_mlp": Spec("MLP, Schroer & Just (their code)", "external"),
    "persistence": Spec("Persistence", "persistence"),
    "ridge": Spec("Ridge", "ridge"),
    "lgbm": Spec("LightGBM", "lgbm"),
    "lgbm_res": Spec("Res-LightGBM", "lgbm", residual=True),
    "rnn": Spec("RNN", "recurrent"),
    "gru": Spec("GRU", "recurrent"),
    "lstm": Spec("LSTM", "recurrent"),
    "rnn_res": Spec("Res-RNN (proposed)", "recurrent", residual=True),
    "gru_res": Spec("Res-GRU (proposed)", "recurrent", residual=True),
    "lstm_res": Spec("Res-LSTM (proposed)", "recurrent", residual=True),
}
if set(MODELS) != set(get_args(ModelName)):
    raise RuntimeError("MODELS and config.ModelName must list the same models")


# --- baselines -----------------------------------------------------------------------------

RIDGE_CV_GAP = 300  # their notebook: rows left out between each CV fold's train and val rows


def fit_ridge(ds: Dataset) -> tuple[Pipeline, dict]:
    """Their linear model (SCADA-linear-regression.ipynb): StandardScaler + RidgeCV with its
    default penalties (0.1, 1, 10); the one with the best R2 over 5 TimeSeriesSplit folds is
    refit on all rows. Fit on every pre-test row (train + val, as theirs saw their whole training
    set), so no val split is needed; every minute, whatever the stride. float64, as the lag
    features are collinear."""
    rows = np.flatnonzero(ds.split != "test")
    model = make_pipeline(StandardScaler(), RidgeCV(cv=TimeSeriesSplit(gap=RIDGE_CV_GAP)))
    model.fit(ds.X[rows].astype(float), ds.y[rows].astype(float))
    ridge = model.named_steps["ridgecv"]
    info = {"epochs": 0, "alpha": float(ridge.alpha_), "n_params": ridge.coef_.size + 1}
    return model, info


def predict_ridge(model: Pipeline, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample."""
    return ds.unscale(np.asarray(model.predict(ds.X.astype(float))))


def predict_persistence(ds: Dataset) -> np.ndarray:
    """The mean of the previous clock hour (ft^3/min), for every sample."""
    return ds.y_prev


# --- LightGBM ------------------------------------------------------------------------------

LGBM_MAX_ROUNDS = 5000
LGBM_PATIENCE = 100  # boosting rounds without a better val RMSE


def space_lgbm(trial: Trial) -> dict:
    """Search space: LightGBM's own, with the same loss and window choices as the recurrent
    models."""
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
    """(len(idx), L * F): the recurrent models' windows flattened, one column per (hour,
    feature)."""
    return ds.windows(idx, window).reshape(len(idx), -1)


def fit_lgbm(residual: bool, hp: dict, ds: Dataset, seed: int) -> tuple[lgb.Booster, dict]:
    """Boost on the training samples; keep the trees up to the best validation round. Not
    pruned while tuning: early stopping already ends a poor trial."""
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
    val_rmse = evals["val"]["rmse"]
    trees = booster.dump_model(num_iteration=best)["tree_info"]
    info = {
        "epochs": best,
        "val_rmse_scaled": val_rmse[best - 1],
        "n_params": sum(t["num_leaves"] for t in trees),  # one fitted value per leaf
        "device": "cpu",
    }
    return booster, info


def predict_lgbm(booster: lgb.Booster, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample, CHUNK samples at a time."""
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


def space_recurrent(trial: Trial) -> dict:
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


def fit_recurrent(
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


def predict_recurrent(net: RecurrentNet, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
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


# --- dispatch ------------------------------------------------------------------------------

Fitted = RecurrentNet | lgb.Booster | Pipeline | None  # None: persistence (nothing is fitted)


def fit(
    name: ModelName,
    hp: dict,
    ds: Dataset,
    train: Training,
    seed: int,
    on_step: OnStep | None = None,
) -> tuple[Fitted, dict]:
    """The fitted model and training info. on_step (recurrent models only) can stop the
    training early: Optuna pruning."""
    spec = MODELS[name]
    match spec.family:
        case "persistence":
            return None, {"epochs": 0, "n_params": 0}
        case "ridge":
            return fit_ridge(ds)
        case "lgbm":
            return fit_lgbm(spec.residual, hp, ds, seed)
        case "recurrent":
            cell = name.removesuffix("_res")
            return fit_recurrent(cell, spec.residual, hp, ds, train, seed, on_step)
        case "external":
            raise ValueError(f"{name} is not trained here")


def predict(fitted: Fitted, name: ModelName, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample."""
    spec = MODELS[name]
    match spec.family:
        case "persistence":
            return predict_persistence(ds)
        case "ridge":
            assert isinstance(fitted, Pipeline)
            return predict_ridge(fitted, ds)
        case "lgbm":
            assert isinstance(fitted, lgb.Booster)
            return predict_lgbm(fitted, spec.residual, hp, ds)
        case "recurrent":
            assert isinstance(fitted, RecurrentNet)
            return predict_recurrent(fitted, spec.residual, hp, ds)
        case "external":
            raise ValueError(f"{name} is not predicted here")


def space(trial: Trial, name: ModelName) -> dict:
    """The model's hyper-parameter search space."""
    match MODELS[name].family:
        case "lgbm":
            return space_lgbm(trial)
        case "recurrent":
            return space_recurrent(trial)
        case _:
            raise ValueError(f"{name} is not tuned")
