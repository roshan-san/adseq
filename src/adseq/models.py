"""The models, by name: persistence, ridge and their MLP (Schroer & Just's models), LightGBM on the
flattened window, and rnn / gru / lstm over the last L hours. *_res models predict the change
from persistence, which is then added back (the proposed Res-RNN / GRU / LSTM; lgbm_res is the
ablation: residual target without the recurrent network).

Every model is fitted twice. fit(): on the train rows only, early stopping (recurrent, LightGBM) on
val, so its val score is honest for every model alike. refit(): on train + val, every pre-test
minute, for the epochs / boosting rounds fit() stopped at; ridge and their MLP re-run their own
model selection (RidgeCV, their early stopping), as in their notebooks. The test predictions come
from refit(): every model has seen the same, most recent data."""

import copy

import lightgbm as lgb
import mlflow
import numpy as np
import torch
from optuna.trial import Trial
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import EPOCH_WINDOWS, ModelName, Training
from .data import WINDOWS, Dataset, chunks

# --- ridge ---------------------------------------------------------------------------------

RIDGE_CV_GAP = 300  # their notebook: rows left out between each CV fold's train and val rows


def fit_ridge(ds: Dataset, rows: np.ndarray) -> tuple[Pipeline, dict]:
    """As their SCADA-linear-regression notebook: RidgeCV over 5 TimeSeriesSplit folds of the rows
    (time order), against the raw target in ft^3/min. float64: the lag features are collinear."""
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


def fit_sj_mlp(ds: Dataset, rows: np.ndarray, seed: int) -> tuple[Pipeline, dict]:
    """Against the raw target in ft^3/min: alpha=155 was tuned for that scale."""
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
    """The recurrent models' input, flattened: one column per (hour, sensor), then the static
    features."""
    return np.hstack([ds.windows(idx, window).reshape(len(idx), -1), ds.static[idx]])


def fit_lgbm(
    residual: bool, hp: dict, ds: Dataset, seed: int, rounds: int | None = None
) -> tuple[lgb.Booster, dict]:
    """Squared loss. rounds None: on train, early stopping on val RMSE (mlflow.lightgbm.autolog logs
    the val curve); else the refit: on train + val, exactly that many rounds."""
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
    if rounds is not None:
        rows = ds.pretest()
        d_all = lgb.Dataset(tree_input(ds, rows, hp["window"]), y[rows], free_raw_data=True)
        booster = lgb.train(params, d_all, num_boost_round=rounds)
        return booster, {"rounds": rounds}

    train, val = ds.part("train"), ds.part("val")
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
                booster.predict(  # best_iteration 0 (a refit): every round
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
    """A (batch, L, F_seq) window of hourly sensor means, oldest first -> the last hidden state,
    joined by the (batch, F_static) summarising features -> one number. A residual model's head
    starts at zero, so the untrained model predicts exactly persistence."""

    def __init__(self, cell: str, residual: bool, hp: dict, n_seq: int, n_static: int):
        super().__init__()
        # torch's dropout only acts between stacked layers, so 1 layer must get 0
        self.rnn = CELLS[cell](
            n_seq,
            hp["hidden_size"],
            hp["num_layers"],
            batch_first=True,
            dropout=hp["dropout"] if hp["num_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(hp["dropout"])  # on the hidden state, for any number of layers
        self.head = nn.Linear(hp["hidden_size"] + n_static, 1)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(seq)
        return self.head(torch.cat([self.dropout(out[:, -1]), static], dim=-1)).squeeze(-1)


def space_recurrent(trial: Trial) -> dict:
    """Widened where protocol-1 tuning's best trials sat at an edge: lr and dropout (the residual
    models), hidden_size and batch_size (the plain RNN)."""
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "window": trial.suggest_categorical("window", list(WINDOWS)),
    }


def net_input(ds: Dataset, idx: np.ndarray, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(ds.windows(idx, window)).to(DEVICE),
        torch.from_numpy(ds.static[idx]).to(DEVICE),
    )


def fit_recurrent(
    cell: str,
    residual: bool,
    hp: dict,
    ds: Dataset,
    train: Training,
    seed: int,
    epochs: int | None = None,
) -> tuple[RecurrentNet, dict]:
    """AdamW on squared loss, gradient clipping. epochs None: on train, early stopping on val RMSE,
    keeping the best epoch's weights; else the refit: on train + val, exactly that many epochs.
    An epoch is EPOCH_WINDOWS training windows drawn at random, so epochs are the same size at
    any stride and an epoch count carries over from tune to evaluate and to the refit.
    Logs the learning curve to the active MLflow run, step = epoch (step 0 = untrained)."""
    refit = epochs is not None
    train_idx = ds.pretest() if refit else ds.part("train")
    y = torch.from_numpy((ds.y - ds.base_level(residual)).astype(np.float32))

    torch.manual_seed(seed)  # also seeds CUDA
    model = RecurrentNet(cell, residual, hp, ds.n_seq, ds.n_static).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())

    def train_epoch() -> float:
        model.train()
        draw = torch.randperm(len(train_idx))[:EPOCH_WINDOWS].numpy()
        order = train_idx[draw]
        loss_sum = 0.0
        for i in range(0, len(order), hp["batch_size"]):
            batch = order[i : i + hp["batch_size"]]
            loss = loss_fn(model(*net_input(ds, batch, hp["window"])), y[batch].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(batch)
        return loss_sum / len(order)

    if refit:
        for epoch in range(1, epochs + 1):
            mlflow.log_metrics({"epoch_train_mse_scaled": train_epoch()}, step=epoch)
        return model, {"epochs": epochs, "n_params": n_params}

    val_idx = ds.part("val")
    val_chunks = chunks(val_idx)  # the ~77k validation windows would take ~0.8 GB at once
    val_base, val_true = ds.base_level(residual)[val_idx], ds.y_true[val_idx]

    def val_scores() -> dict[str, float]:
        """RMSE in scaled units (early stopping), MAE and MAPE in ft^3/min and % (to watch, set
        against the baselines' val_mae / val_mape)."""
        model.eval()
        sq, preds = 0.0, []
        with torch.no_grad():
            for c in val_chunks:
                pred = model(*net_input(ds, c, hp["window"]))
                sq += (pred - y[c].to(DEVICE)).pow(2).sum().item()
                preds.append(pred.cpu().numpy().astype(float))
        pred_ft = ds.unscale(np.concatenate(preds) + val_base)
        return {
            "epoch_val_rmse_scaled": (sq / len(val_idx)) ** 0.5,
            "epoch_val_mae": float(mean_absolute_error(val_true, pred_ft)),
            "epoch_val_mape": 100 * float(mean_absolute_percentage_error(val_true, pred_ft)),
        }

    # epoch 0 = untrained weights, so there is always a state to restore; for a residual model
    # that is exactly persistence, so step 0 of the curve is the baseline to beat
    scores = val_scores()
    mlflow.log_metrics(scores, step=0)
    best_rmse, best_state, best_epoch = (
        scores["epoch_val_rmse_scaled"],
        copy.deepcopy(model.state_dict()),
        0,
    )
    for epoch in range(1, train.max_epochs + 1):
        train_loss = train_epoch()
        scores = val_scores()
        mlflow.log_metrics({"epoch_train_mse_scaled": train_loss, **scores}, step=epoch)
        rmse = scores["epoch_val_rmse_scaled"]
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = rmse, copy.deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= train.patience:
            break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "val_rmse_scaled": best_rmse, "n_params": n_params}


def predict_recurrent(net: RecurrentNet, residual: bool, hp: dict, ds: Dataset) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        out = np.concatenate(
            [
                net(*net_input(ds, c, hp["window"])).cpu().numpy().astype(float)
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
    """On the train rows; val only for early stopping. What tune scores and evaluate's val
    metrics come from."""
    residual = name.endswith("_res")
    if name == "persistence":
        return None, {"n_params": 0}
    if name == "ridge":
        return fit_ridge(ds, ds.part("train"))
    if name == "sj_mlp":
        return fit_sj_mlp(ds, ds.part("train"), seed)
    if name.startswith("lgbm"):
        return fit_lgbm(residual, hp, ds, seed)
    return fit_recurrent(name.removesuffix("_res"), residual, hp, ds, train, seed)


def refit(
    name: ModelName,
    hp: dict,
    ds: Dataset,
    train: Training,
    seed: int,
    info: dict,
) -> tuple[Fitted, dict]:
    """On train + val, for the epochs / rounds fit() (its info) stopped at. What evaluate's test
    metrics come from."""
    residual = name.endswith("_res")
    if name == "persistence":
        return None, {"n_params": 0}
    if name == "ridge":
        return fit_ridge(ds, ds.pretest())
    if name == "sj_mlp":
        return fit_sj_mlp(ds, ds.pretest(), seed)
    if name.startswith("lgbm"):
        return fit_lgbm(residual, hp, ds, seed, rounds=int(info["best_round"]))
    cell = name.removesuffix("_res")
    return fit_recurrent(cell, residual, hp, ds, train, seed, epochs=int(info["best_epoch"]))


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
