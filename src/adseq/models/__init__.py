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
              test predictions imported by evaluate

Each family's fit, predict and search space are in its own file (baselines, lgbm, recurrent).
fit is always called inside an MLflow run (a tuning trial or an evaluation seed) and logs the
learning curve into it (epoch_train_loss / epoch_val_rmse, scaled units).
"""

from dataclasses import dataclass
from typing import Literal, get_args

import lightgbm as lgb
import numpy as np
from optuna.trial import Trial
from sklearn.pipeline import Pipeline

from ..config import ModelName, Training
from ..data import Dataset
from . import baselines, lgbm, recurrent
from .recurrent import OnStep, RecurrentNet

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
            return baselines.fit_ridge(ds)
        case "lgbm":
            return lgbm.fit(spec.residual, hp, ds, seed)
        case "recurrent":
            cell = name.removesuffix("_res")
            return recurrent.fit(cell, spec.residual, hp, ds, train, seed, on_step)
        case "external":
            raise ValueError(f"{name} is not trained here")


def predict(fitted: Fitted, name: ModelName, hp: dict, ds: Dataset) -> np.ndarray:
    """Predictions (ft^3/min) for every sample."""
    spec = MODELS[name]
    match spec.family:
        case "persistence":
            return baselines.predict_persistence(ds)
        case "ridge":
            assert isinstance(fitted, Pipeline)
            return baselines.predict_ridge(fitted, ds)
        case "lgbm":
            assert isinstance(fitted, lgb.Booster)
            return lgbm.predict(fitted, spec.residual, hp, ds)
        case "recurrent":
            assert isinstance(fitted, RecurrentNet)
            return recurrent.predict(fitted, spec.residual, hp, ds)
        case "external":
            raise ValueError(f"{name} is not predicted here")


def space(trial: Trial, name: ModelName) -> dict:
    """The model's hyper-parameter search space."""
    match MODELS[name].family:
        case "lgbm":
            return lgbm.space(trial)
        case "recurrent":
            return recurrent.space(trial)
        case _:
            raise ValueError(f"{name} is not tuned")
