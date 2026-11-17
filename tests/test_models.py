"""Model checks that need no data."""

import numpy as np
import pytest
import torch

from adseq.models import RECURRENT_MODELS, RESIDUAL_MODELS, fit_ridge, make_model

HP = {"hidden_size": 8, "num_layers": 2, "dropout": 0.1}


@pytest.mark.parametrize("name", RECURRENT_MODELS)
def test_every_recurrent_model_maps_a_window_to_one_number(name):
    out = make_model(name, HP, n_features=5)(torch.randn(3, 12, 5))
    assert out.shape == (3,) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", RESIDUAL_MODELS)
def test_untrained_residual_model_predicts_no_change(name):
    model = make_model(name, HP, n_features=5).eval()
    assert (model(torch.rand(3, 12, 5)) == 0).all()


def test_ridge_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    w_true = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ w_true + 3.0
    w, b, alpha, rmse = fit_ridge(X[:400], y[:400], X[400:], y[400:])
    assert alpha == pytest.approx(1e-4)  # noise-free: the weakest penalty wins
    assert np.allclose(w, w_true, atol=1e-3) and b == pytest.approx(3.0, abs=1e-3)
    assert rmse < 1e-3
