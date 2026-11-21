"""Schroer & Just's own MLP, run from their notebooks by scripts/run_schroer.py: its test
predictions become an eval/sj_mlp run, so the report can put it next to ours."""

import numpy as np
import pandas as pd

from .config import SJ_PREDS
from .data import Dataset
from .metrics import regression_metrics
from .tracking import eval_run, log_preds, log_test_summary

SJ_INPUTS = 36  # inputs of their MLP, for its adjusted R2


def import_sj_mlp(ds: Dataset) -> None:
    """Schroer & Just's MLP test predictions (their notebook's, via scripts/run_schroer.py) as an
    eval/sj_mlp run laid out like ours: scored on our test samples, NaN elsewhere (their MLP is
    only scored on test). Checks first that their test minutes and labels are ours."""
    if not SJ_PREDS.exists():
        print(f"[eval] sj_mlp: {SJ_PREDS} missing, skipped (run scripts/run_schroer.py)")
        return
    sj = pd.read_csv(SJ_PREDS, index_col=0, parse_dates=True)
    te = ds.split == "test"
    if not ds.dates[te].equals(sj.index) or not np.allclose(sj["y_true"], ds.y_true[te]):
        raise ValueError("their MLP was scored on different test minutes or labels")
    pred = np.full(len(te), np.nan)
    pred[te] = sj["pred"].to_numpy()
    t = regression_metrics(ds.y_true[te], pred[te], SJ_INPUTS)
    with eval_run("sj_mlp"):
        log_test_summary([t])
        log_preds(ds, pred[None])
    print(f"[eval] sj_mlp      (their code)  test RMSE={t['rmse']:.4g}  R2={t['r2']:.3f}")
