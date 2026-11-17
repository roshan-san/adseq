"""Results: summary table (with Schroer & Just's own MLP), significance tests, test figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data import HORIZON
from .metrics import diebold_mariano, regression_metrics
from .models import RESIDUAL_MODELS

NAMES = {
    "sj_mlp": "MLP, Schroer & Just (their code)",
    "persistence": "Persistence",
    "ridge": "Ridge",
    "lgbm": "LightGBM",
    "lgbm_res": "Res-LightGBM",
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "rnn_res": "Res-RNN (proposed)",
    "gru_res": "Res-GRU (proposed)",
    "lstm_res": "Res-LSTM (proposed)",
}
BASELINES = (
    "sj_mlp",
    "persistence",
    "ridge",
    "lgbm",
    "lgbm_res",
    "rnn",
    "gru",
    "lstm",
)  # DM tests against these
SJ_PREDS = Path("runs/schroer_repro/schroer_mlp_test.csv")  # from scripts/run_schroer.py
SJ_INPUTS = 36  # inputs of their MLP, for its adjusted R2


def load_preds(cfg: Config, name: str) -> dict | None:
    """dates, split, y_true and preds (seeds x samples) of one model, or None if not run."""
    if name == "sj_mlp":
        return load_sj_mlp(cfg)
    f = cfg.run_dir / "preds" / f"{name}.npz"
    return dict(np.load(f)) if f.exists() else None


def load_sj_mlp(cfg: Config) -> dict | None:
    """Schroer & Just's MLP test predictions, laid out like ours: we borrow the samples of any
    model of ours (all are scored on the same ones) and fill in their predictions on the test
    minutes (NaN elsewhere; their MLP is only scored on test)."""
    ours = sorted((cfg.run_dir / "preds").glob("*.npz"))
    if not ours or not SJ_PREDS.exists():
        return None
    layout = dict(np.load(ours[0]))
    sj = pd.read_csv(SJ_PREDS, index_col=0, parse_dates=True)
    te = layout["split"] == "test"
    same_minutes = pd.to_datetime(layout["dates"][te]).equals(sj.index)
    if not same_minutes or not np.allclose(sj["y_true"], layout["y_true"][te]):
        raise ValueError("their MLP was scored on different test minutes or labels")
    pred = np.full(len(te), np.nan)
    pred[te] = sj["pred"].to_numpy()
    return {**layout, "preds": pred[None]}


def summary_table(cfg: Config, split: str = "test") -> pd.DataFrame:
    """Mean and std over seeds of every metric on one split, one row per model."""
    m = pd.concat(map(pd.read_csv, sorted((cfg.run_dir / "metrics").glob("*.csv"))))
    sj = load_preds(cfg, "sj_mlp")
    if sj is not None and split == "test":
        te = sj["split"] == "test"
        scores = regression_metrics(sj["y_true"][te], sj["preds"][0][te], SJ_INPUTS)
        m = pd.concat(
            [m, pd.DataFrame([{"model": "sj_mlp", "seed": 1, "split": "test", **scores}])]
        )
    g = m[m.split == split].groupby("model", sort=False)
    cols = ["adj_r2", "mape", "rmse", "mae", "teei", "r2"]
    s = g[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_seeds"] = g.seed.nunique()
    s = s.reset_index()
    order = {name: i for i, name in enumerate(NAMES)}
    return s.sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)


def significance(cfg: Config) -> pd.DataFrame:
    """Diebold-Mariano test of each proposed model vs. each baseline (seed-averaged preds)."""
    rows = []
    for name in RESIDUAL_MODELS:
        for base in BASELINES:
            a, b = load_preds(cfg, name), load_preds(cfg, base)
            if a is None or b is None:
                continue
            if not np.array_equal(a["dates"], b["dates"]):
                raise ValueError(f"{name} and {base} were scored on different samples")
            te = a["split"] == "test"
            y, pa, pb = a["y_true"][te], a["preds"].mean(0)[te], b["preds"].mean(0)[te]
            stat, p_two, p_better = diebold_mariano(y, pa, pb, max_lag=HORIZON)
            rows.append(
                {
                    "model": name,
                    "vs": base,
                    "DM": stat,
                    "p_two_sided": p_two,
                    "p_model_better": p_better,
                    # the test is on seed-averaged predictions, so show their RMSE too
                    "rmse_model_ens": float(np.sqrt(np.mean((pa - y) ** 2))),
                    "rmse_vs_ens": float(np.sqrt(np.mean((pb - y) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def plot_test_predictions(cfg: Config, models=("sj_mlp", "gru_res")) -> None:
    """Measured vs. predicted over the test period (single-column figure)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    (cfg.run_dir / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    measured = False
    for k, name in enumerate(models):
        p = load_preds(cfg, name)
        if p is None:
            continue
        te = p["split"] == "test"
        dates = pd.to_datetime(p["dates"][te])
        if not measured:
            ax.plot(dates, p["y_true"][te], "k-", lw=1, label="Measured")
            measured = True
        ax.plot(
            dates,
            p["preds"].mean(0)[te],
            lw=0.8,
            ls=["--", "-"][k % 2],
            label=NAMES.get(name, name),
        )
    ax.set_ylabel("Next-24-h biogas (ft$^3$/min)")
    ax.legend(frameon=False, fontsize=6)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(cfg.run_dir / "figures" / "test_predictions.pdf")
    plt.close(fig)


def make_report(cfg: Config) -> None:
    if not any((cfg.run_dir / "metrics").glob("*.csv")):
        raise SystemExit(f"no results in {cfg.run_dir}/metrics: run `adseq evaluate` first")
    missing = [n for n in NAMES if load_preds(cfg, n) is None]
    if missing:
        print(f"[report] not run, left out: {', '.join(missing)}")
    s = summary_table(cfg)
    dm = significance(cfg)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    dm.to_csv(cfg.run_dir / "significance.csv", index=False)
    plot_test_predictions(cfg)
    show = s[["model", "adj_r2_mean", "mape_mean", "rmse_mean", "teei_mean", "n_seeds"]]
    print(show.round(4).to_string(index=False))
    if len(dm):
        print("\nDiebold-Mariano (negative DM / small p_model_better => model beats baseline):")
        print(dm.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, significance.csv, figures/")
