"""Results: summary table (with Schroer & Just's own MLP) and test figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .metrics import regression_metrics

NAMES = {  # table / legend label of every model, in table order
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
SJ_PREDS = Path("runs/schroer_repro/schroer_mlp_test.csv")  # from scripts/run_schroer.py
SJ_INPUTS = 36  # inputs of their MLP, for its adjusted R2
METRICS = ["adj_r2", "mape", "rmse", "mae", "teei", "r2"]


def preds_file(cfg: Config, name: str) -> Path:
    return cfg.run_dir / "preds" / f"{name}.npz"


def has_preds(cfg: Config, name: str) -> bool:
    return SJ_PREDS.exists() if name == "sj_mlp" else preds_file(cfg, name).exists()


def load_preds(cfg: Config, name: str) -> dict | None:
    """dates, split, y_true and preds (seeds x samples) of one model, or None if not run."""
    if name == "sj_mlp":
        return load_sj_mlp(cfg)
    f = preds_file(cfg, name)
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


def test_metrics(cfg: Config) -> pd.DataFrame:
    """One row per (model, seed) on the test split: our models' saved metrics, plus Schroer &
    Just's MLP scored here."""
    ours = pd.concat(map(pd.read_csv, sorted((cfg.run_dir / "metrics").glob("*.csv"))))
    rows = [ours[ours.split == "test"]]
    sj = load_sj_mlp(cfg)
    if sj is not None:
        te = sj["split"] == "test"
        scores = regression_metrics(sj["y_true"][te], sj["preds"][0][te], SJ_INPUTS)
        rows.append(pd.DataFrame([{"model": "sj_mlp", "seed": 1, "split": "test", **scores}]))
    return pd.concat(rows)


def summary_table(cfg: Config) -> pd.DataFrame:
    """Mean and std over seeds of every test metric, one row per model, in NAMES order."""
    g = test_metrics(cfg).groupby("model", sort=False)
    s = g[METRICS].agg(["mean", "std"])
    s.columns = [f"{metric}_{stat}" for metric, stat in s.columns]
    s["n_seeds"] = g.seed.nunique()
    s = s.reset_index()
    order = {name: i for i, name in enumerate(NAMES)}
    return s.sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)


def plot_test_predictions(cfg: Config, models=("sj_mlp", "gru_res")) -> None:
    """Measured vs. predicted (mean over seeds) over the test period (single-column figure)."""
    loaded = {name: p for name in models if (p := load_preds(cfg, name)) is not None}
    if not loaded:
        return
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    first = next(iter(loaded.values()))  # every model has the same test minutes and labels
    te = first["split"] == "test"
    dates = pd.to_datetime(first["dates"][te])
    ax.plot(dates, first["y_true"][te], "k-", lw=1, label="Measured")
    for k, (name, p) in enumerate(loaded.items()):
        line = ["--", "-"][k % 2]
        ax.plot(dates, p["preds"].mean(0)[te], lw=0.8, ls=line, label=NAMES.get(name, name))

    ax.set_ylabel("Next-24-h biogas (ft$^3$/min)")
    ax.legend(frameon=False, fontsize=6)
    fig.autofmt_xdate()
    fig.tight_layout()
    (cfg.run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(cfg.run_dir / "figures" / "test_predictions.pdf")
    plt.close(fig)


def make_report(cfg: Config) -> None:
    if not any((cfg.run_dir / "metrics").glob("*.csv")):
        raise SystemExit(f"no results in {cfg.run_dir}/metrics: run `adseq evaluate` first")
    missing = [name for name in NAMES if not has_preds(cfg, name)]
    if missing:
        print(f"[report] not run, left out: {', '.join(missing)}")

    s = summary_table(cfg)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    plot_test_predictions(cfg)
    show = s[["model", "adj_r2_mean", "mape_mean", "rmse_mean", "teei_mean", "n_seeds"]]
    print(show.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, figures/")
