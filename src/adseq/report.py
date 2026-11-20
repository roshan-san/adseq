"""Results: summary table (with Schroer & Just's own MLP) and test figure, from the newest
FINISHED eval/<model> MLflow run of each model. runs/<name>/summary.csv and figures/ are views
for the paper, regenerated from MLflow at any time; summary.csv names each row's run_id.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import Config
from .tracking import latest_evals, load_preds

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
METRICS = ["adj_r2", "mape", "rmse", "mae", "teei", "r2"]


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds of every test metric, one row per model, in NAMES order, with
    the eval run each row comes from."""
    stats = [f"metrics.test_{m}_{stat}" for m in METRICS for stat in ("mean", "std")]
    s = runs[["tags.model", *stats, "metrics.n_seeds", "run_id"]]
    s = s.rename(columns=lambda c: c.split(".")[-1].removeprefix("test_"))
    s = s.astype({"n_seeds": int})
    order = {name: i for i, name in enumerate(NAMES)}
    return s.sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)


def plot_test_predictions(cfg: Config, runs: dict[str, str], models=("sj_mlp", "gru_res")) -> None:
    """Measured vs. predicted (mean over seeds) over the test period (single-column figure).
    runs: model -> eval run_id."""
    loaded = {name: load_preds(runs[name]) for name in models if name in runs}
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
    runs = latest_evals(cfg)
    if runs.empty:
        raise SystemExit(
            f"no finished evaluations in MLflow experiment {cfg.name}: run `adseq evaluate` first"
        )
    missing = [name for name in NAMES if name not in set(runs["tags.model"])]
    if missing:
        print(f"[report] not run, left out: {', '.join(missing)}")

    s = summary_table(runs)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    plot_test_predictions(cfg, dict(zip(s.model, s.run_id, strict=True)))
    show = s[["model", "adj_r2_mean", "mape_mean", "rmse_mean", "teei_mean", "n_seeds"]]
    print(show.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, figures/")
