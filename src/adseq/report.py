import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts

from .config import Config

NAMES = {
    "sj_mlp": "MLP, Schroer & Just (retrained)",
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
PLOTTED = ("sj_mlp", "gru_res")


def load_preds(run_id: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = download_artifacts(run_id=run_id, artifact_path="preds.npz", dst_path=d)
        with np.load(path) as f:
            return dict(f)


def latest_evals(cfg: Config) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string="tags.mlflow.runName LIKE 'eval/%' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return runs
    runs["model"] = runs["tags.mlflow.runName"].str.removeprefix("eval/")
    return runs.drop_duplicates("model")


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    stats = [
        f"metrics.{split}_{m}_{stat}"
        for split in ("val", "test")
        for m in METRICS
        for stat in ("mean", "std")
    ]
    cols = ["model", *stats, "metrics.n_seeds", "run_id"]
    s = runs.filter(items=cols).rename(columns={c: c.removeprefix("metrics.") for c in cols})
    s = s.astype({"n_seeds": int})
    order = {name: i for i, name in enumerate(NAMES)}
    return s.sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)


def plot_test_predictions(cfg: Config, run_ids: dict[str, str]) -> None:
    shown = {name: load_preds(run_ids[name]) for name in PLOTTED if name in run_ids}
    ref = next(iter(shown.values())) if shown else load_preds(next(iter(run_ids.values())))
    te = ref["split"] == "test"
    dates = pd.to_datetime(ref["dates"][te])

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(dates, ref["y_true"][te], "k-", lw=1, label="Measured")
    for k, (name, p) in enumerate(shown.items()):
        ax.plot(dates, p["preds"].mean(0)[te], lw=0.8, ls=["--", "-"][k % 2], label=NAMES[name])

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
    run_ids = dict(zip(runs["model"], runs["run_id"], strict=True))
    missing = [name for name in NAMES if name not in run_ids]
    if missing:
        print(f"[report] not run, left out: {', '.join(missing)}")

    s = summary_table(runs)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    plot_test_predictions(cfg, run_ids)
    shown = [
        "val_rmse_mean",
        "val_mape_mean",
        "test_adj_r2_mean",
        "test_mape_mean",
        "test_rmse_mean",
    ]
    show = s.filter(items=["model", *shown, "test_teei_mean", "n_seeds"])
    print(show.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, figures/")
