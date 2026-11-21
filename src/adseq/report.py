"""runs/<name>/summary.csv and figures/: views for the paper, rebuilt from MLflow at any time."""

import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts

from .config import RUNS, Config
from .run import regression_metrics

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
SJ_PREDS = RUNS / "schroer_repro" / "schroer_mlp_test.csv"  # from scripts/run_schroer.py
SJ_INPUTS = 36  # inputs of their MLP, for its adjusted R2


def load_preds(run_id: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = download_artifacts(run_id=run_id, artifact_path="preds.npz", dst_path=d)
        with np.load(path) as f:
            return dict(f)


def latest_evals(cfg: Config) -> pd.DataFrame:
    """The newest FINISHED eval/<model> run of each model."""
    runs = mlflow.search_runs(
        experiment_names=[cfg.name],
        filter_string="tags.mlflow.runName LIKE 'eval/%' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return runs
    runs["model"] = runs["tags.mlflow.runName"].str.removeprefix("eval/")
    return runs.drop_duplicates("model")


def load_sj_mlp(ours: dict) -> np.ndarray | None:
    """Their MLP's test predictions, checked against our test minutes and labels (ours: any eval
    run's preds)."""
    if not SJ_PREDS.exists():
        print(f"[report] sj_mlp: {SJ_PREDS} missing, left out (run scripts/run_schroer.py)")
        return None
    sj = pd.read_csv(SJ_PREDS, index_col=0, parse_dates=True)
    te = ours["split"] == "test"
    same = pd.to_datetime(ours["dates"][te]).equals(sj.index)
    if not same or not np.allclose(sj["y_true"], ours["y_true"][te]):
        raise ValueError("their MLP was scored on different test minutes or labels")
    return sj["pred"].to_numpy()


def summary_table(runs: pd.DataFrame, extra: list[dict]) -> pd.DataFrame:
    """One row per model, in NAMES order; extra: rows already in that shape (their MLP)."""
    stats = [f"metrics.test_{m}_{stat}" for m in METRICS for stat in ("mean", "std")]
    cols = ["model", *stats, "metrics.n_seeds", "run_id"]
    s = runs.filter(items=cols).rename(
        columns={c: c.split(".")[-1].removeprefix("test_") for c in cols}
    )
    s = pd.concat([pd.DataFrame(extra, columns=s.columns), s]).astype({"n_seeds": int})
    order = {name: i for i, name in enumerate(NAMES)}
    return s.sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)


def plot_test_predictions(cfg: Config, ours: dict, curves: dict[str, np.ndarray]) -> None:
    """Measured vs. predicted over the test period, single-column. curves: model -> test preds."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    te = ours["split"] == "test"
    dates = pd.to_datetime(ours["dates"][te])
    ax.plot(dates, ours["y_true"][te], "k-", lw=1, label="Measured")
    for k, (name, pred) in enumerate(curves.items()):
        ax.plot(dates, pred, lw=0.8, ls=["--", "-"][k % 2], label=NAMES[name])

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
    ours = load_preds(run_ids.get("gru_res", runs["run_id"].iloc[0]))
    te = ours["split"] == "test"
    extra, curves = [], {}
    sj = load_sj_mlp(ours)
    if sj is not None:
        t = regression_metrics(ours["y_true"][te], sj, SJ_INPUTS)
        means = {f"{m}_mean": t[m] for m in METRICS}
        stds = {f"{m}_std": float("nan") for m in METRICS}  # one set of predictions
        extra.append({"model": "sj_mlp", **means, **stds, "n_seeds": 1, "run_id": ""})
        curves["sj_mlp"] = sj
    if "gru_res" in run_ids:
        curves["gru_res"] = ours["preds"].mean(0)[te]  # mean over seeds
    missing = [name for name in NAMES if name not in run_ids and name != "sj_mlp"]
    if missing:
        print(f"[report] not run, left out: {', '.join(missing)}")

    s = summary_table(runs, extra)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    plot_test_predictions(cfg, ours, curves)
    show = s[["model", "adj_r2_mean", "mape_mean", "rmse_mean", "teei_mean", "n_seeds"]]
    print(show.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, figures/")
