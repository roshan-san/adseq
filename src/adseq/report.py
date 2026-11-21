"""Results: summary table (with Schroer & Just's own MLP) and test figure, from the newest
FINISHED eval/<model> MLflow run of each model. runs/<name>/summary.csv and figures/ are views
for the paper, regenerated from MLflow at any time; summary.csv names each row's run_id.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import Config
from .models import MODELS
from .tracking import latest_evals, load_preds

METRICS = ["adj_r2", "mape", "rmse", "mae", "teei", "r2"]


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds of every test metric, one row per model, in MODELS order, with
    the eval run each row comes from."""
    stats = [f"metrics.test_{m}_{stat}" for m in METRICS for stat in ("mean", "std")]
    cols = ["tags.model", *stats, "metrics.n_seeds", "run_id"]
    s = runs.filter(items=cols).rename(
        columns={c: c.split(".")[-1].removeprefix("test_") for c in cols}
    )
    s = s.astype({"n_seeds": int})
    order = {name: i for i, name in enumerate(MODELS)}
    return s.sort_values("model", key=lambda c: c.map(order)).reset_index(drop=True)


def plot_test_predictions(
    cfg: Config, run_ids: dict[str, str], models=("sj_mlp", "gru_res")
) -> None:
    """Measured vs. predicted (mean over seeds) over the test period (single-column figure).
    run_ids: model -> its eval run."""
    loaded = {name: load_preds(run_ids[name]) for name in models if name in run_ids}
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
        ax.plot(dates, p["preds"].mean(0)[te], lw=0.8, ls=line, label=MODELS[name].label)

    ax.set_ylabel("Next-24-h biogas (ft$^3$/min)")
    ax.legend(frameon=False, fontsize=6)
    fig.autofmt_xdate()
    fig.tight_layout()
    (cfg.run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(cfg.run_dir / "figures" / "test_predictions.pdf")
    plt.close(fig)


def make_report(cfg: Config) -> None:
    """Table and figure from the newest FINISHED eval/<model> run of each model."""
    runs = latest_evals(cfg)
    if runs.empty:
        raise SystemExit(
            f"no finished evaluations in MLflow experiment {cfg.name}: run `adseq evaluate` first"
        )
    missing = [name for name in MODELS if name not in set(runs["tags.model"])]
    if missing:
        print(f"[report] not run, left out: {', '.join(missing)}")

    s = summary_table(runs)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    plot_test_predictions(cfg, dict(zip(s.model, s.run_id, strict=True)))
    show = s[["model", "adj_r2_mean", "mape_mean", "rmse_mean", "teei_mean", "n_seeds"]]
    print(show.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, figures/")
