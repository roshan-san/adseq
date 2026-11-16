"""Paper Sec. V: results table (LaTeX, IEEE style), significance tests, prediction figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data import TARGET
from .metrics import diebold_mariano
from .models import RECURRENT_MODELS

NAMES = {
    "persistence": "Persistence",
    "fnn_paper": "Lag-FNN (base paper)",
    "fnn_tuned": "Lag-FNN (tuned)",
    "window_fnn": "Window-FNN",
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
}
UNIT = "Biogas (ft$^3$/min)"
METRICS = ("rmse", "mae", "r2", "acc10")


def load_preds(cfg: Config, name: str) -> dict | None:
    f = cfg.run_dir / "preds" / f"{TARGET}__{name}.npz"
    return dict(np.load(f)) if f.exists() else None


def summary_table(cfg: Config) -> pd.DataFrame:
    """Test-set mean and std over seeds for each model."""
    m = pd.read_csv(cfg.run_dir / "metrics.csv")
    g = m[m.split == "test"].groupby(["target", "model"], sort=False)
    s = g[list(METRICS)].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_seeds"] = g.seed.nunique()
    s = s.reset_index()
    order = {name: i for i, name in enumerate(NAMES)}  # fixed row order, even after merges
    return s.sort_values("model", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)


def significance(cfg: Config) -> pd.DataFrame:
    """Diebold-Mariano test of each window model vs. each FNN baseline (seed-averaged preds)."""
    rows = []
    for name in (*RECURRENT_MODELS, "window_fnn"):
        for base in ("fnn_paper", "fnn_tuned"):
            a, b = load_preds(cfg, name), load_preds(cfg, base)
            if a is None or b is None:
                continue
            if not np.array_equal(a["dates"], b["dates"]):
                raise ValueError(f"{name} and {base} were scored on different days")
            te = a["split"] == "test"
            y, pa, pb = a["y_true"][te], a["preds"].mean(0)[te], b["preds"].mean(0)[te]
            stat, p_two, p_better = diebold_mariano(y, pa, pb)
            rows.append(
                {
                    "target": TARGET,
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


def num(x: float) -> str:
    """Readable fixed-point numbers: 2741 / 25.3 / 0.231."""
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}" if abs(x) >= 1 else f"{x:.3f}"


def latex_table(s: pd.DataFrame, cfg: Config) -> str:
    """Needs \\usepackage{booktabs} in the IEEE LaTeX template. Best value per column in bold."""
    n = int(s.n_seeds.max())  # seeds actually evaluated (persistence: 1, deterministic)
    lower_is_better = {"rmse": True, "mae": True, "r2": False, "acc10": False}
    best = {
        k: s[f"{k}_mean"].min() if low else s[f"{k}_mean"].max()
        for k, low in lower_is_better.items()
    }
    out = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{Test results, {UNIT} (from {cfg.test_start}), mean$\pm$std over {n} seeds.}}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & RMSE$\downarrow$ & MAE$\downarrow$ & $R^2\uparrow$ & Acc$_{\pm10\%}\uparrow$ \\",
        r"\midrule",
    ]
    for r in s.itertuples():
        cells = []
        for k in METRICS:
            mean, std = getattr(r, f"{k}_mean"), getattr(r, f"{k}_std")
            txt = num(mean) + (rf"$\pm${num(std)}" if std > 0 else "")  # NaN std (1 seed): none
            cells.append(rf"\textbf{{{txt}}}" if mean == best[k] else txt)
        out.append(f"{NAMES[r.model]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def plot_test_predictions(cfg: Config, models=("fnn_paper", "gru", "lstm")) -> None:
    """One IEEE single-column figure: measured vs. predicted over the test period."""
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
        if not measured:  # from the first model that has predictions
            ax.plot(dates, p["y_true"][te], "k-", lw=1, label="Measured")
            measured = True
        ax.plot(
            dates, p["preds"].mean(0)[te], lw=0.8, ls=["--", "-", ":"][k % 3], label=NAMES[name]
        )
    ax.set_ylabel(UNIT)
    ax.legend(frameon=False, fontsize=6)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(cfg.run_dir / "figures" / f"test_{TARGET}.pdf")
    plt.close(fig)


def make_report(cfg: Config) -> None:
    s = summary_table(cfg)
    dm = significance(cfg)
    s.to_csv(cfg.run_dir / "summary.csv", index=False)
    dm.to_csv(cfg.run_dir / "significance.csv", index=False)
    (cfg.run_dir / "results_table.tex").write_text(latex_table(s, cfg))
    plot_test_predictions(cfg)
    print(s.round(4).to_string(index=False))
    if len(dm):
        print("\nDiebold-Mariano (negative DM / small p_model_better => model beats baseline):")
        print(dm.round(4).to_string(index=False))
    print(f"\nSaved to {cfg.run_dir}: summary.csv, significance.csv, results_table.tex, figures/")
