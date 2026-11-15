"""Pull the numbers the deck needs from a run into one JSON file.

uv run python presentation/export_results.py runs/<name> presentation/results.json "<banner>"
banner: shown above the results (e.g. that a run is preliminary); "" for none.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run, out, banner = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

s = pd.read_csv(run / "summary.csv")
summary = {
    t: {r.model: {"rmse": r.rmse_mean, "mae": r.mae_mean, "r2": r.r2_mean} for r in g.itertuples()}
    for t, g in s.groupby("target", sort=False)
}

target = next(iter(summary))  # the chart shows the first target
p = {m: dict(np.load(run / "preds" / f"{target}__{m}.npz")) for m in ("fnn_paper", "rnn")}
te = p["rnn"]["split"] == "test"
step = 3  # every 3rd day keeps the native line chart light
series = {
    "target": target,
    "dates": pd.to_datetime(p["rnn"]["dates"][te][::step]).strftime("%b %y").tolist(),
    "measured": p["rnn"]["y_true"][te][::step].round(1).tolist(),
    "fnn_paper": p["fnn_paper"]["preds"].mean(0)[te][::step].round(1).tolist(),
    "rnn": p["rnn"]["preds"].mean(0)[te][::step].round(1).tolist(),
}
n = {sp: int((p["rnn"]["split"] == sp).sum()) for sp in ("train", "val", "test")}
sig_file = run / "significance.csv"  # Diebold-Mariano tests vs. both FNN baselines
sig = pd.read_csv(sig_file).to_dict("records") if sig_file.exists() else []
res = {"banner": banner, "summary": summary, "series": series, "n": n, "sig": sig}
out.write_text(json.dumps(res, indent=1))
print(json.dumps(summary, indent=1), n)
