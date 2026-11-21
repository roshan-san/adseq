"""Run Schroer & Just's (2024) own notebooks, unmodified, to rebuild their processed SCADA data
set and reproduce their final MLP. Their code: reference/schroer2024 (doi:10.5281/zenodo.8360096).

Each notebook's code cells are executed verbatim, in order, in the notebook's own folder (their
relative paths), with a non-interactive plot backend. Their processed data set (SCADA-FINAL.csv)
is moved to data/processed/, where adseq reads it, and linked back into their folder for their
MLP notebook. Their SHAP / figure cells after the MLP
metrics are not run (SHAP takes hours and is not needed). Run in an environment from their time (pandas 2.1; their code uses the
since-removed freq='T' and fillna(method=...)):

    uv run --no-project --with "pandas==2.1.4" --with "numpy<2" --with "scikit-learn==1.3.2" \\
        --with matplotlib --with seaborn --with statsmodels python scripts/run_schroer.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# these must match adseq.data: copied, as this script runs in its own old environment
REF = ROOT / "reference" / "schroer2024"
RAW = ROOT / "data" / "raw" / "muscatine_scada.csv"  # = their SCADA-raw.csv
PROCESSED = ROOT / "data" / "processed" / "SCADA-FINAL.csv"  # where adseq reads it
PREDS = ROOT / "runs" / "schroer_repro"


def run(notebook: str, cells: slice = slice(None)) -> dict:
    """Execute the notebook's code cells verbatim in its folder; return their variables."""
    path = REF / notebook
    code = [c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in code[cells]:
            src = "".join(cell["source"])
            if src.strip():
                exec(compile(src, f"{notebook}", "exec"), ns)
            plt.close("all")
    finally:
        os.chdir(cwd)
    print(f"ran {notebook}")
    return ns


def link(target: Path, name: Path) -> None:
    """Put an input where their notebook expects it (their relative file names)."""
    if name.is_symlink():  # ours, possibly stale (repo moved): renew it
        name.unlink()
    if not name.exists():
        name.symlink_to(target)


theirs = REF / "SCADA" / "SCADA-FINAL.csv"  # where their notebooks write / read it
if not PROCESSED.exists():
    if not theirs.exists() or theirs.is_symlink():  # not built yet (else: built by an older layout)
        link(RAW, REF / "SCADA" / "SCADA-raw.csv")
        run("SCADA/SCADA-pretreatment.ipynb")  # writes theirs
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    theirs.replace(PROCESSED)
link(PROCESSED, theirs)
ns = run("SCADA/SCADA-MLP.ipynb", slice(0, 4))  # their final MLP, trained and scored
PREDS.mkdir(parents=True, exist_ok=True)
out = PREDS / "schroer_mlp_test.csv"
pd.DataFrame({"y_true": ns["y_test"], "pred": ns["y_test_ann1"]}).to_csv(out)
print(f"saved {out}")
