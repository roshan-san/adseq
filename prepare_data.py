# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "pandas==2.1.4",
#     "numpy<2",
#     "scikit-learn==1.3.2",
#     "matplotlib",
#     "seaborn",
#     "statsmodels",
# ]
# ///
"""Get the data and reproduce Schroer & Just's (2024) MLP: `uv run prepare_data.py`, once.

1. download the raw SCADA data (Muscatine, IA WRRF)   -> data/raw/muscatine_scada.csv
2. download their code (doi:10.5281/zenodo.8360096)     -> references/schroer2024/
3. run their SCADA-pretreatment notebook, unmodified   -> data/processed/SCADA-FINAL.csv
4. run their SCADA-MLP notebook (cells 0-3), unmodified -> data/processed/schroer_mlp_test.csv

Every step is skipped when its output exists. Their code needs the environment above, from its
time (it uses pandas' since-removed freq='T' and fillna(method=...)), so this file declares its
own and isn't part of the adseq package. Their SHAP / figure cells are not run (hours, not needed).
"""

import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
# PROCESSED and SJ_PREDS must match adseq.data / adseq.report (adseq can't be imported here)
RAW = ROOT / "data" / "raw" / "muscatine_scada.csv"
CODE = ROOT / "references" / "schroer2024"
PROCESSED = ROOT / "data" / "processed" / "SCADA-FINAL.csv"
SJ_PREDS = ROOT / "data" / "processed" / "schroer_mlp_test.csv"

# minute SCADA data 2022-03-18..2023-02-28, doi:10.25820/data.006715, licence ODC-By 1.0
SCADA_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605000002771&download=true"
# their code, MIT licence
CODE_URL = "https://zenodo.org/api/records/8360096/files/hunter-schroer/municipal-biogas-forecasting-v1.0.0.zip/content"


def run(notebook: str, cells: slice = slice(None)) -> dict:
    """Their notebook's code cells, verbatim, in its own folder (their relative paths); returns
    their variables."""
    path = CODE / notebook
    code = [c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in code[cells]:
            src = "".join(cell["source"])
            if src.strip():
                exec(compile(src, notebook, "exec"), ns)
            plt.close("all")
    finally:
        os.chdir(cwd)
    print(f"ran {notebook}")
    return ns


def link(target: Path, name: Path) -> None:
    """Put a file where their notebooks expect it."""
    if name.is_symlink():  # ours, possibly stale (repo moved): renew it
        name.unlink()
    if not name.exists():
        name.symlink_to(target)


# 1. raw data: not needed once the processed file exists
if not PROCESSED.exists() and not RAW.exists():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    part = RAW.with_suffix(".part")  # renamed only once complete
    urllib.request.urlretrieve(SCADA_URL, part)
    part.rename(RAW)
    print(f"downloaded {RAW}")

# 2. their code
if not CODE.exists():
    with urllib.request.urlopen(CODE_URL) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(CODE.parent)
    (CODE.parent / top).rename(CODE)
    print(f"downloaded {CODE}")

# 3. their pretreatment: writes SCADA-FINAL.csv in their folder; we keep it in data/processed
theirs = CODE / "SCADA" / "SCADA-FINAL.csv"
if not PROCESSED.exists():
    if not theirs.exists() or theirs.is_symlink():
        link(RAW, CODE / "SCADA" / "SCADA-raw.csv")
        run("SCADA/SCADA-pretreatment.ipynb")
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    theirs.replace(PROCESSED)
link(PROCESSED, theirs)  # their MLP notebook reads it from their folder

# 4. their final MLP, trained and scored on their test set
if not SJ_PREDS.exists():
    ns = run("SCADA/SCADA-MLP.ipynb", slice(0, 4))
    pd.DataFrame({"y_true": ns["y_test"], "pred": ns["y_test_ann1"]}).to_csv(SJ_PREDS)
    print(f"saved {SJ_PREDS}")

print(f"ready: {PROCESSED}, {SJ_PREDS}")
