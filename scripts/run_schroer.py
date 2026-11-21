"""Setup: download the SCADA data and Schroer & Just's (2024) code (doi:10.5281/zenodo.8360096),
then run their own notebooks, unmodified, to build their processed data set (moved to
data/processed/, where adseq reads it) and reproduce their final MLP's test predictions. Their
SHAP / figure cells are not run (hours, not needed). Needs an environment from their time
(pandas 2.1: their code uses the since-removed freq='T' and fillna(method=...)):

    uv run --no-project --with "pandas==2.1.4" --with "numpy<2" --with "scikit-learn==1.3.2" \\
        --with matplotlib --with seaborn --with statsmodels python scripts/run_schroer.py
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

ROOT = Path(__file__).resolve().parent.parent
# these must match adseq.data: copied, as this script runs in its own old environment
REF = ROOT / "reference" / "schroer2024"
RAW = ROOT / "data" / "raw" / "muscatine_scada.csv"  # = their SCADA-raw.csv
PROCESSED = ROOT / "data" / "processed" / "SCADA-FINAL.csv"  # where adseq reads it
PREDS = ROOT / "runs" / "schroer_repro"

# Public dataset: Muscatine, IA water resource recovery facility, minute SCADA data
# 2022-03-18..2023-02-28. Schroer & Just (2024), doi:10.25820/data.006715, licence ODC-By 1.0.
# Their code (MIT), doi:10.5281/zenodo.8360096 -> reference/schroer2024/
CODE_URL = "https://zenodo.org/api/records/8360096/files/hunter-schroer/municipal-biogas-forecasting-v1.0.0.zip/content"
SCADA_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605000002771&download=true"


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


def download() -> None:
    """The raw SCADA data (85 MB; not needed once PROCESSED exists) and their code, skipping
    whatever is already there."""
    if not PROCESSED.exists() and not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        part = RAW.with_suffix(".part")  # renamed only once complete
        urllib.request.urlretrieve(SCADA_URL, part)
        part.rename(RAW)
    if not REF.exists():
        with urllib.request.urlopen(CODE_URL) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(REF.parent)
        (REF.parent / top).rename(REF)


download()
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
