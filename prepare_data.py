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
"""Get the data every model trains on, built by Schroer & Just's (2024) own code:
`uv run prepare_data.py`, once.

1. download the raw SCADA data (Muscatine, IA WRRF)   -> data/raw/muscatine_scada.csv
2. download their code (doi:10.5281/zenodo.8360096)     -> references/schroer2024/
3. run their SCADA-pretreatment notebook, unmodified   -> data/processed/SCADA-FINAL.csv

Every step is skipped when its output exists. Their code needs the environment above, from its
time (it uses pandas' since-removed freq='T' and fillna(method=...)), so this file declares its
own and isn't part of the adseq package.
"""

import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# PROCESSED must match adseq.data (adseq can't be imported here)
RAW = ROOT / "data" / "raw" / "muscatine_scada.csv"
CODE = ROOT / "references" / "schroer2024"
PROCESSED = ROOT / "data" / "processed" / "SCADA-FINAL.csv"

# minute SCADA data 2022-03-18..2023-02-28, doi:10.25820/data.006715, licence ODC-By 1.0
SCADA_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605000002771&download=true"
# their code, MIT licence
CODE_URL = "https://zenodo.org/api/records/8360096/files/hunter-schroer/municipal-biogas-forecasting-v1.0.0.zip/content"


def run(notebook: str) -> None:
    """Their notebook's code cells, verbatim, in its own folder (their relative paths)."""
    path = CODE / notebook
    code = [
        c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"
    ]
    ns: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in code:
            src = "".join(cell["source"])
            if src.strip():
                exec(compile(src, notebook, "exec"), ns)
            plt.close("all")
    finally:
        os.chdir(cwd)
    print(f"ran {notebook}")


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
if theirs.is_symlink():  # older versions linked it to PROCESSED: writing through it breaks both
    theirs.unlink()
if not PROCESSED.exists():
    if not theirs.exists():
        raw_copy = CODE / "SCADA" / "SCADA-raw.csv"  # where their notebook reads it
        raw_copy.unlink(missing_ok=True)  # also the symlink older versions made
        shutil.copyfile(RAW, raw_copy)  # a copy: Windows allows symlinks only to admins
        run("SCADA/SCADA-pretreatment.ipynb")
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    theirs.replace(PROCESSED)

print(f"ready: {PROCESSED}")
