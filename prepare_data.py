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
RAW = ROOT / "data" / "raw" / "muscatine_scada.csv"
CODE = ROOT / "references" / "schroer2024"
PROCESSED = ROOT / "data" / "processed" / "SCADA-FINAL.csv"

SCADA_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605000002771&download=true"
CODE_URL = "https://zenodo.org/api/records/8360096/files/hunter-schroer/municipal-biogas-forecasting-v1.0.0.zip/content"


def run(notebook: str) -> None:
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


if not PROCESSED.exists() and not RAW.exists():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    part = RAW.with_suffix(".part")
    urllib.request.urlretrieve(SCADA_URL, part)
    part.rename(RAW)
    print(f"downloaded {RAW}")

if not CODE.exists():
    with urllib.request.urlopen(CODE_URL) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(CODE.parent)
    (CODE.parent / top).rename(CODE)
    print(f"downloaded {CODE}")

theirs = CODE / "SCADA" / "SCADA-FINAL.csv"
if theirs.is_symlink():
    theirs.unlink()
if not PROCESSED.exists():
    if not theirs.exists():
        raw_copy = CODE / "SCADA" / "SCADA-raw.csv"
        raw_copy.unlink(missing_ok=True)
        shutil.copyfile(RAW, raw_copy)
        run("SCADA/SCADA-pretreatment.ipynb")
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    theirs.replace(PROCESSED)

print(f"ready: {PROCESSED}")
