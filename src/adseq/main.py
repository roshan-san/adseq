"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

import dataclasses
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Annotated

import tyro

from .config import Config
from .data import PROCESSED, build_dataset, load_processed
from .evaluate import evaluate
from .models import TUNED_MODELS
from .report import make_report
from .tracking import start_tracking
from .tune import tune

# flags read --n-trials instead of --cfg.n-trials
Cfg = Annotated[Config, tyro.conf.arg(name="")]

# Public dataset: Muscatine, IA water resource recovery facility, minute SCADA data
# 2022-03-18..2023-02-28. Schroer & Just (2024), doi:10.25820/data.006715, licence ODC-By 1.0.
# Their code (MIT), doi:10.5281/zenodo.8360096 -> reference/schroer2024/
CODE_URL = "https://zenodo.org/api/records/8360096/files/hunter-schroer/municipal-biogas-forecasting-v1.0.0.zip/content"
SCADA_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605000002771&download=true"
RAW = Path("data/raw/muscatine_scada.csv")  # their SCADA-raw.csv; input of their pretreatment
CODE_DIR = Path("reference/schroer2024")


def fetch_cmd(cfg: Cfg) -> None:
    """Download SCADA-raw.csv (85 MB) and Schroer & Just's code, skipping whatever is already
    there. Then run scripts/run_schroer.py (see its docstring) to build
    data/processed/SCADA-FINAL.csv; the raw file is not downloaded once that exists."""
    if PROCESSED.exists():
        print(f"have {PROCESSED}; raw download not needed")
    elif not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        part = RAW.with_suffix(".part")  # renamed only once complete
        urllib.request.urlretrieve(SCADA_URL, part)
        part.rename(RAW)
        print(f"have {RAW}")

    if not CODE_DIR.exists():
        with urllib.request.urlopen(CODE_URL) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(CODE_DIR.parent)
        (CODE_DIR.parent / top).rename(CODE_DIR)
    print(f"have {CODE_DIR}")


def tune_cmd(cfg: Cfg) -> None:
    """Optuna search for every tuned model (same trial budget for all), with the --tune-*
    stride and epoch limits."""
    start_tracking(cfg)
    cfg = dataclasses.replace(
        cfg, stride=cfg.tune_stride, max_epochs=cfg.tune_max_epochs, patience=cfg.tune_patience
    )
    ds = build_dataset(load_processed(), cfg)  # built once, shared by every model
    for name in cfg.models:
        if name in TUNED_MODELS:
            tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    """Train each model with every seed using its best hyper-parameters; score on the test period."""
    start_tracking(cfg)
    ds = build_dataset(load_processed(), cfg)  # built once, shared by every model
    for name in cfg.models:
        evaluate(cfg, ds, name)


def report_cmd(cfg: Cfg) -> None:
    """Results table (incl. Schroer & Just's own MLP) and figure."""
    start_tracking(cfg)
    make_report(cfg)


def run_all_cmd(cfg: Cfg) -> None:
    """tune -> evaluate -> report."""
    tune_cmd(cfg)
    evaluate_cmd(cfg)
    report_cmd(cfg)


def main() -> None:
    commands = {
        "fetch": fetch_cmd,
        "tune": tune_cmd,
        "evaluate": evaluate_cmd,
        "report": report_cmd,
        "run-all": run_all_cmd,
    }
    tyro.extras.subcommand_cli_from_dict(commands)
