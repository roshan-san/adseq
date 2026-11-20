"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Annotated

import mlflow
import tyro

from .config import Config
from .data import PROCESSED, build_dataset, load_processed
from .evaluate import evaluate
from .models import TUNED_MODELS
from .report import make_report
from .tune import tune

# flags read --stride instead of --cfg.stride
Cfg = Annotated[Config, tyro.conf.arg(name="")]

# Public dataset: Muscatine, IA water resource recovery facility, minute SCADA data
# 2022-03-18..2023-02-28. Schroer & Just (2024), doi:10.25820/data.006715, licence ODC-By 1.0.
# Their code (MIT), doi:10.5281/zenodo.8360096 -> reference/schroer2024/
CODE_URL = "https://zenodo.org/api/records/8360096/files/hunter-schroer/municipal-biogas-forecasting-v1.0.0.zip/content"
SCADA_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605000002771&download=true"


def fetch_cmd(cfg: Cfg) -> None:
    """Download SCADA-raw.csv (85 MB) and Schroer & Just's code, skipping whatever is already
    there; then run scripts/run_schroer.py. The raw file is only needed to build
    data/processed/SCADA-FINAL.csv, so it is skipped when that already exists."""
    if PROCESSED.exists():
        print(f"have {PROCESSED}; raw download not needed")
    elif not cfg.data_path.exists():
        cfg.data_path.parent.mkdir(parents=True, exist_ok=True)
        part = cfg.data_path.with_suffix(".part")  # renamed only once complete
        urllib.request.urlretrieve(SCADA_URL, part)
        part.rename(cfg.data_path)
        print(f"have {cfg.data_path}")
    dest = Path("reference/schroer2024")
    if not dest.exists():
        with urllib.request.urlopen(CODE_URL) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(dest.parent)
        (dest.parent / top).rename(dest)
    print(f"have {dest}")


def start_tracking(cfg: Config) -> None:
    """MLflow: log to cfg.mlflow_uri, in the experiment named --name."""
    Path("runs").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.name)


def tune_cmd(cfg: Cfg) -> None:
    """Optuna search for every tuned model (same trial budget for all)."""
    start_tracking(cfg)
    ds = build_dataset(load_processed(), cfg)  # built once, shared by every tuned model
    for name in cfg.models:
        if name in TUNED_MODELS:
            tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    """Train each model with every seed using its best hyper-parameters; score on the test period."""
    start_tracking(cfg)
    evaluate(cfg, load_processed())


def report_cmd(cfg: Cfg) -> None:
    """Results table (incl. Schroer & Just's own MLP), significance tests, figure."""
    make_report(cfg)


def run_all_cmd(cfg: Cfg) -> None:
    """tune -> evaluate -> report."""
    tune_cmd(cfg)
    evaluate_cmd(cfg)
    report_cmd(cfg)


def main() -> None:
    tyro_map = {
        "fetch": fetch_cmd,
        "tune": tune_cmd,
        "evaluate": evaluate_cmd,
        "report": report_cmd,
        "run-all": run_all_cmd,
    }
    tyro.extras.subcommand_cli_from_dict(tyro_map)
