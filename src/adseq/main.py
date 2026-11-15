"""Command line (tyro). Run `uv run adseq --help` or `uv run adseq <command> --help`."""

from typing import Annotated

import tyro

from .config import Config
from .data import build_dataset, clean_data
from .evaluate import evaluate
from .report import make_report
from .tune import TUNED_MODELS, tune

# flags read --window instead of --cfg.window
Cfg = Annotated[Config, tyro.conf.arg(name="")]


# Public main dataset: Muscatine, IA water resource recovery facility, daily lab/operational data
# 2020-01-01..2023-03-31. Schroer & Just (2023), doi:10.25820/data.006715, licence ODC-By 1.0.
MUSCATINE_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605020002771&download=true"


def tune_cmd(cfg: Cfg) -> None:
    """W&B Sweeps search for every tuned model (same run budget for all)."""
    cleaned_data = clean_data(cfg.data_path)
    ds = build_dataset(cleaned_data, cfg)  # built once, shared by every tuned model
    for name in cfg.models:
        if name in TUNED_MODELS:
            tune(cfg, ds, name)


def evaluate_cmd(cfg: Cfg) -> None:
    """Train each model with every seed using its best hyper-parameters; score on the test period."""
    evaluate(cfg, clean_data(cfg.data_path))


def report_cmd(cfg: Cfg) -> None:
    """Write summary.csv, significance.csv, results_table.tex and figures/."""
    make_report(cfg)


def run_all_cmd(cfg: Cfg) -> None:
    """tune -> evaluate -> report."""
    tune_cmd(cfg)
    evaluate_cmd(cfg)
    report_cmd(cfg)


def main() -> None:
    tyro_map = {
        "tune": tune_cmd,
        "evaluate": evaluate_cmd,
        "report": report_cmd,
        "run-all": run_all_cmd,
    }
    tyro.extras.subcommand_cli_from_dict(tyro_map)
