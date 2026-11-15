# adseq: recurrent vs. lag-FNN forecasting of a full-scale anaerobic digester

Compares **RNN / GRU / LSTM** with the **lag-feature FNN** of Long et al. (2026,
*Energy Convers. Manage.* 365, 121811) on daily biogas production at a full-scale plant. Main dataset:
the public **Muscatine, IA water resource recovery facility** data (Schroer & Just, 2023).
See `docs/paper_map.md` for how each IEEE paper section maps to the code.

## Setup
```bash
uv sync                 # CPU PyTorch, W&B, tyro
uv run wandb login      # required once: tuning uses W&B Sweeps (online)
```

## Data
```bash
mkdir -p data/raw && curl -L -o data/raw/muscatine_labs.csv \
  "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&filePid=13894605020002771&download=true"
# LABS-raw.csv, doi:10.25820/data.006715, ODC-By 1.0
```
Muscatine: two 1,840 m³ mesophilic CSTRs co-digesting sewage sludge with food/industrial waste and FOG;
daily 2020-01-01..2023-03-31 (1,103 days; no data 2021-04-06..06-27). Target `Biogas` (daily mean,
ft³/min); every other numeric column (feeds, temperature, pH, alkalinity, VFA, SRT, …) is an input.
Test set = last 12 months (`--test-start 2022-04-01`).

Cleaning on load: two impossible Dig2 temperature readings (121.5 / 126.1 °F on 2022-04-10/11)
are set to missing and forward-filled like any short gap (`data.py: VALID_RANGE`).

## Run
```bash
uv run adseq tune                    # W&B Sweeps (Bayesian), 40 runs per tuned model, resumable
uv run adseq evaluate                # 10 seeds per model, test 2022-04..2023-03
uv run adseq report                  # results_table.tex, significance.csv, figures/
uv run adseq run-all                 # all of the above
uv run adseq evaluate --help         # every option, documented
```
Outputs go to `runs/<name>/`. Learning curves and per-run metrics go to W&B (project `adseq`,
grouped by `--name`; tuning runs are in group `<name>-tune`, one sweep per model).
`evaluate`/`report` also work with `--wandb-mode offline` (upload later with `uv run wandb sync runs/<name>/wandb/offline-run-*`).

## Quick pipeline check (tiny budget, numbers mean nothing)
```bash
uv run adseq run-all --name quick --n-trials 2 --seeds 0 1 --max-epochs 40
```

## Code (one file per method step)
| file | paper section |
|---|---|
| `config.py` | all settings (tyro flags) |
| `data.py` | III / IV-A: loading, features, split, scaling, windows |
| `models.py` | IV-B: FNN, RNN, GRU, LSTM |
| `train.py` | IV-C: shared training loop |
| `tune.py` | IV-C: equal-budget W&B Sweeps search |
| `evaluate.py`, `metrics.py` | IV-D: multi-seed scoring, Diebold–Mariano |
| `report.py` | V: LaTeX table, figures |

Lint: `uv run ruff check . && uv run ruff format .` · Tests: `uv run pytest`
