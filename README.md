# adseq

Residual recurrent networks (Res-RNN / Res-GRU / Res-LSTM) for 24-h-ahead biogas forecasting from
full-scale co-digestion SCADA data (Muscatine, IA WRRF), compared with persistence, ridge,
LightGBM, plain RNN/GRU/LSTM and Schroer & Just's (2024) MLP, retrained with their settings.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Pick the PyTorch build for your machine:

```bash
uv sync --extra cpu     # no NVIDIA GPU
uv sync --extra cu132   # NVIDIA GPU (nvidia-smi must show CUDA Version >= 13.2)
```

A plain `uv sync` without an extra removes torch.

On Windows, run everything in Git Bash (comes with Git for Windows); install uv from PowerShell
with `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
`.gitattributes` keeps the scripts' line endings bash-compatible. There is no tmux: keep the window
open and the laptop awake (re-running the script resumes it).

## Data

```bash
uv run prepare_data.py
```

Once, before anything else. It downloads the raw SCADA data (85 MB, `data/raw/`) and Schroer &
Just's code (`references/schroer2024/`), runs their pretreatment notebook unmodified to build the
data every model trains on (`data/processed/SCADA-FINAL.csv`). Steps already done are skipped. Their code needs an
environment from its time; the file declares it, and uv builds it.

## Quick start: everything in one command

```bash
git clone https://github.com/roshan-san/adseq.git && cd adseq
scripts/setup_and_run.sh
```

The one script for every result. In order: install the environment (CUDA PyTorch when
`nvidia-smi` works), build the data, evaluate the baselines (persistence, ridge, `sj_mlp`), tune
and evaluate the six recurrent models (`AT_ONCE`, 3, at a time), then LightGBM on its own (it needs several GB),
then write the report to `runs/main/`. Each step prints what it is doing; logs go to `runs/logs/`.

To change what runs, edit the script: the settings (`NAME`, `TRIALS` 50, `SEEDS` 1–5, `AT_ONCE` 3) and the
model lists are at the top, one model per line, and the steps are listed at the bottom. Comment
out a model or a step to skip it. Ctrl-C stops everything; re-run to resume.

The script also opens MLflow: watch progress at http://127.0.0.1:5000 while it runs (set
`MLFLOW_PORT` to 5001 if 5000 is taken). When it finishes, or a model fails, MLflow stays open
until you press Ctrl-C. To look at results later without training, run `scripts/mlflow_ui.sh`.

## Single commands by hand (what the script calls)

```bash
uv run adseq tune     --name main   # Optuna search per model -> MLflow tune/<model> runs
uv run adseq evaluate --name main   # train with every seed, score -> MLflow eval/<model> runs
uv run adseq report   --name main   # table + figure from MLflow -> runs/main/summary.csv, figures/
```

Useful flags (all commands take the same ones; `uv run adseq <command> --help` lists them):

| Flag | Default | |
|---|---|---|
| `--name` | `main` | output folder `runs/<name>/` and MLflow experiment |
| `--models` | all | e.g. `--models gru_res lstm_res` |
| `--n-trials` | 50 | Optuna trials per tuned model |
| `--seeds` | 1 2 3 4 5 | evaluation seeds |

Tuning resumes where it stopped: re-running continues each model's Optuna study from
`runs/<name>/optuna-<model>.db`. Run one process per model at a time.

Tuning trains and validates on every 10th minute, with longer epoch limits (`TUNE` in
`config.py`): neighbouring minutes are near-duplicates, so it finds the same kind of models
several times faster. Evaluation (`EVAL`), which produces the reported numbers, trains on every
minute.

## Outputs

Every result is in MLflow (`runs/mlflow.db`, artifacts in MLflow's default `mlruns/`), one
experiment per `--name`:

```
tune/<model>        best_val_rmse, best hyper-parameters (params); a new one per `tune`
  trial-<n>         hyper-parameters, learning curve
eval/<model>        test_<metric>_mean / _std over seeds, n_seeds, preds.npz (every seed's predictions)
  seed-<s>          hyper-parameters, learning curve, train / val / test metrics
```

`evaluate` makes a new `eval/<model>` run each time; the report uses the newest finished one per
model (a crashed evaluation is never used), and reads nothing but MLflow.

What stays in `runs/<name>/`:

```
optuna-<model>.db            Optuna study (resume state; evaluate reads the best hyper-parameters)
summary.csv                  the report's table, with the eval run_id of every row
figures/test_predictions.pdf
```

`summary.csv` and `figures/` are regenerated from MLflow by `adseq report`. Browse the runs
(filter with `tags.mlflow.runName LIKE 'eval/%' AND attributes.status = 'FINISHED'`):

```bash
scripts/mlflow_ui.sh   # http://127.0.0.1:5000; --port 5001 if 5000 is taken
```

## Development

```bash
uv run ruff format src && uv run ruff check src
```

## References

- Data: Muscatine, IA WRRF minute SCADA data, 2022-03-18 to 2023-02-28,
  doi:10.25820/data.006715 (ODC-By 1.0).
- Schroer, H. & Just, C. (2024), code: doi:10.5281/zenodo.8360096 (MIT). `sj_mlp` is their final
  SCADA MLP (`SCADA-MLP.ipynb`, cell 2). With their split and seed, adseq's version reproduces
  their test predictions (max difference 1e-13 ft³/min; MAPE 13.39 %, adj. R² 0.776; checked
  2026-11-21). In adseq it trains on the same pre-test minutes and seeds as the other models.
