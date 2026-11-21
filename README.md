# adseq

Residual recurrent networks (Res-RNN / Res-GRU / Res-LSTM) for 24-h-ahead biogas forecasting from
full-scale co-digestion SCADA data (Muscatine, IA WRRF), compared with persistence, ridge,
LightGBM, plain RNN/GRU/LSTM and Schroer & Just's (2024) own MLP.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Pick the PyTorch build for your machine:

```bash
uv sync --extra cpu     # no NVIDIA GPU
uv sync --extra cu132   # NVIDIA GPU (nvidia-smi must show CUDA Version >= 13.2)
```

A plain `uv sync` without an extra removes torch.

## Data

One script downloads the raw SCADA data (85 MB, `data/raw/`) and Schroer & Just's code
(`reference/schroer2024/`), then runs their notebooks, unmodified, to build the processed data
set (`data/processed/SCADA-FINAL.csv`) and reproduce their MLP's test predictions. Their code
needs an environment from its time, so it runs outside the project environment:

```bash
uv run --no-project --with "pandas==2.1.4" --with "numpy<2" --with "scikit-learn==1.3.2" \
    --with matplotlib --with seaborn --with statsmodels python scripts/run_schroer.py
```

## Run

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
`runs/<name>/optuna/<model>.log`. Run one process per model at a time.

### All models (CPU)

```bash
tmux new -s adseq                      # so it survives closing the terminal
scripts/run_baselines.sh --name main   # persistence + ridge: a few minutes
scripts/run_recurrent.sh --name main   # then Ctrl-b d to detach; `tmux attach -t adseq` to return
scripts/run_lgbm.sh --name main        # once the recurrent models are done
```

`run_baselines.sh` evaluates persistence and ridge (nothing to tune). `run_recurrent.sh` tunes and
evaluates the six recurrent models, one single-threaded process each. `run_lgbm.sh` tunes and
evaluates LightGBM using every core; run it after the recurrent models, alone: its flattened
windows need several GB. Each script ends with the report, which leaves out models not run yet. Logs go
to `runs/logs/`; re-run a script to resume after a failure.

Tuning trains and validates on every 10th minute, with longer epoch limits (`TUNE` in
`config.py`): neighbouring minutes are near-duplicates, so it finds the same kind of models
several times faster. Evaluation (`EVAL`), which produces the reported numbers, trains on every
minute.

## Outputs

Every result is in MLflow (`runs/mlflow.db`, artifacts in MLflow's default `mlruns/`), one
experiment per `--name`:

```
tune/<model>        best_val_rmse, best_params.json (best hyper-parameters); a new one per `tune`
  trial-<n>         hyper-parameters, learning curve; tag pruned
eval/<model>        test_<metric>_mean / _std over seeds, n_seeds, preds.npz (every seed's predictions)
  seed-<s>          hyper-parameters, learning curve, train / val / test metrics
```

`evaluate` makes a new `eval/<model>` run each time; the report uses the newest finished one per
model (a crashed evaluation is never used). The report adds Schroer & Just's MLP from
`runs/schroer_repro/schroer_mlp_test.csv` (written by `scripts/run_schroer.py`), scored on the
same test minutes.

What stays in `runs/<name>/`:

```
optuna/<model>.log           Optuna study (resume state)
summary.csv                  the report's table, with the eval run_id of every row
figures/test_predictions.pdf
```

`summary.csv` and `figures/` are regenerated from MLflow by `adseq report`. Browse the runs
(filter with `tags.kind = 'eval'`):

```bash
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```

## Development

```bash
uv run ruff format src && uv run ruff check src
```

## References

- Data: Muscatine, IA WRRF minute SCADA data, 2022-03-18 to 2023-02-28,
  doi:10.25820/data.006715 (ODC-By 1.0).
- Schroer, H. & Just, C. (2024), code: doi:10.5281/zenodo.8360096 (MIT).
