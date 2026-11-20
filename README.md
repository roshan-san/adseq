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

```bash
uv run adseq fetch
```

Downloads the raw SCADA data (85 MB, `data/raw/`) and Schroer & Just's code
(`reference/schroer2024/`). Then run their notebooks, unmodified, to build the processed data set
(`data/processed/SCADA-FINAL.csv`) and reproduce their MLP's test predictions. Their code needs an
environment from its time, so it runs outside the project environment:

```bash
uv run --no-project --with "pandas==2.1.4" --with "numpy<2" --with "scikit-learn==1.3.2" \
    --with matplotlib --with seaborn --with statsmodels python scripts/run_schroer.py
```

## Run

```bash
uv run adseq tune     --name main   # Optuna search per model -> runs/main/best_hp/
uv run adseq evaluate --name main   # train with every seed, score -> runs/main/{metrics,preds}/
uv run adseq report   --name main   # table + figure -> runs/main/summary.csv, figures/
uv run adseq run-all  --name main   # all three in sequence
```

Useful flags (all commands take the same ones; `uv run adseq <command> --help` lists them):

| Flag | Default | |
|---|---|---|
| `--name` | `main` | output folder `runs/<name>/` and MLflow experiment |
| `--models` | all | e.g. `--models gru_res lstm_res` |
| `--n-trials` | 50 | Optuna trials per tuned model |
| `--max-epochs` / `--patience` | 50 / 10 | recurrent training and early stopping |
| `--stride` | 1 | train / validate on every k-th minute (test: always every minute) |
| `--seeds` | 1 2 3 4 5 | evaluation seeds |

Quick check that everything runs (a few minutes on a CPU):

```bash
uv run adseq run-all --name smoke --models persistence ridge gru_res --n-trials 2 --max-epochs 1 --seeds 1
```

Tuning resumes where it stopped: re-running continues each model's Optuna study from
`runs/<name>/optuna/<model>.log`. Run one process per model at a time.

### All models in parallel (CPU)

```bash
tmux new -s adseq                     # so it survives closing the terminal
scripts/run_parallel.sh --name main   # then Ctrl-b d to detach; `tmux attach -t adseq` to return
```

One single-threaded process per recurrent model (tune, then evaluate) plus one tuning LightGBM;
once those finish, LightGBM and the baselines are evaluated alone (their flattened windows need
several GB), then the report. Logs go to `runs/logs/`; re-run the script to resume after a
failure.

Tuning trains and validates on every 10th minute (`--stride 10 --max-epochs 200 --patience 20`):
neighbouring minutes are near-duplicates, so it finds the same kind of models several times
faster. Evaluation, which produces the reported numbers, trains on every minute.

## Outputs

```
runs/<name>/
  best_hp/<model>.json       best hyper-parameters per tuned model
  optuna/<model>.log         Optuna study (resumable)
  metrics/<model>.csv        metrics per seed and split (train / val / test)
  preds/<model>.npz          every seed's predictions
  summary.csv                test metrics, mean and std over seeds
  figures/test_predictions.pdf
runs/mlflow.db               MLflow tracking: one run per tuning trial / evaluation seed
```

Browse the MLflow runs (filter by the `job` and `model` tags):

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
