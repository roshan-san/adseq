# adseq

Residual recurrent networks (Res-RNN / Res-GRU / Res-LSTM) for 24-h-ahead biogas forecasting from
full-scale co-digestion SCADA data (Muscatine, IA WRRF), compared with persistence, ridge,
LightGBM, plain RNN/GRU/LSTM and Schroer & Just's (2024) MLP, retrained with their settings.

## Setup (once)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. On Windows, use Git Bash (comes with Git
for Windows) for every command below; install uv from PowerShell with
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`. If Git Bash
then says `uv: command not found`:

```bash
echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc   # then open a new window
```

Install the environment with the PyTorch build for your machine:

```bash
uv sync --extra cu132   # NVIDIA GPU (nvidia-smi must show CUDA Version >= 13.2)
uv sync --extra cpu     # no NVIDIA GPU
```

A plain `uv sync` without an extra removes torch.

## Data (once)

```bash
uv run prepare_data.py
```

It downloads the raw SCADA data (85 MB, `data/raw/`) and Schroer & Just's code
(`references/schroer2024/`), and runs their pretreatment notebook unmodified to build the data
every model trains on (`data/processed/SCADA-FINAL.csv`). Steps already done are skipped. Their
code needs an environment from its time; the file declares it, and uv builds it.

## Run, step by step

Every command prints what it is doing. Every command can be stopped with Ctrl-C and re-run: tuning
continues from the trials already done. Keep the windows open and the laptop awake.

**1. Baselines** (a few minutes): persistence, ridge and Schroer & Just's MLP, nothing to tune.

```bash
uv run adseq evaluate --models persistence ridge sj_mlp
```

**2. Recurrent models**: three Git Bash windows, one line each, all at the same time. Each tunes
its two models, then evaluates them. Every trial trains 3 seeds, so tuning takes about 3× as long
as one seed would. Three at a time is what a laptop GPU's memory allows (six ran
out). Leave a window out to train those models later.

```bash
uv run adseq tune --models rnn_res rnn   && uv run adseq evaluate --models rnn_res rnn     # window 1
uv run adseq tune --models gru_res gru   && uv run adseq evaluate --models gru_res gru     # window 2
uv run adseq tune --models lstm_res lstm && uv run adseq evaluate --models lstm_res lstm   # window 3
```

Without an NVIDIA GPU, give each window a third of the CPU cores, e.g. on 12 cores start each line
with `OMP_NUM_THREADS=4 `.

**3. LightGBM**, after step 2 has finished and alone: its flattened windows need several GB.

```bash
uv run adseq tune --models lgbm_res lgbm && uv run adseq evaluate --models lgbm_res lgbm
```

**4. Report**: the results table and figure, in `runs/main/summary.csv` and `runs/main/figures/`.
Models not evaluated yet are left out, so it can be run at any point.

```bash
uv run adseq report
```

**Watching progress** in the browser, in one more window, at any time:

```bash
uv run adseq ui   # http://127.0.0.1:5000; the first start can take ~30 s; Ctrl-C closes it
```

Training does not need it: every result is recorded either way. If port 5000 is taken:
`uv run adseq ui --port 5001`.

## Settings

Defaults are in `src/adseq/config.py`; the commands above use them, so no flags are needed.
`uv run adseq <command> --help` lists them.

| Flag | Default | |
|---|---|---|
| `--name` | `main` | output folder `runs/<name>/` and MLflow experiment; e.g. `--name test` for a trial run |
| `--models` | all | which models a command runs |
| `--n-trials` | 30 | Optuna trials per tuned model; if changed, pass the same to `tune` and `evaluate` |
| `--trial-seeds` | 0 1 2 | seeds each trial trains with; it scores their mean val RMSE |
| `--seeds` | 1 2 3 4 5 | evaluation seeds |
| `--max-epochs` | `TUNE` / `EVAL` limits | caps the recurrent models' epochs in `tune` and `evaluate`; for test runs only |

A quick test of the whole pipeline, then delete `runs/test/` and the `test` experiment (in the UI):

```bash
uv run adseq tune --name test --models rnn_res --n-trials 2 --trial-seeds 0 --max-epochs 3
uv run adseq evaluate --name test --models rnn_res --n-trials 2 --seeds 1 --max-epochs 3
uv run adseq report --name test
```

Tuning resumes from `runs/<name>/optuna-<model>.db`; never run the same model in two windows at once.
A study tuned under an earlier protocol (`PROTOCOL` in `config.py`) is refused: tune again under a
new `--name`. A study is continued only with the `--trial-seeds` it was started with.

## Protocol

- **Inputs.** Ridge and Schroer & Just's MLP see their 36 features at minute t. The recurrent
  models and LightGBM see the 13 sensor readings as a sequence of hourly means (one per hour of
  the window), plus the 23 features that already summarise the past (previous-hour and 24-h
  means and their daily lags) and the time of day once, at t.
- **Two fits per model and seed.** One on the train period alone, with val used only for early
  stopping: the val metrics come from it, so every model, baselines included, is scored on val
  data it never trained on. Then a refit on train + val (every pre-test minute) for the number of
  epochs / boosting rounds the first fit stopped at; ridge and the MLP re-run their own selection
  (RidgeCV, their early stopping), as in their notebooks. The test metrics come from the refit, so
  every model has seen the same, most recent data before the test period.
- **Tuning** trains and validates on every 10th minute (neighbouring minutes are near-duplicates),
  evaluation on every minute. A recurrent epoch is 30,000 randomly drawn training windows in both
  (`EPOCH_WINDOWS`), with the same limits (`TUNE`, `EVAL`), so an epoch count carries over from
  tuning to evaluation and to the refit. Each trial is scored by its 3 seeds' mean val RMSE.
- Model selection uses val only; the test period is scored once, by `evaluate`.

## Outputs

Every result is in MLflow (`runs/mlflow.db`, artifacts in MLflow's default `mlruns/`), one
experiment per `--name`:

```
tune/<model>        best_val_rmse (ft³/min), best_val_rmse_scaled, best hyper-parameters (params);
                    a new one per `tune`
  trial-<n>         hyper-parameters, val_rmse_scaled (mean over its seeds) and _std
    seed-<s>        learning curve (below), best_epoch / best_round, val_rmse_scaled
eval/<model>        val_ / test_<metric>_mean / _std over seeds, n_seeds, preds.npz (every
                    seed's predictions)
  seed-<s>          hyper-parameters, learning curve, train / val metrics (the fit on train) and
                    test metrics (the refit)
    refit           the refit on train + val: its training curve, epochs / rounds
```

A recurrent model's learning curve has one step per epoch (30,000 training windows, then the val
set): `epoch_train_mse_scaled`, `epoch_val_rmse_scaled` (what early stopping uses),
`epoch_val_mae` (ft³/min) and `epoch_val_mape` (%). Step 0 is the untrained network; for a `_res`
model that is exactly persistence, so the curve starts at the baseline it has to beat. LightGBM's
curve (`val-rmse`, MLflow's autolog) has one step per boosting round.

Metrics ending in `_scaled` are in the units the networks train in: biogas min-max scaled on the
training period, so 1 = its range there (about 168 ft³/min); multiply an RMSE or MAE by that range
for ft³/min. Every other metric is in ft³/min (or %, or unitless for R² and TeEI).

`evaluate` makes a new `eval/<model>` run each time; the report uses the newest finished one per
model (a crashed evaluation is never used), and reads nothing but MLflow.

What stays in `runs/<name>/`:

```
optuna-<model>.db            Optuna study (resume state; evaluate reads the best hyper-parameters)
summary.csv                  the report's table, with the eval run_id of every row
figures/test_predictions.pdf
```

`summary.csv` (val and test, mean ± std over seeds) and `figures/` are regenerated from MLflow by
`adseq report`. Browse the runs with
`uv run adseq ui` (filter with `tags.mlflow.runName LIKE 'eval/%' AND attributes.status = 'FINISHED'`).

## Future work

- **Ridge as the residual base.** The `_res` models now learn the change from persistence, a weak
  base (val RMSE about 25 ft³/min against ridge's about 20). With ridge's prediction as the base
  instead, the recurrent network would only learn the temporal correction a single-minute linear
  model misses. Ridge must then be fitted on the train rows only (as in the first fit), so the
  residuals it leaves on train are as large as on unseen data.

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
  2026-11-21). In adseq it is fitted and refitted like every other model (see Protocol), with the
  same seeds.
