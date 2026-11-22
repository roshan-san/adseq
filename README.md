# adseq

Residual recurrent networks (Res-RNN, Res-GRU, Res-LSTM) for forecasting the next-24-h mean biogas
flow from minute-level SCADA data of the Muscatine, IA water resource recovery facility, compared
with persistence, ridge, LightGBM, plain RNN/GRU/LSTM and the MLP of Schroer & Just (2024).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --extra cu132   # NVIDIA GPU
uv sync --extra cpu     # CPU only
uv run prepare_data.py  # downloads the data and builds data/processed/SCADA-FINAL.csv
```

## Run

```bash
uv run adseq evaluate --models persistence ridge sj_mlp
uv run adseq tune --models rnn_res rnn && uv run adseq evaluate --models rnn_res rnn
uv run adseq tune --models gru_res gru && uv run adseq evaluate --models gru_res gru
uv run adseq tune --models lstm_res lstm && uv run adseq evaluate --models lstm_res lstm
uv run adseq tune --models lgbm_res lgbm && uv run adseq evaluate --models lgbm_res lgbm
uv run adseq report
uv run adseq ui
```

Results go to MLflow (`runs/mlflow.db`) and `runs/main/`. `uv run adseq <command> --help` lists
the options.

## Data and references

- SCADA data: doi:10.25820/data.006715 (ODC-By 1.0)
- Schroer, H. W. & Just, C. L. (2024). ACS ES&T Engineering 4(3), 660-672.
  doi:10.1021/acsestengg.3c00435. Code: doi:10.5281/zenodo.8360096 (MIT)
