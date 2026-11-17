# adseq

Residual recurrent networks (Res-RNN / Res-GRU / Res-LSTM) that forecast the next-24-h mean
biogas flow of the Muscatine, IA co-digestion plant from minute SCADA data
(Schroer & Just 2024, doi:10.25820/data.006715). Base paper: `basepaper_schroer2024.pdf`.

```bash
uv run adseq fetch                          # SCADA-raw.csv (85 MB) + Schroer & Just's code
# their notebooks, unmodified -> their SCADA-FINAL.csv + their MLP's test predictions
uv run --no-project --with "pandas==2.1.4" --with "numpy<2" --with "scikit-learn==1.3.2" \
    --with matplotlib --with seaborn --with statsmodels python scripts/run_schroer.py
uv run wandb login                          # once; tuning uses W&B Sweeps
uv run adseq run-all --name main            # tune -> evaluate -> report
uv run pytest                               # data, split and metric checks
```

Models: Schroer & Just's own MLP (run from their notebooks), persistence, ridge, RNN/GRU/LSTM
(ablation: predict the level), Res-RNN/GRU/LSTM (proposed: predict the change from the last hour),
all scored on their test set (98,353 minutes). Results land in `runs/<name>/` (summary.csv,
significance.csv, figures/).
