# Paper map: IEEE section → what to explain → code → artifact

Template: IEEEtran conference (`\documentclass[conference]{IEEEtran}`), usually 6 pages
including references. Base paper = Long et al., *Energy Convers. Manage.* 365 (2026) 121811.

| IEEE section | What you must explain | Code | Artifact |
|---|---|---|---|
| **Abstract** | Problem, the gap (hand-made lag features + static FNN), what you did (RNN/GRU/LSTM vs the base paper's lag-FNN method on a public full-scale dataset, equal tuning budget), the main number with significance | – | `summary.csv`, `significance.csv` |
| **I. Introduction** | AD needs reliable gas forecasting. The base paper models SCAD with a 128-64-1 FNN on lag/rolling features (it calls this RL, but it's trained like supervised regression with MSE). **Gap:** the memory length is fixed by hand (1, 2, 7, 15 days). **Contributions:** (1) recurrent models that learn memory from the sequence, (2) a leakage-free and equal-budget protocol, (3) significance testing and ablations | – | – |
| **II. Related Work** | ML for AD (RF, ANN, SVM: base paper refs [5,10,15,20]); RNN/LSTM/GRU for bioprocess and energy time series; the base paper | – | – |
| **III. Dataset & Problem Formulation** | Public Muscatine, IA WRRF data (Schroer & Just, *ACS ES&T Eng.* 2023; data doi:10.25820/data.006715, ODC-By): two 1,840 m³ CSTRs at 36 °C, sewage sludge co-digested with food/industrial waste and FOG; daily 2020-01..2023-03; 21 operational inputs, target = daily biogas flow (ft³/min). Say why not SCAD: the base paper's data is only "available on request", so a public plant makes the study reproducible. Cleaning: the authors' pre-cleaning (data dictionary) + 2021-04-06..06-27 meter outage left as a gap (no windows cross it); VFA starts 2020-04-21. Formulation ŷ_t = f(x_{t−L+1..t}). Split: test = last 12 months (2022-04-01..2023-03-31, a full seasonal cycle like the base paper's test year); before that, 80 % train / 20 % validation **chronologically** (random splits leak neighbouring days, the pitfall in the base paper's ref. [33]). | `data.py: clean_data, build_dataset` | split table: train 467 d (2020-05-05..2021-12-04), val 117 d (..2022-03-31), test 365 d (2022-04-01..2023-03-31) |
| **IV-A. Features** | Cyclic month/day-of-year, lags {1,2,7,15}, rolling mean/std {5,10,15}, all built from days < t only. One deviation: the base paper's year-number input is dropped (with ~2 training years it only extrapolates). Causal forward-fill of inputs over gaps (limit 15 d). Labels never imputed. Standard (z-score) scaling fit on the training period only; the base-paper FNN keeps its published min-max [0, 1] scaling (same days and features) | `data.py: make_features` | – |
| **IV-B. Models** | Equations for the Elman RNN, GRU (update/reset gates) and LSTM (input/forget/output gates); last hidden state → linear head. Baselines: persistence, the base paper's FNN exactly as published, FNN tuned with the same budget, Window-FNN | `models.py` | Model table with params (`n_params` in `metrics.csv`) |
| **IV-C. Training & Tuning** | Adam, MSE, gradient clipping 1.0, early stopping (patience 60) on validation RMSE. W&B Sweeps, Bayesian optimisation (cite Biewald 2020), **the same number of sweep runs for every tuned model**, and a search-space table; give the public sweep links as supplementary material | `train.py`, `tune.py: sweep_config` | `best_hp/*.json` → hyper-parameter table (L fixed at 15, not tuned) |
| **IV-D. Evaluation Protocol** | RMSE, MAE, R², ±10 % accuracy (base paper's metrics + MAE; MAPE avoided because H₂S is near 0 ppm on many days). 10 seeds → mean ± std. Diebold–Mariano test on seed-averaged predictions with a Newey–West (HAC) variance, because daily loss differentials are autocorrelated (lag-1 ACF ≈ 0.4); cite Newey & West 1994 for the bandwidth. All models scored on **the same test days** | `metrics.py`, `evaluate.py` | – |
| **V. Results** | Main table; which model wins, and by how much versus the base-paper FNN; p-values; test-period prediction plots; learning curves. Also compare with Schroer & Just's own 1-day-ahead MLP/ridge (MAPE ≈ 27 %) as published context | `report.py` | `results_table.tex`, `significance.csv`, `figures/test_*.pdf`, W&B learning curves |
| **VI. Ablations & Discussion** | (a) **Window-FNN**: same L-day input without recurrence, so is the gain from recurrence or just from more input? (b) window length: rerun with `--window 7` / `30` (main runs use L = 15, the base paper's longest lag). Why GRU vs LSTM. Limitations: one plant, one test year, biogas only (no CH₄/H₂S in the public data) | flags below | extra rows / small table |
| **VII. Conclusion** | One paragraph plus future work (multi-step horizons, other plants, control) | – | – |

## Ablation runs (each gets its own `--name`)

```bash
uv run adseq run-all --name main                                     # main table
uv run adseq run-all --name abl_window30 --window 30 \
    --models fnn_paper fnn_tuned window_fnn rnn gru lstm                   # (b) window length (FNNs re-scored on the same days)
```

## Reviewer-proofing checklist

- [ ] Base-paper FNN reproduced as published (`fnn_paper`) **and** tuned (`fnn`). Beating only an untuned baseline gets papers rejected.
- [ ] Equal sweep budget per model; state the run count and link the W&B sweeps.
- [ ] Persistence baseline included: a model that can't beat "tomorrow = today" isn't useful.
- [ ] mean ± std over 10 seeds plus DM p-values; don't claim a win where p > 0.05.
- [ ] Report where the FNN wins too. Honest mixed results are publishable; hidden ones are found by reviewers.
- [ ] No leakage: scalers fit on train only, past-only lags, chronological split (`tests/test_data.py`).
- [ ] Cite the Muscatine dataset (doi:10.25820/data.006715) and paper (Schroer & Just 2023); ODC-By needs attribution.
- [ ] Release code (GitHub link in the paper) for reproducibility.
