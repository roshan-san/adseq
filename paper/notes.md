# Paper notes

Brainstorm here: decisions, outline, key numbers, open questions. Short bullets only.
Final sentences go in `main.tex`, never in both places.

## Decided
- Data: public Muscatine WRRF (2020–2023), target = daily biogas (ft³/min)
- Test year 2022-04-01 to 2023-03-31; earlier days split 80/20 train/val by time
- Min-max scaling to [0, 1] for every model, fit on training period only
- Models: persistence, base-paper FNN, tuned FNN, window FNN, RNN, GRU, LSTM
- Equal budget: 40 W&B sweep runs per tuned model; 10 seeds each for evaluation
- DM test with HAC variance, one-sided p < 0.05

## Open questions
- Say why min-max (lower validation error than z-score in earlier runs)?

## I. Introduction

## II. Related Work

## III. Data

## IV. Methodology

## V. Results

## VI. Discussion

## VII. Conclusion
