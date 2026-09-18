# B1 — Final Over-6 Output

## Purpose

This directory contains the validated B1 — Score at Six model outputs. The
final handoff file is `pred_over6.csv`, which contains one actual model
prediction for each available match/innings Over-6 checkpoint.

The Week 5 dummy output was used only for the interface handshake. It has been
replaced by the validated final prediction output described below.

## Final Output

```text
results/pred_over6.csv
```

Each row represents one match/innings prediction made from the Over-6
checkpoint. The current dataset produces 769 rows.

### Columns

| Column | Meaning | Unit |
|---|---|---|
| `match` | Match identifier corresponding to the source checkpoint | identifier |
| `innings` | Innings number for the prediction | innings number |
| `predicted_score` | Predicted final innings score | runs |
| `low_estimate` | Lower bound of the prediction interval | runs |
| `high_estimate` | Upper bound of the prediction interval | runs |

The output contract is intentionally limited to these five columns so that
downstream modules can consume the handoff without depending on internal
model fields.

## Input

The B1 pipeline consumes Over-6 rows from:

```text
data/processed/checkpoints.csv
```

with:

```text
over_mark = 6
```

The supervised target is `final_score`. The model features are:

```text
runs_so_far
wickets_down
current_run_rate
balls_since_boundary
batting_team
```

## Final Prediction Method

The final prediction pipeline uses Gradient Boosting. Predictions are generated
out-of-fold using 10 match-level outer folds for each seed from 0 through 9.
Within each outer development set, a match-level calibration split is used for
the uncertainty calculation.

The prediction range uses a **90% split-conformal prediction interval** based
on absolute calibration residuals. The current validation run reports:

- Empirical interval coverage: **90.77%** (698 of 769 rows)
- Mean interval width: **71.85 runs**
- Median interval width: **71.69 runs**
- OOF point MAE: **16.95 runs**
- OOF point RMSE: **21.58 runs**

These figures describe the current validation output and should not be treated
as a guarantee for future unseen matches.

## Validation Files

Additional final-validation artifacts are:

```text
results/final_prediction_validation.csv
results/prediction_interval_coverage.csv
```

`final_prediction_validation.csv` retains the actual score and coverage flag
for validation. `prediction_interval_coverage.csv` records the aggregate
coverage and interval-width summary.

## Regeneration

From the repository root, run:

```bash
python src/train.py
```

The command regenerates the model evaluation outputs and the final B1 handoff
files under `results/`.

## Assumptions and Limitations

1. Each prediction corresponds to one match/innings at the Over-6 checkpoint.
2. `predicted_score`, `low_estimate`, and `high_estimate` are expressed in runs.
3. `low_estimate <= predicted_score <= high_estimate`.
4. Downstream modules should join predictions using `match` and `innings`.
5. The reported 90% coverage is empirical validation coverage on the current
   dataset, not a guarantee of 90% coverage on every future dataset.
6. The final handoff is based on the available processed Over-6 checkpoint
   observations in `data/processed/checkpoints.csv`.
