# Integration Note — B1

## What I produce

B1 — Score at Six produces the validated final handoff:

```text
results/pred_over6.csv
```

The current output contains one prediction row for each available
match/innings Over-6 checkpoint (769 rows in the current processed dataset).

### Output columns

| Column | Meaning | Unit |
|---|---|---|
| `match` | Match identifier | identifier |
| `innings` | Innings number | innings number |
| `predicted_score` | Predicted final innings score | runs |
| `low_estimate` | Lower prediction interval bound | runs |
| `high_estimate` | Upper prediction interval bound | runs |

`match + innings` is the row identity. The final handoff contains only these
five contract columns; model-specific fields are kept in the validation
artifacts rather than exposed to downstream consumers.

## What I consume

Primary input:

```text
data/processed/checkpoints.csv
```

B1 uses rows where:

```text
over_mark = 6
```

Relevant model inputs are:

```text
runs_so_far
wickets_down
current_run_rate
balls_since_boundary
batting_team
```

The supervised target is:

```text
final_score
```

## Final model and uncertainty

The final handoff pipeline uses Gradient Boosting. Final predictions are
validated with 10 match-level outer folds for each seed from 0 through 9.
A 90% split-conformal prediction interval is calculated from match-level
calibration residuals inside each training portion.

Current validation summary:

```text
Rows: 769
Empirical interval coverage: 90.77% (698/769)
Mean interval width: 71.85 runs
OOF MAE: 16.95 runs
OOF RMSE: 21.58 runs
```

These are validation measurements on the current dataset; they are not a
guarantee for future unseen matches.

## Downstream use

Downstream modules should:

1. Read `results/pred_over6.csv`.
2. Join rows using `match` and `innings`.
3. Use `predicted_score` as the point prediction.
4. Treat `low_estimate` and `high_estimate` as the validated prediction range
   produced by the B1 uncertainty pipeline.

## Contract history

During Week 5, a five-row dummy `pred_over6.csv` was used to establish and
test the interface. That artifact has now been replaced by the actual final
model output. The Week 5 contract columns remain unchanged.

## Assumptions that could break

1. The input checkpoint must represent the Over-6 state.
2. `match` and `innings` must uniquely identify a prediction row.
3. Source feature names in `checkpoints.csv` must remain unchanged.
4. Downstream consumers must preserve the five-column handoff contract.
5. The empirical coverage value applies to the current validation data and
   should not be interpreted as a guaranteed coverage rate for future data.

## Regeneration

From the repository root:

```bash
python src/train.py
```

This regenerates the final handoff and the validation artifacts in `results/`.
