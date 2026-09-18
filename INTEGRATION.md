# Integration Note — B1

## What I produce

B1 — Score at Six produces:

```text
results/pred_over6.csv
```

The output contains one row for each match/innings prediction at the Over-6
checkpoint.

### Output columns

| Column | Meaning | Unit |
|---|---|---|
| `match` | Match identifier | identifier |
| `innings` | Innings number | innings number |
| `predicted_score` | Predicted final innings score | runs |
| `low_estimate` | Lower prediction bound | runs |
| `high_estimate` | Upper prediction bound | runs |

The file is updated when a prediction is generated for a match/innings after
the Over-6 checkpoint. The final batch output will contain the required B1
predictions for the available Over-6 observations.

## What I consume

Primary input:

```text
data/processed/checkpoints.csv
```

B1 uses rows where:

```text
over_mark = 6
```

Relevant model inputs are the Over-6 checkpoint features:

```text
runs_so_far
wickets_down
current_run_rate
balls_since_boundary
batting_team
```

The target during supervised training is:

```text
final_score
```

## Downstream use

Downstream teams should use:

```text
match + innings
```

as the row identity and read the prediction fields from the same row.

`predicted_score` is the point prediction. `low_estimate` and `high_estimate`
represent the prediction range once the final uncertainty method has been
validated.

## Week 5 contract status

`results/pred_over6.csv` currently contains five **dummy rows** to establish
and test the interface. These values are placeholders and are not final B1
model results.

## Assumptions that could break

1. The input checkpoint must represent the Over-6 state.
2. `match` and `innings` must uniquely identify a prediction row.
3. The downstream consumer must not treat the Week 5 dummy values as model
   performance results.
4. Final interval bounds will only be considered valid after coverage
   evaluation.
5. Source column names in `checkpoints.csv` remain unchanged.

## Tested against

The contract was checked against the current processed checkpoint dataset:
769 rows, all at `over_mark = 6`, with no missing values in the checked
columns.

## Known incompatibilities

The Week 5 dummy output is an interface artifact only. It does not yet contain
validated final predictions or validated uncertainty intervals.
