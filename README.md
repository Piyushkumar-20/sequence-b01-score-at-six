# B1 — Score at Six

## Project Overview

**Track:** B — Sequence and Uncertainty
**Module:** B1 — Score at Six
**Type:** Regression — Foundation Difficulty

The objective of this project is to build a model that uses the state of a T20 innings after the first six overs to predict the final score of that innings.

The project uses the department's **Simulated T20 League (SRL)** dataset containing 800+ matches. The dataset is generated ball by ball and is used consistently across the Sequence and Uncertainty track.

The final module will produce a predicted final score together with a **low estimate and high estimate**, providing an uncertainty range around the prediction. This output is intended to be consumed by other modules in the track.

---

## Team

| Member   | Role       |
| -------- | ---------- |
| Member 1 | Data       |
| Member 2 | Model      |
| Member 3 | Evaluation |
| Member 4 | Writing    |

All team members are expected to contribute to the repository and understand the complete project pipeline.

---

## Dataset

The project uses the three shared source files provided by the department:

```text
matches.csv
deliveries.csv
checkpoints.csv
```

For this module, the primary input is `checkpoints.csv`. It contains one row per match/innings/over-mark snapshot with:

```text
match_id
innings
over_mark
runs_so_far
wickets_down
current_run_rate
balls_since_boundary
final_score
```

The B1 module works specifically with the **over-6 checkpoint** to predict `final_score`.

### Data Rules

* Raw data will be stored in `data/raw/`.
* Raw files must not be modified.
* Processed data will be stored in `data/processed/`.
* Original column names must be preserved.
* Dates must follow `YYYY-MM-DD`.
* Missing values are represented by empty cells.
* IDs such as `M0412` must retain their prefixes and padding.

---

## Project Roadmap

The planned modelling progression is:

```text
Over-6 Match State
        ↓
Baseline
Current Run Rate × 20
        ↓
Feature Engineering
        ↓
Linear Regression
        ↓
Gradient Boosting
        ↓
Model Comparison
        ↓
Prediction Range
        ↓
Coverage / Uncertainty Evaluation
        ↓
Final Output
```

The project will first establish a simple baseline using the current run rate, then compare it with more advanced regression approaches.

---

## Weekly Plan

### Week 1 — Setup

* Create the shared repository.
* Set up the required repository structure.
* Add team members and roles.
* Ensure all four members clone the repository and make a contribution.

### Week 2 — Understand the Data

No modelling will be performed.

The team will:

* Load and inspect the provided datasets.
* Check shapes, columns and data types.
* Analyse missing values.
* Explore score distributions and over-by-over patterns.
* Identify unusual observations and data issues.
* Record five important observations from the dataset.

The main deliverable is:

```text
notebooks/01_exploration.ipynb
```

### Week 3 — Literature Review

The team will:

* Review 8–10 relevant research papers.
* Record the methods, datasets and useful ideas from each paper.
* Identify two public cricket or sports-analytics datasets as backup sources.
* Decide which modelling methods will be tested and why.

The literature review will later contribute to the project paper.

---

## Repository Structure

```text
sequence-b01-score-at-six/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── results/
│   ├── plots/
│   ├── pred_over6.csv
│   └── README.md
│
├── experiments.csv
├── INTEGRATION.md
│
└── paper/
```

The repository follows the standard structure specified for all modules in the track.

---

## Planned Output

The final B1 output will be stored in:

```text
results/pred_over6.csv
```

Expected columns:

```text
match
innings
predicted_score
low_estimate
high_estimate
```

This output will allow other modules to use the B1 predictions and uncertainty estimates.

---

## Experiment Tracking

All experiments will be recorded in:

```text
experiments.csv
```

Each experiment will contain:

```text
run_id
date
who
what_changed
main_metric
value
seed
notes
```

Randomised experiments will use controlled seeds and multiple runs, with the average and spread reported rather than relying on a single run.

---

## Evaluation Principles

Every reported model result will be compared against an appropriate baseline.

The project will report:

* Baseline performance
* Model performance
* Spread across multiple seeds
* Failure cases
* Prediction interval coverage

The project will avoid reporting model performance without a baseline or relying on a single random run.

## Collaboration

This is a shared team repository. All members can contribute to any part of the project.

### First-Time Setup

If you are joining the project for the first time, clone the repository:

```bash
git clone https://github.com/Piyushkumar-20/sequence-b01-score-at-six.git
```

Then open the project folder:

```bash
cd sequence-b01-score-at-six
```

Check the repository:

```bash
git status
```

### Working on the Project

Before starting your work, get the latest changes:

```bash
git pull
```

Make your changes, then add them:

```bash
git add .
```

Commit your changes with a meaningful message:

```bash
git commit -m "describe the change"
```

Finally, push your changes:

```bash
git push
```

### Important

* Clone the repository only once.
* Always run `git pull` before starting new work.
* Use meaningful commit messages.
* All team members can contribute to any relevant part of the project.


## Current Status

**Week 1:** Repository setup
**Week 2:** Data exploration — waiting for department dataset
**Week 3:** Literature review

---

## Reference

This project follows the requirements and workflow described in:

**B1 · Score at Six — Guidebook, Track B · Sequence and Uncertainty**
