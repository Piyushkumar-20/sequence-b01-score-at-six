# B1 Literature Sources

## Research Papers

These are the 10 papers reviewed for **B1 — Score at Six**.  
The papers are existing research by other authors; our team is reviewing them to understand previous approaches and identify methods that can be applied to B1.

### 1. Duckworth & Lewis (1998)

**Paper:** *A Fair Method for Resetting the Target in Interrupted One-Day Cricket Matches*  
**Authors:** F. C. Duckworth, A. J. Lewis  
**Relevance:** Cricket scoring/resource model; useful background for understanding the relationship between overs remaining, wickets and scoring potential.

- [Official paper — Taylor & Francis](https://doi.org/10.1057/palgrave.jors.2600524)
- [DBLP bibliographic record](https://dblp.org/rec/journals/jors/DuckworthL98.html)

### 2. Bailey & Clarke (2006)

**Paper:** *Predicting the Match Outcome in One Day International Cricket Matches, While the Game is in Progress*  
**Authors:** Michael J. Bailey, Stephen R. Clarke  
**Relevance:** In-progress cricket prediction using match information and multiple linear regression.

- [Full paper — PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3861745/)
- [Journal of Sports Science and Medicine](https://www.jssm.org/mobile/abstmobile.php?id=jssm-05-480.xml)

### 3. Singh, Singla & Bhatia (2015)

**Paper:** *Score and Winning Prediction in Cricket Through Data Mining*  
**Authors:** Tejinder Singh, Vishal Singla, Parteek Bhatia  
**Relevance:** Cricket score/winning prediction using data-mining approaches.

- [IEEE DOI / Paper record](https://doi.org/10.1109/ICSCTI.2015.7489605)
- [Bibliographic record — ResearchGate](https://www.researchgate.net/publication/305284754_Score_and_winning_prediction_in_cricket_through_data_mining)

### 4. Tonmoy et al. (2024)

**Paper:** *A Data-Driven Approach to Predict Scores in T20 Cricket Match Using Machine Learning Classifier*  
**Authors:** Md. All Shahoriar Tonmoy, Samrat Kumar Dey, Tania Islam, Jakaria Apu  
**Relevance:** Directly related to T20 first-innings score prediction; compares XGBoost, Lasso and Ridge regression.

- [Springer DOI / Paper record](https://doi.org/10.1007/978-981-99-8937-9_49)
- [ResearchGate paper record](https://www.researchgate.net/publication/379414836_A_Data-Driven_Approach_to_Predict_Scores_in_T20_Cricket_Match_Using_Machine_Learning_Classifier)

### 5. Robberechts, Van Haaren & Davis (2021)

**Paper:** *A Bayesian Approach to In-Game Win Probability in Soccer*  
**Authors:** Pieter Robberechts, Jan Van Haaren, Jesse Davis  
**Relevance:** In-game prediction from evolving match-state information and evaluation of probability calibration.

- [ACM DOI](https://doi.org/10.1145/3447548.3467194)
- [Full paper PDF](https://www.janvanhaaren.be/assets/papers/kdd-2021-win-probability.pdf)
- [arXiv version](https://arxiv.org/abs/1906.05029)

### 6. Pelechrinis (2017)

**Paper:** *iWinRNFL: A Simple, Interpretable & Well-Calibrated In-Game Win Probability Model for NFL*  
**Author:** Konstantinos Pelechrinis  
**Relevance:** In-game sports prediction, calibration, and the comparison of simple and more complex models.

- [arXiv — Full paper](https://arxiv.org/abs/1704.00197)

### 7. Chen & Guestrin (2016)

**Paper:** *XGBoost: A Scalable Tree Boosting System*  
**Authors:** Tianqi Chen, Carlos Guestrin  
**Relevance:** Foundational work for XGBoost/tree boosting, relevant to the Gradient Boosting approach considered for B1.

- [ACM official paper](https://doi.org/10.1145/2939672.2939785)
- [arXiv — Full paper](https://arxiv.org/abs/1603.02754)
- [KDD paper PDF](https://kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)

### 8. Meinshausen (2006)

**Paper:** *Quantile Regression Forests*  
**Author:** Nicolai Meinshausen  
**Relevance:** Estimation of conditional quantiles for regression, relevant to producing lower and upper score estimates.

- [JMLR official paper page](https://www.jmlr.org/papers/v7/meinshausen06a.html)
- [Direct PDF — JMLR](https://jmlr.csail.mit.edu/papers/volume7/meinshausen06a/meinshausen06a.pdf)

### 9. Romano, Patterson & Candès (2019)

**Paper:** *Conformalized Quantile Regression*  
**Authors:** Yaniv Romano, Evan Patterson, Emmanuel J. Candès  
**Relevance:** Prediction intervals using conformal prediction and quantile regression, relevant to B1's uncertainty estimation and coverage evaluation.

- [NeurIPS official paper](https://papers.nips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- [arXiv — Full paper](https://arxiv.org/abs/1905.03222)

### 10. Tyralis & Papacharalampous (2024)

**Paper:** *A Review of Predictive Uncertainty Estimation with Machine Learning*  
**Authors:** Hristos Tyralis, Georgia Papacharalampous  
**Relevance:** Review of predictive uncertainty methods, including probabilistic prediction, quantile regression, random forests, boosting and related evaluation approaches.

- [Springer official paper](https://doi.org/10.1007/s10462-023-10698-8)
- [Springer full article](https://link.springer.com/article/10.1007/s10462-023-10698-8)

---

## Backup Datasets

These datasets are **backup/alternative sources only**. The primary B1 dataset remains the department-provided SRL simulated T20 dataset.

### 1. Cricsheet

**Purpose:** Public cricket ball-by-ball dataset that can be used as a backup source for cricket/T20 analysis.

Cricsheet provides structured ball-by-ball data for many cricket formats and competitions, including T20 internationals and the IPL.

- [Cricsheet official website](https://cricsheet.org/)
- [Cricsheet match-data formats](https://cricsheet.org/format/)

### 2. Indian Premier League (IPL) Ball-by-Ball Dataset

**Purpose:** Public T20/IPL ball-by-ball dataset that can be used as a backup for score-prediction experiments.

This Kaggle dataset contains IPL ball-by-ball data from 2008 through 2023 and was derived from Cricsheet data.

- [Kaggle — Indian Premier League (IPL) Ball-by-Ball Data](https://www.kaggle.com/datasets/jamiewelsh2/ball-by-ball-ipl)

---

## Methodology Decisions Informed by the Literature

The literature review supports evaluating:

1. **Current Run Rate × 20** — simple baseline.
2. **Linear Regression** — interpretable regression benchmark.
3. **Gradient Boosting / tree-based regression** — captures nonlinear relationships in tabular match-state data.
4. **Prediction intervals / uncertainty estimation** — provides low and high estimates instead of only a point prediction.
5. **Coverage and interval width** — evaluates whether the uncertainty intervals are both reliable and useful.

The final method will be selected based on empirical performance rather than model complexity alone.

---

