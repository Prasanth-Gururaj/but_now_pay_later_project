# BNPL Default Prediction System

A production-grade ML system that predicts loan defaults for Buy Now Pay Later (BNPL) transactions — covering the full lifecycle from data preparation through model serving, monitoring, A/B testing, and an operator dashboard. Built on the LendingClub Accepted Loans dataset (2007–2018, ~2.26M rows).

---

## Problem Statement

BNPL providers must make real-time credit approval decisions on millions of transactions daily, for a population that traditional credit scoring was never designed to serve. Most ML projects treat default prediction as a static modeling problem — train, evaluate, done. This project treats it as an **operational problem**, addressing three challenges that static modeling ignores:

1. **Delayed feedback** — loan outcomes aren't known for weeks/months after the approval decision, so standard label-dependent monitoring breaks down.
2. **Population drift** — borrower behavior shifts with economic conditions; a model trained on one period silently degrades as that period ends.
3. **Pipeline fragility** — upstream data sources change without warning, and naive drift detection cannot tell a broken pipeline from a real population shift.

**Problem type:** Supervised binary classification. `default = 1` if `loan_status == "Charged Off"`, `default = 0` if `loan_status == "Fully Paid"`. `grade`/`sub_grade` are used as *input features* (LendingClub's own risk segmentation), not as the prediction target — a probability output is strictly more useful since risk tiers can always be derived from it post-hoc, but not the reverse.

**Known limitation — selection bias:** The model is trained exclusively on *accepted* loans. LendingClub's own underwriting already filtered the applicant pool before this data was generated, so the model learns default patterns only within an already-approved population. This is the classic **reject inference** problem in credit risk.

---

## What This Project Solves

Rather than a one-off notebook model, this is a full ML system:

| Stage | What it does |
|-------|---------|
| **Data & Features** | Leakage-free temporal split, statistical feature selection (151 → 17 raw → 47 model-ready features), reproducible preprocessing config |
| **Training** | Champion/challenger model comparison (Logistic Regression, XGBoost, LightGBM), cost-based threshold selection, SHAP explainability |
| **Serving** | FastAPI `/predict` endpoint, <50ms inference, training-serving skew protection |
| **Monitoring** | Drift detection (Evidently AI) that distinguishes real population drift from broken data pipelines, business metrics tracking |
| **A/B Testing** | Statistically rigorous champion-vs-challenger evaluation (two-proportion z-test, chi-square test) before any model reaches production |
| **Dashboard** | Streamlit operator UI — live prediction, model performance, feature importance, drift monitoring, A/B test results |

---

## Model Performance — How Useful Is the Baseline?

| Metric | Validation (2016) | Test (2017, touched once) |
|--------|--------------------|----------------------------|
| **Model** | XGBoost (champion) | XGBoost (champion) |
| **AUC** | 0.7141 | 0.7046 |
| **Decision threshold** | 0.47 (cost-optimized, see below) | 0.47 |
| **Recall (catches defaulters)** | 61.3% | **71.3%** |
| **Precision** | 37.6% | 33.9% |
| **Approval rate** | — | **51.4%** |

The champion model catches **71.3% of actual defaulters** before approval on completely unseen 2017 data, while still approving just over half of all applicants — a deliberate trade-off, not an accident. The threshold (0.47) was selected by minimizing total business cost (below) under a constraint of at least 55% minimum approval rate, not by maximizing F1 or accuracy. AUC ~0.70-0.71 reflects a real ceiling: the model uses only application-time features (no post-origination data), and `sub_grade` alone accounts for ~70% of feature importance — the model is meaningfully refining LendingClub's own risk grading, not replacing it from scratch.

---

## Business Impact (Revenue Numbers)

Using the documented cost assumptions in `config/thresholds.yaml` — **$300** for approving a loan that defaults (false negative), **$45** in lost revenue for wrongly denying a good applicant (false positive) — the test-set confusion matrix at threshold 0.47 (derived from the stored precision/recall, cross-checked against the stored approval rate) is:

| | Predicted: Deny | Predicted: Approve |
|---|---|---|
| **Actual: Default** (39,148) | TP ≈ 27,926 | FN ≈ 11,222 |
| **Actual: Fully Paid** (130,152) | FP ≈ 54,395 | TN ≈ 75,757 |

- **Cost with the model**: (11,222 × $300) + (54,395 × $45) = **$5,814,375**
- **Cost with no model** (approve every applicant): 39,148 × $300 = **$11,744,400**
- **Net impact: ~$5.93M saved (≈50.5% reduction) on this single 169,300-loan test cohort** — while still approving 51.4% of applicants rather than shutting off all revenue.

> **Caveat:** the $300 / $45 figures are documented placeholder cost assumptions sized to typical BNPL transaction amounts, not audited unit economics. The dollar figure should be read as "the model cuts default-driven losses roughly in half relative to no screening at all," not as a verified P&L number.

---

## A/B Testing: Champion vs Challenger

To validate the champion stays the best choice, a LightGBM challenger was trained on the same data/features and run through a simulated 70/30 A/B test on the test set, with statistical significance testing (`reports/ab_test_results.json`):

| Metric | Champion (XGBoost) | Challenger (LightGBM) |
|--------|---------------------|------------------------|
| Threshold | 0.47 | 0.29 (re-derived; champion's threshold gave 100% approval on this model) |
| AUC | **0.7047** | 0.6916 |
| Approval rate | 55.4% | 52.8% |
| Sample size | 118,408 | 50,892 |

**Two-proportion z-test**: z = 9.71 (statistically significant approval-rate difference). **Recommendation: keep champion** — XGBoost outperforms the challenger on AUC and the difference is not in the challenger's favor. This is the system working as intended: a worse model was caught and rejected before reaching production, rather than promoted on vibes.

---

## Architecture

```
                     .env (secrets)
                          │
                          ▼
  base.yaml ──► deep_merge ──► {env}.yaml
                     │
                     ▼
               Settings (Pydantic)
               ╱        │        ╲
              ▼         ▼         ▼
          Logger    MLflow     App Modules
          Config    Config     ├── data/
                               ├── features/
                               ├── models/
                               ├── serving/
                               ├── monitoring/
                               └── pipelines/
```

| Layer | Purpose |
|-------|---------|
| **Config** (`config/`) | Environment-aware YAML + Pydantic settings with fail-fast validation |
| **Logger** (`src/bnpl/logger/`) | Structured logging with automatic module/class/function context |
| **Tracking** (`src/bnpl/tracking/`) | MLflow wrapper with DagsHub remote support |
| **Data** (`src/bnpl/data/`) | Loading, cleaning, validation, temporal splitting |
| **Features** (`src/bnpl/features/`) | Leakage removal, encoding, imputation pipeline |
| **Models** (`src/bnpl/models/`) | Training, tuning (Optuna), evaluation, SHAP explainability |
| **Serving** (`src/bnpl/serving/`) | FastAPI with `/predict` and `/health` endpoints, A/B router |
| **Monitoring** (`src/bnpl/monitoring/`) | Drift detection, business metrics, A/B statistical analysis |
| **Pipelines** (`src/bnpl/pipelines/`) | Orchestration entrypoints for training/monitoring/retraining |
| **Dashboard** (`dashboard/`) | Streamlit operator UI: predict, performance, drift, A/B testing |

---

## Quick Start

```bash
# 1. Create virtual environment (Python 3.12) and install
uv venv --python 3.12 npl
uv pip install --python npl/Scripts/python -e ".[dev,test]"

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your secrets (optional for dev)

# 3. Run the smoke test
uv run python scripts/smoke_test_foundation.py

# 4. Run the test suite
uv run pytest tests/ -v

# 5. Run the dashboard
streamlit run dashboard/app.py
```

---

## Project Structure

```
├── config/                 # YAML configs + Pydantic settings
├── src/bnpl/               # Main Python package
│   ├── logger/             # Structured logging system
│   ├── tracking/           # MLflow experiment tracking
│   ├── data/                # Data loading and preparation
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Training, tuning, evaluation
│   ├── serving/            # FastAPI prediction API + A/B router
│   ├── monitoring/         # Drift detection, business metrics, A/B analysis
│   └── pipelines/          # Orchestration entrypoints
├── dashboard/               # Streamlit dashboard (Predict, Performance,
│                            # Feature Importance, Drift Monitoring, A/B Testing, About)
├── tests/                   # Unit and integration tests
├── scripts/                 # CLI entrypoints (training, monitoring, A/B test)
├── data/                     # Data directories (gitignored)
├── models/                   # Model artifacts: champion + challenger (gitignored)
├── notebooks/                # Jupyter exploration notebooks
└── .github/workflows/        # CI/CD pipeline definitions
```

---

## Tech Stack

pandas/pyarrow · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · MLflow (DagsHub) · FastAPI · Pydantic · Docker · Evidently AI · Streamlit · GitHub Actions

---

## How It Was Built

Source data: [LendingClub Accepted Loans 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2.26M rows, 151 columns), filtered to 1,345,310 resolved loans (80% Fully Paid / 20% Charged Off). Post-issuance columns (`recoveries`, `total_pymnt`, etc.) were proven to leak the target via correlation analysis (0.7–0.85+ vs. 0.05–0.15 for legitimate features) and removed. Statistical feature selection (ANOVA, mutual information, VIF, permutation importance), run on the training period only, took the column count from 151 raw down to 17 legitimate application-time features. A temporal split (train 2013–2015, validation 2016, test 2017) keeps the evaluation honest, with 2018 and 2008–2009 windows held out separately for drift simulation. Every preprocessing parameter (imputation medians, encoding maps, outlier caps) is fit on the training split only, expanding the 17 raw features to 47 model-ready columns. Three models were compared (Logistic Regression, XGBoost, LightGBM); XGBoost was selected as champion on validation AUC, and its decision threshold was chosen by minimizing total business cost under a minimum-approval-rate constraint, not by maximizing F1. Full step-by-step detail lives in `notebooks/01_data_exploration.ipynb` through `04_model_training.ipynb`.

---

## Known Limitations

**Selection bias:** Model trained on accepted loans only. LendingClub's own underwriting filtered the applicant pool before this data was generated — the classic reject inference problem.

**Calibration:** The model overestimates default probability; the calibration curve sits below the diagonal throughout. Platt scaling has not been applied.

**Sub_grade dominance:** ~70% of feature importance comes from LendingClub's own `sub_grade` field. The model primarily refines an existing expert system rather than learning entirely fresh signal from raw financial features.

**Label delay:** BNPL loan outcomes take ~6 weeks to materialize. Monitoring uses proxy signals (approval rate shifts, feature drift) rather than final default labels, introducing uncertainty into drift-detection timing.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Class imbalance, time axis, leakage proof, missing values, distributions |
| `02_feature_selection.ipynb` | Statistical feature selection on training period only — 151 → 17 features |
| `03_data_preparation.ipynb` | Temporal split, imputation, encoding (17 → 47 cols), outlier capping |
| `04_model_training.ipynb` | Baseline vs XGBoost vs LightGBM, MLflow tracking, cost-based threshold selection |
