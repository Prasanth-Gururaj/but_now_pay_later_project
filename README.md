# BNPL Default Prediction System
 
A production-grade ML system that predicts loan defaults for Buy Now Pay Later (BNPL) transactions. Built on the LendingClub Accepted Loans dataset (2007-2018, ~2.26M rows), the system delivers a binary credit-risk model (Charged Off vs Fully Paid, ~20% positive class) with the full production lifecycle: model serving via FastAPI, monitoring and drift detection with Evidently AI, a Streamlit operator dashboard, and CI/CD-triggered retraining through GitHub Actions.
 
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
| **Serving** (`src/bnpl/serving/`) | FastAPI with /predict and /health endpoints |
| **Monitoring** (`src/bnpl/monitoring/`) | Drift detection, business metrics, alert classification |
| **Pipelines** (`src/bnpl/pipelines/`) | Orchestration entrypoints for training/monitoring/retraining |
| **Dashboard** (`dashboard/`) | Streamlit operator UI for drift reports and business metrics |
 
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
```
 
## Project Structure
 
```
├── config/                 # YAML configs + Pydantic settings
├── src/bnpl/              # Main Python package
│   ├── logger/            # Structured logging system
│   ├── tracking/          # MLflow experiment tracking
│   ├── data/              # Data loading and preparation
│   ├── features/          # Feature engineering pipeline
│   ├── models/            # Training, tuning, evaluation
│   ├── serving/           # FastAPI prediction API
│   ├── monitoring/        # Drift detection and metrics
│   └── pipelines/         # Orchestration entrypoints
├── dashboard/             # Streamlit monitoring frontend
├── tests/                 # Unit and integration tests
├── scripts/               # CLI entrypoints and utilities
├── data/                  # Data directories (gitignored)
├── models/                # Model artifacts (gitignored)
├── notebooks/             # Jupyter exploration notebooks
├── docker/                # Dockerfiles for API and dashboard
└── .github/workflows/     # CI/CD pipeline definitions
```
 
## Configuration
 
Settings are environment-aware via the `APP_ENV` variable (`dev` | `staging` | `prod`):
 
- **`config/base.yaml`** — shared defaults
- **`config/{env}.yaml`** — environment overrides (deep-merged over base)
- **`.env`** — secrets only (never committed)
Load settings anywhere:
```python
from config.settings import get_settings
settings = get_settings()
```
 
## Development
 
```bash
make lint      # ruff + black check
make format    # auto-fix lint issues
make test      # pytest
make smoke     # foundation smoke test
```
 
## Tech Stack
 
pandas/pyarrow · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · MLflow (DagsHub) · FastAPI · Pydantic · Docker · Render.com · Evidently AI · Streamlit · GitHub Actions · Supabase/PostgreSQL
 
---
 
## Problem Statement
 
BNPL providers must make real-time credit approval decisions on millions of transactions daily, for a population that traditional credit scoring was never designed to serve. Existing research treats default prediction as a static modeling problem — train, evaluate, done. This project treats it as an **operational problem**, addressing three challenges that static modeling ignores:
 
1. **Delayed feedback** — loan outcomes aren't known for weeks/months after the approval decision, so standard label-dependent monitoring breaks down.
2. **Population drift** — borrower behavior shifts with economic conditions; a model trained on one period silently degrades as that period ends.
3. **Pipeline fragility** — upstream data sources change without warning, and naive drift detection cannot tell a broken pipeline from a real population shift.
**Problem type:** Supervised binary classification. `grade`/`sub_grade` are used as *input features* (LendingClub's own expert-system risk segmentation), not as the prediction target — a probability output is strictly more useful since risk tiers can always be derived from it post-hoc (binning), but not the reverse. Multi-class risk-tier prediction was considered and deliberately rejected for this reason.
 
**Target variable:** `default = 1` if `loan_status == "Charged Off"`, `default = 0` if `loan_status == "Fully Paid"`. All other statuses (`Current`, `Late`, `In Grace Period`) are dropped — their final outcome is not yet resolved.
 
**Known limitation — selection bias:** The model is trained exclusively on *accepted* loans. LendingClub's own underwriting already filtered the applicant pool before this data was generated, so the model learns default patterns only within an already-approved population. This is the classic **reject inference** problem in credit risk — `rejected_2007_to_2018Q4.csv.gz` is retained in `data/raw/` for a possible future selection-bias analysis but is **not** used in training (no outcome label exists for rejected applicants).
 
---
 
## Progress Log
 
### ✅ Phase 0 — Problem Framing & Domain Research
 
- Defined the problem statement above and documented the three operational challenges that differentiate this from a typical Kaggle-style credit risk project.
- Read external research (BNPL default prediction literature) to ground the class-imbalance handling and ensemble model choice in established practice rather than guesswork.
- Decided: **supervised binary classification**, not multi-class risk tiering.
### ✅ Phase 1 — Data Acquisition & Initial EDA (`01_data_exploration.ipynb`)
 
- Source: [`kaggle.com/datasets/wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — `accepted_2007_to_2018Q4.csv.gz` (2.26M rows, 151 columns).
- **Class imbalance confirmed:** after filtering to resolved loans (1,345,310 rows), exactly **80.0% Fully Paid / 20.0% Charged Off** — a moderate (~4:1) imbalance. Not in the "severe" category (e.g. fraud detection at 99.8/0.2), but significant enough that accuracy is a misleading metric: a model predicting "always Fully Paid" scores 80% accuracy while catching zero defaults.
- **Leakage proven empirically, not just asserted:** correlation of post-issuance columns (`total_pymnt`, `recoveries`, etc.) against `default` measured at **0.7–0.85+**, versus **0.05–0.15** for legitimate application-time features. Leakage columns fall into three patterns:
  - Only exist *after* default (`recoveries`, `hardship_*`, `settlement_*`)
  - Accumulate *during* repayment (`total_pymnt`, `last_pymnt_d`)
  - Checked *after* loan issuance (`last_fico_range_low`, `last_credit_pull_d`)
- **Time axis (`issue_d`) explored:** confirmed 11 years of coverage (2007–2018) including the 2008 financial crisis — basis for the temporal split and later drift simulation windows.
- Column funnel established: **151 raw → ~100 dropped on structural grounds → ~25 legitimate candidate features** entering statistical feature selection.
### ✅ Phase 2 — Statistical Feature Selection (`02_feature_selection.ipynb`)
 
Ran **entirely on the training period (2013–2015) only** — feature selection itself can leak information from val/test if run on the full dataset.
 
Methods run (statistical evidence, not business judgment alone): ANOVA F-test + Chi-Square (univariate signal) → Mutual Information (nonlinear relationships) → VIF + pairwise correlation (redundancy) → baseline XGBoost importance + permutation importance (real-world signal including interactions, with permutation importance specifically used to counter XGBoost's cardinality bias).
 
**Key findings:**
- `fico_range_low` / `fico_range_high` perfectly correlated (r = 1.000) — duplicate.
- `loan_amnt` / `funded_amnt` perfectly correlated (r = 1.000) — on this platform, requested ≈ funded almost always.
- `installment` mathematically derived from `loan_amnt` + `term` + `int_rate` (r = 0.952 with `loan_amnt`) — redundant with its own ingredients.
- `grade` + `sub_grade` jointly account for **~70% of baseline model feature importance** — the model leans heavily on LendingClub's existing expert-grading system. Documented as a finding, not hidden.
- `addr_state` weakest feature across every method tested — dropped.
- `revol_util` showed a disagreement between domain intuition (expected top feature) and statistical evidence (ranked low in this diagnostic) — flagged for investigation rather than resolved by assumption.
- Baseline (untuned) XGBoost on surviving features: **AUC ≈ 0.732** — believable, non-suspicious, indirectly confirming Phase 1 leakage removal was effective (vs. AUC ≈ 0.97 observed when leakage columns were deliberately included as a sanity check).
**Final feature set — 17 features, locked:**
 
| Kept (17) | Dropped (6) | Reason for drop |
|---|---|---|
| `dti`, `fico_range_low`, `revol_util`, `annual_inc`, `loan_amnt`, `int_rate`, `sub_grade`, `term`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal` | `fico_range_high` | Perfect duplicate of `fico_range_low` |
| | `funded_amnt` | Perfect duplicate of `loan_amnt` |
| | `installment` | Derived from features already kept |
| | `grade` | Redundant with more granular `sub_grade` |
| | `addr_state` | Weakest signal across every method |
| | `total_acc` | Weak signal, correlated with `open_acc` |
 
Plus `issue_d` (time axis, not a model input) and `default` (target). Evidence table saved to `reports/feature_selection_table.csv`.
 
### ✅ Phase 3 — Data Preparation (`03_data_preparation.ipynb`)
 
Principle enforced throughout: **every transformation parameter (median, encoding map, outlier cap) is calculated on the TRAIN split only**, then applied unchanged to val/test — never recalculated on data the model shouldn't have "seen" yet. Same logic as the temporal split itself, applied one level deeper.
 
**1. Row filtering + target.** Resolved loans only → 1,345,310 rows. Confirmed 80.0% / 20.0% class balance.
 
**2. Temporal split.** Train: 2013–2015, Val: 2016, Test: 2017. 2018 and 2008–2009 windows held out separately, untouched, reserved for drift simulation in a later phase. Class balance verified consistent (~80/20) across all three splits.
 
**3. Missing values.** Per-feature median imputation (train-fit) for numeric features, with a `_was_missing` binary flag added alongside each — the absence itself can carry signal, not just the imputed value.
 
**4. `emp_length` conversion.** Text (`"5 years"`, `"< 1 year"`, `"10+ years"`) mapped to a numeric 0–10 scale. This is ordinal information stored as text — mapping preserves the natural "more years = more stable" ordering, which a model can use directly, unlike treating each value as an unrelated category.
 
**5. Categorical encoding.**
- `sub_grade` → **ordinal** encoding (A1=0 … G5=34), since the categories have a genuine risk order.
- `term` → numeric extraction (36 / 60).
- `home_ownership`, `verification_status`, `purpose` → **one-hot encoded**, with the category list **fixed from the training split**. Any category appearing in val/test that wasn't seen in train maps to an all-zero row across that feature's dummy columns rather than crashing — this mirrors real production behavior when a genuinely new category appears post-deployment (training-serving skew protection).
- Considered and explicitly rejected: TF-IDF (wrong data shape — these are single categorical labels, not free text with word frequency to compute) and entity embeddings (only justified for high-cardinality features in neural architectures; at 3–14 categories with a tree-based model, one-hot encoding is the standard, defensible, easily-interpretable choice — embeddings here would add complexity with no measurable benefit).
**6. Outlier / range sanity checks** — caught real data quality issues, not just statistical tail behavior:
  - `annual_inc`: 99th percentile cap (train-fit) = $250,000 → 6,661 rows capped (0.91%)
  - `revol_bal`: 99th percentile cap (train-fit) = $93,283 → 7,335 rows capped (1.00%)
  - `revol_util`: observed max of **892.3** on a feature that is mathematically bounded 0–100 (a utilization percentage) — this is a data error, not a real value. Applied a **hard business-logic cap at 100**, not a percentile cap → 3,064 rows capped (0.42%)
  - `dti`: observed max of **999** — a near-certain placeholder/error value (no real borrower has debt payments at 999% of income). 99th percentile cap (train-fit) = 37.57 → 7,302 rows capped (1.00%). The cap landing well below the apparent outliers (75th percentile was only 24.21) confirms these were isolated error values, not a genuinely fat right tail.
**7. Class imbalance strategy — documented, applied at training time.** Train split: ~80/20 → `scale_pos_weight ≈ 4.0`. Chose **class weighting** over SMOTE: natively supported by XGBoost, no synthetic data interpretability concerns, simpler to defend. (If SMOTE is reconsidered later, it must be fit on the train split only, after splitting — applying it pre-split would leak synthetic examples influenced by val/test distributions.)
 
**8. Final leakage re-check.** Re-ran the Phase 1 correlation methodology on the final processed feature set — all values within the expected modest range, no red flags.
 
**9. Feature count: 17 → 47 after encoding.** This is expected expansion, not feature bloat — no new *information* was added, only format changes required for tree-based modeling:
  - One-hot encoding of 3 low-cardinality categoricals: `purpose` (14 categories, +13 columns), `home_ownership` (4 categories, +3), `verification_status` (3 categories, +2) → **+18 columns**
  - `_was_missing` flag columns added for 12 imputed numeric features → **+12 columns**
  - 47 columns against ~733,000 training rows is a healthy ratio (~15,600 rows/column) — nowhere near the dimensionality concerns that would justify a different encoding strategy (those concerns start mattering in the hundreds/thousands of categories range, e.g. merchant IDs or ZIP codes — not applicable here).
**10. Saved outputs:**
- `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- `reports/data_prep_config.json` — full reproducibility record: every imputation value, encoding map, outlier cap, and split boundary, so any transformation can be explained or re-applied without relying on memory.

### ✅ Phase 4 — Model Training & Evaluation (`04_model_training.ipynb`)

- Trained three models: Logistic Regression (AUC 0.707), XGBoost (AUC **0.714**, champion), LightGBM (AUC 0.705).
- All runs logged to DagsHub MLflow with parameters, metrics, and model artifacts.
- Threshold selected via business cost optimization: **0.47** (minimum cost with >=55% approval rate).
- SHAP analysis confirmed `sub_grade` dominates at ~70% of feature importance.
- Champion model registered in MLflow model registry on DagsHub.

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Champion model** | XGBoost |
| **Training period** | 2013-2015 (733,451 loans, 18.81% default rate) |
| **Validation period** | 2016 (293,095 loans, 23.28% default rate) |
| **Test period** | 2017 (169,300 loans, 23.12% default rate) |
| **Validation AUC** | 0.7141 |
| **Decision threshold** | 0.47 |
| **Threshold method** | Minimum total business cost with 55% minimum approval rate constraint |
| **Cost assumptions** | Approving a defaulter: $300 loss. Rejecting a good applicant: $45 lost revenue |

> **Note:** 2016 has the highest default rate in the dataset at 23.28%, making this a genuinely stress-tested validation set.

---

## 60-Second Interview Answer

I built a BNPL default prediction system that addresses three operational problems most ML projects ignore. First, the model trains on 2013-2015 data at 18.81% default rate and validates on 2016 which had a 23.28% default rate, the highest in the dataset, making the validation genuinely stress-tested rather than optimistically easy. Second, I chose threshold 0.47 using business cost optimization rather than F1 maximization, balancing a minimum 55% approval rate against default loss costs of 300 dollars per bad loan approved. Third, I distinguish real population drift from broken data pipelines before triggering any retraining, because those two situations need completely different responses. The system has a 15-command CLI, A/B testing infrastructure for safe model updates using a 90/10 champion challenger split with counterfactual logging, and CI/CD that runs daily monitoring and deploys automatically when a better model is confirmed.

---

## Known Limitations

**Selection bias:** Model trained on accepted loans only. LendingClub's own underwriting filtered the applicant pool before this data was generated. The model learns default patterns only within an already-approved population, not the full applicant pool. This is the classic reject inference problem.

**Calibration:** The model overestimates default probability. The calibration curve sits below the diagonal throughout, meaning predicted probabilities are systematically higher than actual default rates. Platt scaling calibration has not been applied.

**Sub_grade dominance:** Approximately 70 percent of feature importance comes from LendingClub's own sub_grade field, which encodes their internal risk assessment. The model primarily refines an existing expert system rather than learning entirely fresh signal from raw financial features.

**Label delay:** BNPL loan outcomes take 6 weeks to materialize. Monitoring uses proxy signals such as early delinquency indicators rather than final default labels. This introduces uncertainty into drift detection timing.

---

## Notebooks

| Notebook | Status | Purpose |
|---|---|---|
| `01_data_exploration.ipynb` | ✅ Complete | Class imbalance, time axis, leakage proof, missing values, distributions |
| `02_feature_selection.ipynb` | ✅ Complete | Statistical feature selection on training period only — 151 to 17 features |
| `03_data_preparation.ipynb` | ✅ Complete | Temporal split, imputation, encoding (17 to 47 cols), outlier capping, save processed data |
| `04_model_training.ipynb` | ✅ Complete | Baseline vs XGBoost vs LightGBM, MLflow tracking, cost-based threshold selection |