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
# 1. Create virtual environment and install
uv venv .venv
uv pip install -e ".[dev,test]"

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
