# Architecture

## Foundation Layer

The foundation layer provides three cross-cutting concerns consumed by every module in the system.

### Configuration Flow

```
  .env (secrets: DAGSHUB_TOKEN, SUPABASE_URL, ...)
    │
    ▼
  APP_ENV variable (dev / staging / prod)
    │
    ▼
  config/base.yaml ──► deep_merge ──► config/{APP_ENV}.yaml
    │                                        │
    ▼                                        ▼
  config/features.yaml              config/thresholds.yaml
    │                                        │
    └──────────────┬─────────────────────────┘
                   ▼
          Settings (Pydantic BaseSettings)
               │
               ├── Validates all fields on load
               ├── Fails fast on missing secrets in staging/prod
               └── Exposed via get_settings() singleton
```

### Logging Architecture

```
  Any module
    │
    ├── get_logger(__name__)       → module-level logger
    ├── class Foo(LoggerMixin)     → self.logger with module.ClassName
    ├── @log_execution(op="...")   → START / SUCCESS / FAILURE + duration
    └── @log_errors(op="...")      → FAILURE only (lightweight)
    │
    ▼
  Three handlers (configured in config/logging.yaml):
    ├── Console  → colored (dev) or JSON (prod)
    ├── File     → logs/app.log  (INFO+, 10MB × 5 rotations)
    └── File     → logs/error.log (ERROR+, separate rotation)
```

Log line format:
```
2026-06-18 14:32:10 | INFO  | bnpl.models.train.ModelTrainer | fit | op=train_xgboost | Starting
```

### MLflow Tracking

```
  Training / Tuning / Retraining code
    │
    ▼
  with mlflow_run("run-name") as run:
    │
    ├── configure_tracking() → sets URI from config
    ├── get_or_create_experiment() → ensures experiment exists
    ├── Auto-tags: git_commit, config_env
    ├── Logs lifecycle through project logger
    └── On failure: sets FAILED tag, logs error, re-raises
    │
    ▼
  Backend: local mlruns/ (dev) or DagsHub (staging/prod)
```

## System Overview

```
  ┌─────────────┐     ┌──────────────┐
  │  LendingClub │     │   Supabase   │
  │   Dataset    │     │  PostgreSQL  │
  └──────┬──────┘     └──────▲───────┘
         │                    │
         ▼                    │ metrics/alerts
  ┌──────────────┐    ┌──────┴───────┐     ┌──────────────┐
  │  Data +      │    │  Monitoring  │     │  Streamlit   │
  │  Feature     │───►│  Pipeline    │────►│  Dashboard   │
  │  Pipeline    │    └──────────────┘     └──────────────┘
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    ┌──────────────┐
  │  Training +  │    │   FastAPI    │
  │  Tuning      │───►│   Serving    │
  │  Pipeline    │    │  (/predict)  │
  └──────────────┘    └──────────────┘
         │
         ▼
  ┌──────────────┐
  │   MLflow     │
  │  (DagsHub)   │
  └──────────────┘
```
