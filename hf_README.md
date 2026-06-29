---
title: BNPL Default Prediction
emoji: 💳
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# BNPL Default Prediction

Predicts the probability that a Buy Now Pay Later loan applicant will default, using an XGBoost model trained on 1.35M LendingClub loans (2007–2018).

## What it does

Enter 17 loan application fields (income, FICO score, loan amount, etc.) and get an instant **APPROVE** or **DENY** decision with a calibrated default probability.

## Pages

- **Predict** — Interactive form with sample risk profiles (Low / Medium / High)
- **Model Performance** — AUC, F1, precision, recall, ROC curve, calibration plot
- **Feature Importance** — SHAP summary of the 47 engineered features
- **Monitoring** — Drift detection status (demo mode)
- **About** — Model card, dataset details, known limitations

## Architecture

```
Raw Input (17 fields)
  → PreprocessingPipeline (47 features)
  → XGBoost predict_proba
  → Threshold comparison
  → APPROVE / DENY
```

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (champion) |
| Training Data | 2013–2015 (733K loans) |
| Validation AUC | 0.714 |
| Decision Threshold | 0.47 (cost-optimized) |
| Inference Time | < 50ms |
