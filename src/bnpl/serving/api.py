from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = FastAPI(
    title="BNPL Default Prediction API",
    description="Predicts default probability for BNPL loan applications",
    version="1.0.0"
)

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/champion_xgboost.pkl")
CONFIG_PATH = os.getenv("CONFIG_PATH", "reports/data_prep_config.json")
THRESHOLD   = float(os.getenv("DECISION_THRESHOLD", "0.45"))

model  = joblib.load(MODEL_PATH)
with open(CONFIG_PATH) as f:
    config = json.load(f)

FEATURE_COLS     = config["final_model_columns"]
IMPUTE_VALUES    = config["imputation"]["values"]
OUTLIER_CAPS     = config["outlier_handling"]["caps"]
TRAIN_CATEGORIES = config["encoding"]["nominal_categories_from_train"]
SUB_GRADE_MAP    = config["encoding"]["sub_grade"]["mapping"]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10
}


class LoanApplication(BaseModel):
    dti:                 float = Field(..., ge=0,    description="Debt to income ratio")
    fico_range_low:      float = Field(..., ge=300,  le=850, description="Credit score lower bound")
    revol_util:          float = Field(..., ge=0,    le=100, description="Revolving utilization percent")
    annual_inc:          float = Field(..., ge=0,    description="Annual income in dollars")
    loan_amnt:           float = Field(..., ge=0,    description="Loan amount requested")
    int_rate:            float = Field(..., ge=0,    description="Interest rate percent")
    sub_grade:           str   = Field(...,          description="LendingClub sub grade, e.g. B3")
    term:                str   = Field(...,          description="36 months or 60 months")
    emp_length:          str   = Field(...,          description="Employment length, e.g. 5 years")
    home_ownership:      str   = Field(...,          description="RENT, OWN, MORTGAGE or OTHER")
    verification_status: str   = Field(...,          description="Verified, Source Verified or Not Verified")
    purpose:             str   = Field(...,          description="Loan purpose, e.g. debt_consolidation")
    delinq_2yrs:         float = Field(..., ge=0,    description="Delinquencies in last 2 years")
    inq_last_6mths:      float = Field(..., ge=0,    description="Credit inquiries in last 6 months")
    open_acc:            float = Field(..., ge=0,    description="Number of open credit lines")
    pub_rec:             float = Field(..., ge=0,    description="Number of public records")
    revol_bal:           float = Field(..., ge=0,    description="Current revolving balance")


class PredictionResponse(BaseModel):
    decision:            str
    default_probability: float
    threshold_used:      float
    model_version:       str
    timestamp:           str


def preprocess(raw: dict) -> pd.DataFrame:
    d = pd.Series(raw).to_frame().T

    d["emp_length_num"] = d["emp_length"].map(EMP_LENGTH_MAP)

    for col, fill_val in IMPUTE_VALUES.items():
        base = col.replace("_num", "") if col == "emp_length_num" else col
        src  = base if base in d.columns else col
        d[col + "_was_missing"] = d[src].isnull().astype(int) if src in d.columns else 0
        d[col] = d[src].fillna(fill_val) if src in d.columns else fill_val

    d["sub_grade_encoded"] = d["sub_grade"].map(SUB_GRADE_MAP).fillna(17)
    d["term_num"]          = d["term"].str.extract(r"(\d+)").astype(float)

    for col, cap_val in OUTLIER_CAPS.items():
        d[col + "_capped"] = d[col].clip(upper=cap_val) if col in d.columns else cap_val

    for col, cats in TRAIN_CATEGORIES.items():
        for cat in cats:
            d[f"{col}_{cat}"] = (d[col] == cat).astype(int)

    for col in FEATURE_COLS:
        if col not in d.columns:
            d[col] = 0

    return d[FEATURE_COLS]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": "champion_xgboost_v1",
        "threshold": THRESHOLD,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        raw         = application.model_dump()
        features    = preprocess(raw)
        probability = float(model.predict_proba(features)[0][1])
        decision    = "DENY" if probability >= THRESHOLD else "APPROVE"

        return PredictionResponse(
            decision            = decision,
            default_probability = round(probability, 4),
            threshold_used      = THRESHOLD,
            model_version       = "champion_xgboost_v1",
            timestamp           = datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "BNPL Default Prediction API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict"
    }