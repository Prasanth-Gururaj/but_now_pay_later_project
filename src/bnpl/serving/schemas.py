"""Request and response Pydantic schemas for the prediction API.

All schemas used by the FastAPI endpoints are defined here and
imported by api.py for clean separation of concerns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    """Loan application input schema with all 17 raw fields.

    Every field corresponds to information available at the time the
    BNPL applicant clicks checkout. No post-origination or outcome
    data is included, preventing target leakage.
    """

    dti: float = Field(
        ..., ge=0,
        description="Debt-to-income ratio. Higher values indicate more financial stress.",
    )
    fico_range_low: float = Field(
        ..., ge=300, le=850,
        description="Lower bound of the applicant's FICO credit score range.",
    )
    revol_util: float = Field(
        ..., ge=0, le=892,
        description="Revolving utilization rate as percentage. Pipeline caps at 100.",
    )
    annual_inc: float = Field(
        ..., ge=0, description="Self-reported annual income in US dollars.",
    )
    loan_amnt: float = Field(
        ..., ge=0, description="The listed amount of the loan applied for.",
    )
    int_rate: float = Field(
        ..., ge=0, description="Interest rate on the loan as a percentage.",
    )
    sub_grade: str = Field(
        ..., description="LendingClub sub-grade from A1 (lowest risk) to G5 (highest).",
    )
    term: str = Field(
        ..., description='Loan term: "36 months" or "60 months".',
    )
    emp_length: str = Field(
        ..., description='Employment length, e.g. "5 years", "< 1 year", "10+ years".',
    )
    home_ownership: str = Field(
        ..., description='Home ownership: "RENT", "OWN", "MORTGAGE", or "ANY".',
    )
    verification_status: str = Field(
        ..., description='Income verification: "Verified", "Source Verified", "Not Verified".',
    )
    purpose: str = Field(
        ..., description='Loan purpose, e.g. "debt_consolidation", "credit_card".',
    )
    delinq_2yrs: float = Field(
        ..., ge=0, description="Delinquencies (30+ days) in the past 2 years.",
    )
    inq_last_6mths: float = Field(
        ..., ge=0, description="Credit inquiries in the last 6 months.",
    )
    open_acc: float = Field(
        ..., ge=0, description="Number of open credit lines.",
    )
    pub_rec: float = Field(
        ..., ge=0, description="Number of derogatory public records.",
    )
    revol_bal: float = Field(
        ..., ge=0, description="Total credit revolving balance in US dollars.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dti": 18.5, "fico_range_low": 690, "revol_util": 45.2,
                "annual_inc": 68000.0, "loan_amnt": 10000.0, "int_rate": 12.5,
                "sub_grade": "B3", "term": "36 months", "emp_length": "5 years",
                "home_ownership": "RENT", "verification_status": "Verified",
                "purpose": "debt_consolidation", "delinq_2yrs": 0.0,
                "inq_last_6mths": 1.0, "open_acc": 8.0, "pub_rec": 0.0,
                "revol_bal": 12000.0,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Prediction response with credit decision and metadata."""

    decision: str = Field(description='"APPROVE" or "DENY".')
    default_probability: float = Field(description="Probability of default, 4 decimal places.")
    threshold_used: float = Field(description="Threshold applied for the decision.")
    model_version: str = Field(description="Model identifier.")
    timestamp: str = Field(description="UTC timestamp in ISO 8601 format.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description='Service status, e.g. "healthy".')
    model_version: str = Field(description="Currently loaded model version.")
    threshold: float = Field(description="Active decision threshold.")
    timestamp: str = Field(description="Current UTC timestamp.")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing multiple loan applications.

    Usage::

        request = BatchPredictionRequest(applications=[app1, app2, app3])
    """

    applications: list[LoanApplication] = Field(
        ..., description="List of loan applications to score.",
    )
