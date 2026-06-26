"""BNPL Default Prediction — Full ML Lifecycle Dashboard.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="BNPL Default Prediction",
    page_icon="💳",
    layout="wide",
)

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
CONFIG_PATH = REPORTS_DIR / "data_prep_config.json"
METADATA_PATH = MODELS_DIR / "champion_metadata.json"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("💳 BNPL Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict", "📊 Model Performance", "🔍 Feature Importance",
     "📡 Monitoring", "ℹ️ About"],
)


# ---------------------------------------------------------------------------
# Helper: load model and pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    """Load the Predictor once and cache it across reruns."""
    from bnpl.models.predictor import Predictor

    metadata = _load_metadata()
    threshold = metadata.get("decision_threshold", 0.45)
    return Predictor(
        model_path=str(MODELS_DIR / "champion_xgboost.pkl"),
        config_path=str(CONFIG_PATH),
        threshold=threshold,
    )


def _load_metadata() -> dict:
    """Load champion_metadata.json."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Page: Predict
# ---------------------------------------------------------------------------
def page_predict():
    """Interactive loan application prediction form."""
    st.header("🔮 Loan Default Prediction")
    st.markdown("Enter applicant details below or select a sample profile.")

    # Sample profiles
    col_sample, _ = st.columns([1, 2])
    with col_sample:
        profile = st.selectbox(
            "Load sample profile",
            ["Custom", "Low Risk", "Medium Risk (Edge Case)", "High Risk"],
        )

    defaults = _get_profile_defaults(profile)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial")
        annual_inc = st.number_input("Annual Income ($)", value=defaults["annual_inc"], step=1000.0)
        dti = st.number_input("DTI Ratio", value=defaults["dti"], step=0.5)
        loan_amnt = st.number_input("Loan Amount ($)", value=defaults["loan_amnt"], step=500.0)
        int_rate = st.number_input("Interest Rate (%)", value=defaults["int_rate"], step=0.25)
        revol_bal = st.number_input("Revolving Balance ($)", value=defaults["revol_bal"], step=500.0)
        revol_util = st.number_input("Revolving Utilization (%)", value=defaults["revol_util"], step=1.0)

    with col2:
        st.subheader("Applicant")
        fico_range_low = st.number_input("FICO Score", value=defaults["fico_range_low"], min_value=300, max_value=850, step=5)
        sub_grade = st.selectbox("Sub Grade", [f"{l}{n}" for l in "ABCDEFG" for n in range(1, 6)], index=_grade_index(defaults["sub_grade"]))
        term = st.selectbox("Loan Term", ["36 months", "60 months"], index=0 if defaults["term"] == "36 months" else 1)
        emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"], index=_emp_index(defaults["emp_length"]))
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "ANY"], index=["RENT", "OWN", "MORTGAGE", "ANY"].index(defaults["home_ownership"]))
        purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "medical", "moving", "vacation", "house", "wedding", "educational", "renewable_energy", "other"], index=0)

    with col3:
        st.subheader("Credit History")
        verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"], index=["Verified", "Source Verified", "Not Verified"].index(defaults["verification_status"]))
        delinq_2yrs = st.number_input("Delinquencies (2yr)", value=defaults["delinq_2yrs"], step=1.0, min_value=0.0)
        inq_last_6mths = st.number_input("Inquiries (6mo)", value=defaults["inq_last_6mths"], step=1.0, min_value=0.0)
        open_acc = st.number_input("Open Accounts", value=defaults["open_acc"], step=1.0, min_value=0.0)
        pub_rec = st.number_input("Public Records", value=defaults["pub_rec"], step=1.0, min_value=0.0)

    st.divider()

    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        raw_input = {
            "dti": dti, "fico_range_low": float(fico_range_low),
            "revol_util": revol_util, "annual_inc": annual_inc,
            "loan_amnt": loan_amnt, "int_rate": int_rate,
            "sub_grade": sub_grade, "term": term,
            "emp_length": emp_length, "home_ownership": home_ownership,
            "verification_status": verification_status, "purpose": purpose,
            "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        }

        predictor = load_predictor()
        result = predictor.predict(raw_input)

        _display_prediction(result)


def _display_prediction(result: dict):
    """Render the prediction result with visual indicators."""
    col_dec, col_prob, col_thresh = st.columns(3)

    decision = result["decision"]
    prob = result["default_probability"]
    threshold = result["threshold_used"]

    with col_dec:
        if decision == "APPROVE":
            st.success(f"## ✅ {decision}")
        else:
            st.error(f"## ❌ {decision}")

    with col_prob:
        st.metric("Default Probability", f"{prob:.2%}")
        st.progress(min(prob, 1.0))

    with col_thresh:
        st.metric("Threshold Used", f"{threshold:.2f}")
        st.caption(f"Model: {result['model_version']}")


def _get_profile_defaults(profile: str) -> dict:
    """Return default values for a sample profile."""
    profiles = {
        "Low Risk": {
            "dti": 8.5, "fico_range_low": 760, "revol_util": 22.0,
            "annual_inc": 120000.0, "loan_amnt": 5000.0, "int_rate": 6.5,
            "sub_grade": "A3", "term": "36 months", "emp_length": "10+ years",
            "home_ownership": "MORTGAGE", "verification_status": "Verified",
            "delinq_2yrs": 0.0, "inq_last_6mths": 0.0, "open_acc": 12.0,
            "pub_rec": 0.0, "revol_bal": 5000.0,
        },
        "High Risk": {
            "dti": 35.0, "fico_range_low": 640, "revol_util": 89.5,
            "annual_inc": 28000.0, "loan_amnt": 25000.0, "int_rate": 24.5,
            "sub_grade": "F3", "term": "60 months", "emp_length": "< 1 year",
            "home_ownership": "RENT", "verification_status": "Not Verified",
            "delinq_2yrs": 3.0, "inq_last_6mths": 5.0, "open_acc": 4.0,
            "pub_rec": 1.0, "revol_bal": 18000.0,
        },
        "Medium Risk (Edge Case)": {
            "dti": 18.0, "fico_range_low": 685, "revol_util": 55.0,
            "annual_inc": 55000.0, "loan_amnt": 15000.0, "int_rate": 15.0,
            "sub_grade": "C4", "term": "36 months", "emp_length": "3 years",
            "home_ownership": "RENT", "verification_status": "Source Verified",
            "delinq_2yrs": 1.0, "inq_last_6mths": 2.0, "open_acc": 7.0,
            "pub_rec": 0.0, "revol_bal": 14000.0,
        },
    }
    return profiles.get(profile, profiles["Medium Risk (Edge Case)"])


def _grade_index(grade: str) -> int:
    """Find index of a sub_grade in the A1-G5 list."""
    grades = [f"{l}{n}" for l in "ABCDEFG" for n in range(1, 6)]
    return grades.index(grade) if grade in grades else 7


def _emp_index(emp: str) -> int:
    """Find index of employment length in the options list."""
    options = ["< 1 year", "1 year", "2 years", "3 years", "4 years",
               "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
    return options.index(emp) if emp in options else 5


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
def page_performance():
    """Display model performance metrics and plots."""
    st.header("📊 Model Performance")

    metadata = _load_metadata()
    metrics = metadata.get("validation_metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC", f"{metrics.get('auc', 0):.4f}")
    col2.metric("F1 (default)", f"{metrics.get('f1_default', 0):.4f}")
    col3.metric("Precision", f"{metrics.get('precision_default', 0):.4f}")
    col4.metric("Recall", f"{metrics.get('recall_default', 0):.4f}")

    st.divider()

    st.subheader("Model Details")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **Model Type** | {metadata.get('model_type', 'N/A')} |
        | **Training Data** | {metadata.get('trained_on', 'N/A')} |
        | **Validation Data** | {metadata.get('validated_on', 'N/A')} |
        | **Test Data** | {metadata.get('tested_on', 'N/A')} |
        | **Threshold** | {metadata.get('decision_threshold', 'N/A')} |
        | **Scale Pos Weight** | {metadata.get('scale_pos_weight', 'N/A'):.3f} |
        """)

    with col_b:
        st.markdown("**Metrics at Threshold 0.3** (cost-optimal)")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Precision | {metrics.get('precision_default_at_0.3', 0):.4f} |
        | Recall | {metrics.get('recall_default_at_0.3', 0):.4f} |
        | F1 | {metrics.get('f1_default_at_0.3', 0):.4f} |
        | Brier Score | {metrics.get('brier_score', 0):.4f} |
        """)

    st.divider()

    # Display saved plots
    st.subheader("Calibration Curve")
    cal_path = REPORTS_DIR / "calibration_plot.png"
    if cal_path.exists():
        st.image(str(cal_path), use_container_width=True)
    else:
        st.info("Run `python -m bnpl.main evaluation-pipeline` to generate.")

    roc_path = REPORTS_DIR / "model_comparison_roc.png"
    if roc_path.exists():
        st.subheader("ROC Curve")
        st.image(str(roc_path), use_container_width=True)

    thresh_path = REPORTS_DIR / "threshold_selection.png"
    if thresh_path.exists():
        st.subheader("Threshold Cost Sweep")
        st.image(str(thresh_path), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Feature Importance
# ---------------------------------------------------------------------------
def page_features():
    """Display SHAP feature importance."""
    st.header("🔍 Feature Importance (SHAP)")

    shap_path = REPORTS_DIR / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("Run `python -m bnpl.main evaluation-pipeline` to generate SHAP plots.")

    st.divider()
    st.subheader("Feature List (47 Model Inputs)")

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

        cols = config.get("final_model_columns", [])
        col1, col2 = st.columns(2)
        half = len(cols) // 2
        with col1:
            for i, c in enumerate(cols[:half], 1):
                st.text(f"{i:2d}. {c}")
        with col2:
            for i, c in enumerate(cols[half:], half + 1):
                st.text(f"{i:2d}. {c}")


# ---------------------------------------------------------------------------
# Page: Monitoring
# ---------------------------------------------------------------------------
def page_monitoring():
    """Display drift monitoring results."""
    st.header("📡 Monitoring & Drift Detection")

    log_path = REPORTS_DIR / "monitoring_log.json"
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            entries = json.load(f)

        st.success(f"**{len(entries)}** monitoring runs logged")

        for entry in reversed(entries[-5:]):
            with st.expander(
                f"Window: {entry.get('window', 'N/A')} — "
                f"Retrain: {'⚠️ YES' if entry.get('should_retrain') else '✅ No'}"
            ):
                st.json(entry)
    else:
        st.info("No monitoring runs yet. Run `python -m bnpl.main monitor --window 2018`")

    st.divider()
    st.subheader("Retraining History")

    retrain_path = REPORTS_DIR / "retraining_log.json"
    if retrain_path.exists():
        with open(retrain_path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in reversed(entries):
            with st.expander(f"Window: {entry.get('window')} — Decision: {entry.get('decision')}"):
                st.json(entry)
    else:
        st.caption("No retraining runs yet.")

    # Drift reports
    st.divider()
    st.subheader("Drift Reports")
    drift_dir = REPORTS_DIR / "drift_reports"
    if drift_dir.exists():
        reports = sorted(drift_dir.glob("*.html"))
        if reports:
            for r in reports:
                st.markdown(f"- [{r.name}]({r})")
        else:
            st.caption("No drift reports generated yet.")
    else:
        st.caption("No drift reports directory.")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about():
    """Model card and system documentation."""
    st.header("ℹ️ About This System")

    st.markdown("""
    ## BNPL Default Prediction System

    ### What It Does
    Predicts the probability that a Buy Now Pay Later (BNPL) loan applicant
    will default. Returns an **APPROVE** or **DENY** decision within 500ms
    using only information available at application time.

    ### Dataset
    - **Source**: LendingClub accepted loans 2007-2018
    - **Size**: 1.35 million resolved loans
    - **Class Balance**: 80% Fully Paid, 20% Charged Off (4:1 ratio)

    ### Model
    - **Algorithm**: XGBoost (champion, selected over Logistic Regression and LightGBM)
    - **Features**: 17 raw inputs → 47 engineered features
    - **Threshold**: Business cost-optimized (not default 0.5)

    ### Training & Evaluation
    | Split | Period | Rows | Purpose |
    |-------|--------|------|---------|
    | Train | 2013-2015 | 733K | Model fitting |
    | Validation | 2016 | 293K | Threshold selection, model comparison |
    | Test | 2017 | 169K | Final unbiased evaluation (touched once) |
    | Drift | 2018 | ~89K | Monitoring simulation |

    ### Known Limitations
    1. **sub_grade dominates** feature importance at ~70% — the model
       is largely repackaging the lender's own risk grade
    2. **Calibration is off** — predicted probabilities are overconfident
       (calibration curve below the diagonal)
    3. **AUC ceiling ~0.71** — limited by features available at application
       time; post-origination features would help but are leakage

    ### Architecture
    ```
    Raw Input (17 fields)
        → PreprocessingPipeline (47 features)
        → XGBoost predict_proba
        → Threshold comparison
        → APPROVE / DENY
    ```

    ### Tech Stack
    - **ML**: XGBoost, scikit-learn, SHAP, Evidently
    - **Tracking**: MLflow on DagsHub
    - **Serving**: FastAPI + Uvicorn
    - **Config**: Pydantic Settings + YAML
    - **Dashboard**: Streamlit
    """)

    st.divider()
    st.subheader("MLflow Experiment")
    st.markdown(
        "[View on DagsHub](https://dagshub.com/Prasanth-Gururaj/"
        "but_now_pay_later_project.mlflow)"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "🔮 Predict":
    page_predict()
elif page == "📊 Model Performance":
    page_performance()
elif page == "🔍 Feature Importance":
    page_features()
elif page == "📡 Monitoring":
    page_monitoring()
elif page == "ℹ️ About":
    page_about()
