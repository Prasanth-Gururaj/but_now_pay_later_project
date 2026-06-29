"""Unit tests for the PreprocessingPipeline.

Preprocessing correctness is the most critical production invariant.
If the pipeline applies different transformations than those used during
training, every prediction will be wrong regardless of model quality.
These tests verify that each transformation step produces the exact
values that the training pipeline in Notebook 03 produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bnpl.features.pipeline import PreprocessingPipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "reports" / "data_prep_config.json"


@pytest.fixture()
def pipeline() -> PreprocessingPipeline:
    """Create a PreprocessingPipeline loaded from the real config."""
    return PreprocessingPipeline(CONFIG_PATH)


@pytest.fixture()
def valid_input() -> dict:
    """Provide a complete 17-field raw input with realistic values."""
    return {
        "dti": 18.5,
        "fico_range_low": 690.0,
        "revol_util": 45.2,
        "annual_inc": 68000.0,
        "loan_amnt": 10000.0,
        "int_rate": 12.5,
        "sub_grade": "B3",
        "term": "36 months",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "delinq_2yrs": 0.0,
        "inq_last_6mths": 1.0,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_bal": 12000.0,
    }


class TestOutputShape:
    """Verify the pipeline produces the correct output dimensions."""

    def test_output_has_exactly_47_columns(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """17 raw inputs must produce exactly 47 output columns.

        The model was trained on 47 features: 7 uncapped numeric,
        4 capped numeric, 3 engineered (emp_length_num, sub_grade_encoded,
        term_num), 21 one-hot encoded, and 12 was_missing flags.
        Any deviation means the model receives the wrong feature vector.
        """
        result = pipeline.transform(valid_input)
        assert result.shape == (1, 47), f"Expected (1, 47), got {result.shape}"

    def test_all_columns_are_numeric(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """Every output column must be a numeric dtype.

        XGBoost requires all features to be numeric. String or object
        columns would cause a prediction error at serving time.
        """
        result = pipeline.transform(valid_input)
        for col in result.columns:
            assert result[col].dtype.kind in (
                "i",
                "f",
                "u",
            ), f"Column {col} has non-numeric dtype: {result[col].dtype}"


class TestSubGradeEncoding:
    """Verify ordinal encoding of LendingClub sub-grades."""

    def test_subgrade_a1_encodes_to_0(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """A1 is the lowest risk grade and must encode to 0.

        The ordinal encoding preserves risk ordering: A1 (safest) = 0
        through G5 (riskiest) = 34. Getting this wrong would reverse
        the model's understanding of credit risk.
        """
        valid_input["sub_grade"] = "A1"
        result = pipeline.transform(valid_input)
        assert result["sub_grade_encoded"].iloc[0] == 0

    def test_subgrade_g5_encodes_to_34(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """G5 is the highest risk grade and must encode to 34.

        This tests the upper bound of the ordinal encoding range.
        """
        valid_input["sub_grade"] = "G5"
        result = pipeline.transform(valid_input)
        assert result["sub_grade_encoded"].iloc[0] == 34

    def test_unknown_subgrade_maps_to_fallback_17(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """Unknown sub_grade values must map to the midpoint D2 = 17.

        In production, a new sub_grade value (e.g. from a schema change)
        must not crash the pipeline. The fallback of 17 places the
        applicant at median risk rather than at an extreme.
        """
        valid_input["sub_grade"] = "Z9"
        result = pipeline.transform(valid_input)
        assert result["sub_grade_encoded"].iloc[0] == 17


class TestOutlierCapping:
    """Verify outlier caps match training-derived thresholds."""

    def test_annual_inc_capped_at_250000(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """Annual income above 250k must be capped at 250000.

        The 99th percentile of training annual income was 250000.
        Without capping, extreme incomes distort tree splits.
        """
        valid_input["annual_inc"] = 500000
        result = pipeline.transform(valid_input)
        assert result["annual_inc_capped"].iloc[0] == 250000.0

    def test_revol_util_capped_at_100(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """Revolving utilization above 100 must be capped at 100.

        revol_util is a percentage with a mathematical ceiling of 100.
        Values above 100 (observed max was 892.3 in raw data) are data
        quality errors from the credit bureau. The pipeline applies a
        hard business logic cap at 100.
        """
        valid_input["revol_util"] = 150
        result = pipeline.transform(valid_input)
        assert result["revol_util_capped"].iloc[0] == 100.0

    def test_dti_capped_at_37_57(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """DTI above 37.57 must be capped at 37.57.

        The 99th percentile of training DTI was 37.57. Extreme DTI
        values indicate data entry errors rather than real applicants.
        """
        valid_input["dti"] = 999
        result = pipeline.transform(valid_input)
        assert result["dti_capped"].iloc[0] == 37.57


class TestImputation:
    """Verify missing value handling with training medians."""

    def test_missing_dti_filled_with_training_median(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """A missing dti field must be filled with the training median 17.81.

        The training median was calculated from 2013-2015 data only.
        Using a different value would create training-serving skew.
        """
        del valid_input["dti"]
        result = pipeline.transform(valid_input)
        assert result["dti_capped"].iloc[0] == 17.81

    def test_was_missing_flag_is_1_when_field_missing(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """The was_missing flag must be 1 when the field was not provided.

        The was_missing flags carry real predictive signal (emp_length_num
        _was_missing appears in the top 15 SHAP features). They allow the
        model to distinguish between genuine zeros and imputed zeros.
        """
        del valid_input["dti"]
        result = pipeline.transform(valid_input)
        assert result["dti_was_missing"].iloc[0] == 1

    def test_was_missing_flag_is_0_when_field_present(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """The was_missing flag must be 0 when the field was provided.

        When dti is provided, the flag must be 0 to indicate the value
        is genuine, not imputed.
        """
        result = pipeline.transform(valid_input)
        assert result["dti_was_missing"].iloc[0] == 0


class TestCategoricalEncoding:
    """Verify one-hot encoding with unseen category handling."""

    def test_unseen_home_ownership_gets_all_zero_encoding(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """An unseen home_ownership category must produce all-zero one-hot columns.

        Training categories are ANY, MORTGAGE, OWN, RENT. A new category
        (e.g. from a data schema change) must not crash the pipeline.
        All-zero encoding treats the applicant as having no strong signal
        from home ownership.
        """
        valid_input["home_ownership"] = "UNKNOWN"
        result = pipeline.transform(valid_input)
        ohe_cols = [c for c in result.columns if c.startswith("home_ownership_")]
        for col in ohe_cols:
            assert result[col].iloc[0] == 0, f"{col} should be 0 for unseen category"

    def test_unseen_purpose_gets_all_zero_encoding(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """An unseen purpose category must produce all-zero one-hot columns.

        Training has 14 purpose categories. A new purpose value must
        produce all zeros rather than crashing.
        """
        valid_input["purpose"] = "spaceship"
        result = pipeline.transform(valid_input)
        purpose_cols = [c for c in result.columns if c.startswith("purpose_")]
        for col in purpose_cols:
            assert result[col].iloc[0] == 0, f"{col} should be 0 for unseen category"


class TestEmpLength:
    """Verify employment length text to numeric mapping."""

    def test_less_than_1_year_maps_to_0(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """'< 1 year' employment length must map to 0.

        This is the minimum of the 0-10 ordinal scale. Getting it
        wrong would conflate new employees with experienced ones.
        """
        valid_input["emp_length"] = "< 1 year"
        result = pipeline.transform(valid_input)
        assert result["emp_length_num"].iloc[0] == 0

    def test_10_plus_years_maps_to_10(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """'10+ years' employment length must map to 10.

        This is the maximum of the 0-10 ordinal scale.
        """
        valid_input["emp_length"] = "10+ years"
        result = pipeline.transform(valid_input)
        assert result["emp_length_num"].iloc[0] == 10


class TestTermExtraction:
    """Verify term string to numeric extraction."""

    def test_36_months_extracts_to_36(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """'36 months' must extract to 36.0.

        The term field arrives as a string and must be converted to
        a float for the model.
        """
        valid_input["term"] = "36 months"
        result = pipeline.transform(valid_input)
        assert result["term_num"].iloc[0] == 36.0


class TestColumnOrder:
    """Verify output column ordering matches the config."""

    def test_output_columns_match_feature_cols_order(
        self, pipeline: PreprocessingPipeline, valid_input: dict
    ) -> None:
        """Output columns must be in exactly the order from data_prep_config.json.

        XGBoost uses column position, not column names. If the order
        differs from training, every feature gets mapped to the wrong
        tree split and predictions become meaningless.
        """
        result = pipeline.transform(valid_input)
        assert list(result.columns) == pipeline._feature_cols
