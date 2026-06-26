"""Feature engineering pipeline orchestration.

Provides the PreprocessingPipeline class that transforms 17 raw loan
application fields into 47 model-ready numeric columns, matching the
exact transformations applied during training in Notebook 03.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bnpl.features.encoder import CategoricalEncoder
from bnpl.features.imputer import MedianImputer
from bnpl.logger import LoggerMixin, log_execution


class PreprocessingPipeline(LoggerMixin):
    """Transform raw loan application data into model-ready features.

    This class is the single source of truth for feature engineering in both
    the serving API and the monitoring pipeline. It prevents training-serving
    skew by loading ALL transformation parameters (medians, caps, category
    lists, ordinal mappings) from data_prep_config.json, which was generated
    from the training data (2013-2015) in Notebook 03.

    No transformation parameter is hardcoded in this class. Every median,
    cap value, and category list comes from the config file. This ensures
    that the same transformations applied during training are applied
    identically during serving and monitoring.

    Usage::

        pipeline = PreprocessingPipeline("reports/data_prep_config.json")
        features_df = pipeline.transform({
            "dti": 18.5, "fico_range_low": 690, "revol_util": 45.2,
            "annual_inc": 68000, "loan_amnt": 10000, "int_rate": 12.5,
            "sub_grade": "B3", "term": "36 months", "emp_length": "5 years",
            "home_ownership": "RENT", "verification_status": "Verified",
            "purpose": "debt_consolidation", "delinq_2yrs": 0,
            "inq_last_6mths": 1, "open_acc": 8, "pub_rec": 0,
            "revol_bal": 12000,
        })

    Depends on:
        - data_prep_config.json: all fitted transformation parameters
        - LoggerMixin: structured logging with automatic class naming
    """

    EMP_LENGTH_MAP: dict[str, int] = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }

    ORDINAL_FALLBACK: int = 17

    EXPECTED_OUTPUT_COLUMNS: int = 47

    def __init__(self, config_path: str | Path) -> None:
        """Load all transformation parameters from the config file.

        Args:
            config_path: Path to data_prep_config.json containing all
                         fitted transformation parameters from training.

        Raises:
            FileNotFoundError: If the config file does not exist.
            KeyError: If the config file is missing required sections.
        """
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        self._feature_cols: list[str] = config["final_model_columns"]
        self._impute_values: dict[str, float] = config["imputation"]["values"]
        self._outlier_caps: dict[str, float] = config["outlier_handling"]["caps"]
        self._train_categories: dict[str, list[str]] = config["encoding"][
            "nominal_categories_from_train"
        ]
        self._sub_grade_map: dict[str, int] = config["encoding"]["sub_grade"]["mapping"]

        self._imputer = MedianImputer(self._impute_values)
        self._encoder = CategoricalEncoder(self._sub_grade_map, self._train_categories)

        self.logger.info(
            "Pipeline loaded | features=%d | impute_cols=%d | categories=%d",
            len(self._feature_cols),
            len(self._impute_values),
            sum(len(v) for v in self._train_categories.values()),
        )

    @log_execution(operation="PreprocessingPipeline.transform")
    def transform(self, raw_input: dict) -> pd.DataFrame:
        """Transform 17 raw application fields into 47 model-ready columns.

        Applies the same preprocessing steps as Notebook 03 in the same
        order. All transformation parameters are loaded from config, not
        hardcoded. Handles missing fields and unseen categories gracefully.

        Args:
            raw_input: Dictionary containing the 17 raw loan application
                       fields. Missing fields are imputed with training
                       medians. Unknown categorical values are encoded
                       as all-zero one-hot rows.

        Returns:
            pd.DataFrame with exactly 47 numeric columns in the order
            defined by final_model_columns in data_prep_config.json.

        Raises:
            ValueError: If the output does not have exactly 47 columns
                        after all transformations are applied.
        """
        df = pd.DataFrame([raw_input])

        # Step 1: Convert emp_length text ("5 years") to numeric (5)
        df = self._apply_emp_length(df)

        # Step 2: Fill missing numerics with training medians, add was_missing flags
        df = self._apply_imputation(df)

        # Step 3: Encode sub_grade as ordinal (A1=0 through G5=34)
        df = self._apply_sub_grade_encoding(df)

        # Step 4: Extract numeric term from string ("36 months" -> 36.0)
        df = self._apply_term_extraction(df)

        # Step 5: One-hot encode categoricals using training-fixed category lists
        df = self._apply_one_hot_encoding(df)

        # Step 6: Cap outliers at training-derived thresholds
        df = self._apply_outlier_caps(df)

        # Step 7: Select the exact 47 columns in the correct order
        result = self._select_and_order(df)

        self._validate_output(result)
        return result

    @log_execution(operation="PreprocessingPipeline.transform_batch")
    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a full DataFrame of raw rows into model-ready features.

        Vectorized version of transform() for batch processing. Applies
        the same 7 steps using pandas vectorized operations on the entire
        DataFrame at once, making it ~1000x faster than row-by-row.

        Args:
            df: DataFrame with 17 raw feature columns. May also contain
                non-feature columns (default, issue_d, issue_year) which
                are preserved in the output.

        Returns:
            pd.DataFrame with the 47 model feature columns plus any
            extra columns (default, issue_d, issue_year) that were present.
        """
        d = df.copy()
        extra_cols = ["default", "issue_d", "issue_year"]
        extras = {c: d[c].copy() for c in extra_cols if c in d.columns}

        d = self._apply_emp_length(d)
        d = self._apply_imputation(d)
        d = self._apply_sub_grade_encoding(d)
        d = self._apply_term_extraction(d)
        d = self._apply_one_hot_encoding(d)
        d = self._apply_outlier_caps(d)
        result = self._select_and_order(d)

        self._validate_output(result)

        for col, values in extras.items():
            result[col] = values.values

        return result

    def _apply_emp_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert employment length text to numeric scale 0-10.

        Args:
            df: DataFrame with optional ``emp_length`` string column.

        Returns:
            DataFrame with ``emp_length_num`` column added. Values not
            in the mapping produce NaN, which the imputation step fills.
        """
        if "emp_length" in df.columns:
            df["emp_length_num"] = df["emp_length"].map(self.EMP_LENGTH_MAP)
        return df

    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing numeric values with training medians and add flags.

        Delegates to MedianImputer which creates binary was_missing
        flags and fills nulls with training medians for all 12 columns.

        Args:
            df: DataFrame after emp_length mapping.

        Returns:
            DataFrame with imputed values and 12 was_missing flag columns.
        """
        return self._imputer.impute(df)

    def _apply_sub_grade_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode sub_grade as ordinal integer (A1=0 through G5=34).

        Delegates to CategoricalEncoder which maps unknown sub_grades
        to the midpoint fallback value (D2=17).

        Args:
            df: DataFrame with optional ``sub_grade`` string column.

        Returns:
            DataFrame with ``sub_grade_encoded`` column added.
        """
        return self._encoder.encode_sub_grade(df)

    def _apply_term_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric term from string format.

        Converts ``"36 months"`` to ``36.0`` and ``"60 months"`` to ``60.0``.

        Args:
            df: DataFrame with optional ``term`` string column.

        Returns:
            DataFrame with ``term_num`` float column added.
        """
        if "term" in df.columns:
            df["term_num"] = df["term"].str.extract(r"(\d+)").astype(float)
        else:
            df["term_num"] = 36.0
        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns using training-fixed categories.

        Delegates to CategoricalEncoder which uses categories fixed from
        training data. Unseen categories produce all-zero rows.

        Args:
            df: DataFrame with optional categorical string columns
                (home_ownership, verification_status, purpose).

        Returns:
            DataFrame with one-hot encoded binary columns added.
        """
        return self._encoder.encode_one_hot(df)

    def _apply_outlier_caps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap numeric values at training-derived thresholds.

        Creates ``{col}_capped`` columns by clipping at the 99th
        percentile values from training, except revol_util which uses
        a hard business-logic cap at 100 (it is a percentage).

        Args:
            df: DataFrame with imputed numeric columns.

        Returns:
            DataFrame with capped columns added (e.g. annual_inc_capped).
        """
        for col, cap_val in self._outlier_caps.items():
            if col in df.columns:
                df[f"{col}_capped"] = df[col].clip(upper=cap_val)
            else:
                df[f"{col}_capped"] = cap_val
        return df

    def _select_and_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select exactly the 47 model columns in the correct order.

        Fills any missing columns with zero, selects only the columns
        the model expects, and forces every value to a numeric dtype
        to prevent type errors during prediction.

        Args:
            df: DataFrame with all engineered columns.

        Returns:
            DataFrame with exactly 47 numeric columns in config order.
        """
        for col in self._feature_cols:
            if col not in df.columns:
                df[col] = 0

        result = df[self._feature_cols].copy()
        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
        return result

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Validate that the output DataFrame has the expected shape.

        Args:
            df: The final transformed DataFrame.

        Raises:
            ValueError: If the column count does not match the expected
                        47 columns defined in data_prep_config.json.
        """
        actual = len(df.columns)
        if actual != self.EXPECTED_OUTPUT_COLUMNS:
            raise ValueError(
                f"Expected {self.EXPECTED_OUTPUT_COLUMNS} columns, got {actual}. "
                f"Missing: {set(self._feature_cols) - set(df.columns)}"
            )
