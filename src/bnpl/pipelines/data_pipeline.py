"""Data preparation pipeline: load, clean, split, preprocess, save.

Owns only data preparation. Nothing else. Produces processed parquet
files and data_prep_config.json for downstream training and serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class DataPipeline(LoggerMixin):
    """End-to-end data preparation from raw CSV to processed parquets.

    Steps:
    1. Load raw CSV via DataLoader
    2. Validate via DataValidator
    3. Clean via DataCleaner (filter, target, drop leakage/redundant)
    4. Split via TemporalSplitter (train/val/test/drift by year)
    5. Fit preprocessing parameters on train only
    6. Preprocess all splits via PreprocessingPipeline
    7. Check for leakage via LeakageChecker
    8. Save train/val/test/drift parquets
    9. Save data_prep_config.json

    Usage::

        pipeline = DataPipeline()
        result = pipeline.run("data/raw/accepted_2007_to_2018Q4.csv.gz")

    Depends on:
        - DataLoader, DataCleaner, DataValidator, TemporalSplitter
        - PreprocessingPipeline, LeakageChecker
        - config/settings.py for paths
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Initialize with paths from Settings."""
        self._paths = self._load_paths()

    def _load_paths(self) -> dict[str, str]:
        """Load paths from Settings with fallback defaults.

        Returns:
            dict: Path name to string mapping.
        """
        try:
            from config.settings import get_settings
            s = get_settings()
            return {
                "processed_dir": s.paths.processed_data_dir,
                "config_path": s.paths.config_path,
                "raw_data": str(Path(s.data.raw_dir) / "accepted_2007_to_2018Q4.csv.gz"),
            }
        except Exception:
            return {
                "processed_dir": "data/processed/",
                "config_path": "reports/data_prep_config.json",
                "raw_data": "data/raw/accepted_2007_to_2018Q4.csv.gz",
            }

    @log_execution(operation="DataPipeline.run")
    def run(self, raw_data_path: str | None = None) -> dict:
        """Run the full data preparation pipeline.

        Args:
            raw_data_path: Path to raw CSV. If None, uses config default.

        Returns:
            dict with split_sizes, class_balance, feature_count,
            config_save_path.

        Raises:
            ValueError: If data validation fails.
            FileNotFoundError: If raw CSV does not exist.
        """
        from bnpl.data.cleaner import DataCleaner
        from bnpl.data.loader import DataLoader
        from bnpl.data.splitter import TemporalSplitter
        from bnpl.data.validator import DataValidator

        data_path = raw_data_path or self._paths["raw_data"]

        loader = DataLoader()
        df_raw = loader.load(data_path)

        cleaner = DataCleaner()
        df_clean = cleaner.clean(df_raw)

        validator = DataValidator()
        report = validator.validate(df_clean)
        if not report["is_valid"]:
            raise ValueError(f"Data validation failed: {report['errors']}")

        splitter = TemporalSplitter()
        splits = splitter.split(df_clean)

        config = self._fit_and_save_config(splits["train"])
        self._preprocess_and_save(splits, config)
        self._check_leakage(splits["train"])

        return self._build_summary(splits, config)

    def _fit_and_save_config(self, train_df: pd.DataFrame) -> dict:
        """Fit preprocessing parameters on train and save config.

        Args:
            train_df: Raw training DataFrame.

        Returns:
            dict: The saved data_prep_config.
        """
        from bnpl.features.pipeline import PreprocessingPipeline

        numeric_features = [
            "dti", "fico_range_low", "revol_util", "annual_inc", "loan_amnt",
            "int_rate", "delinq_2yrs", "inq_last_6mths", "open_acc",
            "pub_rec", "revol_bal",
        ]
        nominal_cols = ["home_ownership", "verification_status", "purpose"]

        train_with_emp = train_df.copy()
        train_with_emp["emp_length_num"] = train_with_emp["emp_length"].map(
            PreprocessingPipeline.EMP_LENGTH_MAP
        )

        impute_values = {
            col: float(train_with_emp[col].median())
            for col in numeric_features + ["emp_length_num"]
        }

        outlier_caps = {
            col: float(train_df[col].quantile(0.99))
            for col in ["annual_inc", "revol_bal", "dti"]
        }
        outlier_caps["revol_util"] = 100.0

        train_categories = {
            col: sorted(train_df[col].dropna().unique().tolist())
            for col in nominal_cols
        }

        sub_grades = [f"{l}{n}" for l in "ABCDEFG" for n in range(1, 6)]
        sub_grade_map = {sg: i for i, sg in enumerate(sub_grades)}

        final_cols = self._build_final_columns(train_categories, impute_values)

        n_neg = int((train_df["default"] == 0).sum())
        n_pos = int((train_df["default"] == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        config = {
            "feature_set_version": "v2_rerun_safe",
            "temporal_split": {
                "train_years": [2013, 2014, 2015], "val_year": 2016,
                "test_year": 2017,
                "held_out_for_later": {
                    "production_simulation": [2018],
                    "crisis_simulation": [2008, 2009],
                },
            },
            "features": {
                "numeric": numeric_features,
                "categorical_ordinal": ["sub_grade"],
                "categorical_nominal": nominal_cols,
                "binary": ["term"],
                "engineered": ["emp_length_num"],
            },
            "imputation": {
                "method": "median (calculated on TRAIN split only)",
                "values": impute_values, "missing_flag_added": True,
            },
            "encoding": {
                "sub_grade": {"method": "ordinal", "mapping": sub_grade_map},
                "term": {"method": "numeric_extraction"},
                "nominal_categories_from_train": train_categories,
                "unseen_category_handling": "all-zero one-hot row (no crash)",
            },
            "outlier_handling": {
                "method": "99th percentile cap (TRAIN only), except revol_util (hard cap at 100)",
                "caps": outlier_caps,
            },
            "class_imbalance": {
                "train_class_counts": {"fully_paid": n_neg, "charged_off": n_pos},
                "strategy": "class_weighting", "scale_pos_weight": spw,
                "applied_at": "training time, not in this notebook",
            },
            "final_model_columns": final_cols,
        }

        config_path = Path(self._paths["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        self.logger.info("Config saved to %s", config_path)
        return config

    def _build_final_columns(
        self, train_categories: dict, impute_values: dict,
    ) -> list[str]:
        """Build the ordered list of 47 final model columns.

        Args:
            train_categories: Nominal column category lists.
            impute_values: Imputation column-value pairs.

        Returns:
            list[str]: Ordered column names matching Notebook 03.
        """
        capped_sources = ["annual_inc", "revol_bal", "revol_util", "dti"]
        numeric_features = [
            "dti", "fico_range_low", "revol_util", "annual_inc", "loan_amnt",
            "int_rate", "delinq_2yrs", "inq_last_6mths", "open_acc",
            "pub_rec", "revol_bal",
        ]
        base = [c for c in numeric_features if c not in capped_sources]
        capped = [f"{c}_capped" for c in capped_sources]
        engineered = ["emp_length_num", "sub_grade_encoded", "term_num"]
        onehot = [f"{col}_{cat}" for col, cats in train_categories.items() for cat in cats]
        missing = [f"{c}_was_missing" for c in impute_values]
        return base + capped + engineered + onehot + missing

    def _preprocess_and_save(self, splits: dict, config: dict) -> None:
        """Preprocess all splits and save as parquet files.

        Uses transform_batch() for vectorized processing of entire
        DataFrames at once instead of row-by-row iteration.

        Args:
            splits: Dict of split name to raw DataFrame.
            config: data_prep_config dict (just saved).
        """
        from bnpl.features.pipeline import PreprocessingPipeline

        config_path = self._paths["config_path"]
        pipeline = PreprocessingPipeline(config_path)
        processed_dir = Path(self._paths["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)

        final_shapes = {}
        for name, split_df in splits.items():
            if split_df.empty:
                self.logger.info("Skipping empty split: %s", name)
                continue

            result = pipeline.transform_batch(split_df)
            out_path = processed_dir / f"{name}.parquet"
            result.to_parquet(out_path, index=False)
            feature_count = len([c for c in result.columns if c not in ("default", "issue_d", "issue_year")])
            final_shapes[name] = [len(result), feature_count]
            self.logger.info("Saved %s: %s", name, result.shape)

        self._save_final_shapes(final_shapes)

    def _save_final_shapes(self, shapes: dict) -> None:
        """Append final_shapes to data_prep_config.json.

        Args:
            shapes: Dict mapping split names to [rows, cols] lists.
        """
        config_path = Path(self._paths["config_path"])
        if not config_path.exists():
            return
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        config["final_shapes"] = shapes
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        self.logger.info("Saved final_shapes to config: %s", shapes)

    def _check_leakage(self, train_df: pd.DataFrame) -> None:
        """Verify no leakage columns survived into processed data.

        Args:
            train_df: Training DataFrame to check.
        """
        from bnpl.features.leakage import LeakageChecker

        checker = LeakageChecker()
        if "default" in train_df.columns:
            report = checker.check(train_df, target_col="default")
            if report["leakage_found"]:
                self.logger.warning(
                    "Leakage detected in training data: %s",
                    [f["name"] for f in report["flagged_columns"]],
                )

    def _build_summary(self, splits: dict, config: dict) -> dict:
        """Build pipeline run summary.

        Args:
            splits: Split name to DataFrame.
            config: Saved config dict.

        Returns:
            dict: Summary with sizes, balance, feature count.
        """
        return {
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "class_balance": {
                k: round(v["default"].mean(), 4) if len(v) > 0 else 0
                for k, v in splits.items()
            },
            "feature_count": len(config.get("final_model_columns", [])),
            "config_save_path": self._paths["config_path"],
        }
