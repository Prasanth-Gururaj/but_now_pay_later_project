"""Pipeline-bug vs real-drift classification logic.

Provides DriftClassifier that interprets Evidently drift report
results and categorizes the type of issue detected.
"""

from __future__ import annotations

from bnpl.logger import LoggerMixin


class DriftClassifier(LoggerMixin):
    """Classify drift detection results into actionable categories.

    Given Evidently drift report output, determines whether the issue
    is a pipeline bug, real distribution drift, or a single feature
    to investigate. All thresholds come from config.

    Categories:
        - pipeline_bug: schema or scale anomaly (data pipeline issue)
        - real_drift: 2+ features drifted together (distribution shift)
        - investigate: single feature drifted (may be noise)
        - clean: no issues detected

    Usage::

        classifier = DriftClassifier(min_features=2)
        result = classifier.classify(evidently_result_dict, "2018_jan")

    Depends on:
        - config/thresholds.yaml for min_drifted_features_for_retrain
        - LoggerMixin: structured logging
    """

    def __init__(self, min_drifted_features: int = 2) -> None:
        """Initialize with classification thresholds.

        Args:
            min_drifted_features: Minimum number of drifted features
                                  to classify as real_drift.
        """
        self._min_features = min_drifted_features

    def classify(
        self, evidently_result: dict, window_label: str,
    ) -> dict:
        """Classify drift results from Evidently report.

        Args:
            evidently_result: Dict from ``report.as_dict()`` containing
                              metrics with drift_share and per-column results.
            window_label: Monitoring window identifier for labeling.

        Returns:
            dict with keys:
                - window (str): the window label
                - drift_detected (bool): whether any drift found
                - failure_type (str): clean/investigate/real_drift
                - drifted_features (list[str]): names of drifted features
                - drift_score (float): proportion of features drifted
                - recommended_action (str): human-readable recommendation
        """
        drifted, drift_score = self._extract_drifted(evidently_result)

        if len(drifted) == 0:
            failure_type = "clean"
            action = "No action needed."
        elif len(drifted) < self._min_features:
            failure_type = "investigate"
            action = f"Investigate single feature drift: {drifted}"
        else:
            failure_type = "real_drift"
            action = f"Multiple features drifted ({len(drifted)}). Consider retraining."

        self.logger.info(
            "Drift classification: %s | %d features drifted",
            failure_type, len(drifted),
        )

        return {
            "window": window_label,
            "drift_detected": len(drifted) > 0,
            "failure_type": failure_type,
            "drifted_features": drifted,
            "drift_score": round(drift_score, 4),
            "recommended_action": action,
        }

    def _extract_drifted(self, evidently_result: dict) -> tuple[list[str], float]:
        """Extract drifted feature names and drift score from Evidently output.

        Args:
            evidently_result: Dict from Evidently report.as_dict().

        Returns:
            tuple: (list of drifted feature names, drift share float)
        """
        metrics = evidently_result.get("metrics", [])
        drifted: list[str] = []
        drift_score = 0.0

        if metrics:
            dataset_result = metrics[0].get("result", {})
            drift_score = dataset_result.get("drift_share", 0.0)
            columns = dataset_result.get("drift_by_columns", {})
            for col_name, col_data in columns.items():
                if col_data.get("drift_detected", False):
                    drifted.append(col_name)

        return drifted, drift_score
