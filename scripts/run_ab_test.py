"""A/B test: train LightGBM challenger, simulate test, statistical analysis, plots.

Run with: python scripts/run_ab_test.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

NON_FEATURE_COLS = ["default", "issue_d", "issue_year"]
CHAMPION_THRESHOLD = 0.47
RANDOM_SEED = 42
CHAMPION_PCT = 0.70

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    return train, val, test, feature_cols


COST_FALSE_NEGATIVE = 300
COST_FALSE_POSITIVE = 45
MIN_APPROVAL_RATE = 0.55


def _select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Select threshold via minimum business cost with >=55% approval rate."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.50
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
        approval_rate = float((y_proba < t).mean())
        if approval_rate >= MIN_APPROVAL_RATE and total_cost < best_cost:
            best_cost = total_cost
            best_threshold = round(float(t), 2)

    return best_threshold


def train_challenger(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.LGBMClassifier, float]:
    """Train LightGBM challenger and return (model, threshold)."""
    print("=== Step 1: Training LightGBM challenger ===")

    X_train, y_train = train[feature_cols], train["default"]
    X_val, y_val = val[feature_cols], val["default"]
    X_test, y_test = test[feature_cols], test["default"]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        scale_pos_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    val_auc = roc_auc_score(y_val, val_proba)
    print(f"  Validation AUC: {val_auc:.4f}")

    print("  Probability distribution (validation set):")
    print(f"    min={val_proba.min():.4f}  max={val_proba.max():.4f}")
    print(f"    mean={val_proba.mean():.4f}  median={np.median(val_proba):.4f}")

    approval_at_champion = float((val_proba < CHAMPION_THRESHOLD).mean())
    print(f"  Approval rate at champion threshold ({CHAMPION_THRESHOLD}): "
          f"{approval_at_champion:.4f}")

    if 0.40 <= approval_at_champion <= 0.70:
        threshold = CHAMPION_THRESHOLD
        threshold_method = "same as champion (approval rate in valid range)"
        print(f"  Threshold: {threshold} (champion threshold is valid)")
    else:
        print(f"  Approval rate {approval_at_champion:.2%} outside 40-70% range")
        print("  Running constrained threshold selection...")
        threshold = _select_threshold(y_val.values, val_proba)
        new_approval = float((val_proba < threshold).mean())
        threshold_method = (
            "minimum total business cost with 55% minimum approval rate"
        )
        print(f"  Selected threshold: {threshold}")
        print(f"  Approval rate at new threshold: {new_approval:.4f}")

    val_pred = (val_proba >= threshold).astype(int)
    test_pred = (test_proba >= threshold).astype(int)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "challenger_lgbm.pkl")
    print(f"  Saved: {MODELS_DIR / 'challenger_lgbm.pkl'}")

    metadata = {
        "model_type": "lightgbm",
        "trained_on": "2013 to 2015, train split",
        "validated_on": "2016, val split",
        "tested_on": "2017, test split",
        "feature_columns": feature_cols,
        "decision_threshold": threshold,
        "threshold_selection_method": threshold_method,
        "scale_pos_weight": 4,
        "best_iteration": model.best_iteration_,
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "validation_metrics": {
            "auc": float(val_auc),
            "precision_default": float(precision_score(y_val, val_pred)),
            "recall_default": float(recall_score(y_val, val_pred)),
            "f1_default": float(f1_score(y_val, val_pred)),
            "brier_score": float(brier_score_loss(y_val, val_proba)),
        },
        "test_set_metrics": {
            "auc": float(roc_auc_score(y_test, test_proba)),
            "precision_default": float(precision_score(y_test, test_pred)),
            "recall_default": float(recall_score(y_test, test_pred)),
            "f1_default": float(f1_score(y_test, test_pred)),
            "brier_score": float(brier_score_loss(y_test, test_proba)),
            "approval_rate_at_threshold": float(
                (test_proba < threshold).mean()
            ),
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    with open(MODELS_DIR / "challenger_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {MODELS_DIR / 'challenger_metadata.json'}")

    print(f"  Test AUC:       {metadata['test_set_metrics']['auc']:.4f}")
    test_approval = metadata["test_set_metrics"]["approval_rate_at_threshold"]
    print(f"  Test approval:  {test_approval:.4f}")

    return model, threshold


def simulate_ab_test(
    test: pd.DataFrame,
    feature_cols: list[str],
    champion_model: object,
    challenger_model: lgb.LGBMClassifier,
    champion_threshold: float,
    challenger_threshold: float,
) -> pd.DataFrame:
    print("\n=== Step 2: Simulating A/B test on test set ===")
    print(f"  Champion threshold:   {champion_threshold}")
    print(f"  Challenger threshold: {challenger_threshold}")

    X_test = test[feature_cols].values
    y_test = test["default"].values

    rng = np.random.RandomState(RANDOM_SEED)
    assignment = rng.choice(
        ["champion", "challenger"],
        size=len(test),
        p=[CHAMPION_PCT, 1 - CHAMPION_PCT],
    )

    champ_mask = assignment == "champion"
    chall_mask = ~champ_mask

    prediction_proba = np.full(len(test), np.nan)
    prediction_proba[champ_mask] = champion_model.predict_proba(
        X_test[champ_mask]
    )[:, 1]
    prediction_proba[chall_mask] = challenger_model.predict_proba(
        X_test[chall_mask]
    )[:, 1]

    threshold_arr = np.where(champ_mask, champion_threshold, challenger_threshold)
    decision = np.where(prediction_proba >= threshold_arr, "deny", "approve")

    n_champ = int(champ_mask.sum())
    n_chall = int(chall_mask.sum())
    print(f"  Champion:   {n_champ} ({n_champ / len(test):.1%})")
    print(f"  Challenger: {n_chall} ({n_chall / len(test):.1%})")

    champ_approve = (decision[champ_mask] == "approve").sum()
    chall_approve = (decision[chall_mask] == "approve").sum()
    print(f"  Champion approval rate:   {champ_approve / n_champ:.4f}")
    print(f"  Challenger approval rate: {chall_approve / n_chall:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "ab_test_log.jsonl"
    now = datetime.now(UTC).isoformat()

    with open(log_path, "w", encoding="utf-8") as f:
        for i in range(len(test)):
            entry = {
                "model_version": (
                    "champion_xgboost_v1"
                    if champ_mask[i]
                    else "challenger_lgbm_v1"
                ),
                "loan_amnt": float(test.iloc[i]["loan_amnt"]),
                "int_rate": float(test.iloc[i]["int_rate"]),
                "fico_range_low": float(test.iloc[i]["fico_range_low"]),
                "prediction_proba": round(float(prediction_proba[i]), 4),
                "decision": decision[i],
                "timestamp": now,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Logged {len(test)} predictions to {log_path}")

    results_df = pd.DataFrame(
        {
            "assignment": assignment,
            "prediction_proba": prediction_proba,
            "decision": decision,
            "actual_default": y_test,
        }
    )
    return results_df


def run_statistical_analysis(
    results_df: pd.DataFrame,
    out_path: Path,
    champion_threshold: float,
    challenger_threshold: float,
) -> dict:
    print("\n=== Step 3: Statistical analysis ===")

    champ_mask = results_df["assignment"] == "champion"
    chall_mask = results_df["assignment"] == "challenger"

    n_champ = int(champ_mask.sum())
    n_chall = int(chall_mask.sum())

    champ_decisions = results_df.loc[champ_mask, "decision"]
    chall_decisions = results_df.loc[chall_mask, "decision"]
    champ_approved = int((champ_decisions == "approve").sum())
    chall_approved = int((chall_decisions == "approve").sum())

    p_champ = champ_approved / n_champ
    p_chall = chall_approved / n_chall

    champ_avg_prob = float(results_df.loc[champ_mask, "prediction_proba"].mean())
    chall_avg_prob = float(results_df.loc[chall_mask, "prediction_proba"].mean())

    champ_auc = float(
        roc_auc_score(
            results_df.loc[champ_mask, "actual_default"],
            results_df.loc[champ_mask, "prediction_proba"],
        )
    )
    chall_auc = float(
        roc_auc_score(
            results_df.loc[chall_mask, "actual_default"],
            results_df.loc[chall_mask, "prediction_proba"],
        )
    )

    p_pool = (champ_approved + chall_approved) / (n_champ + n_chall)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_champ + 1 / n_chall))
    z_stat = float((p_champ - p_chall) / se) if se > 0 else 0.0
    z_pvalue = float(2 * stats.norm.sf(abs(z_stat)))
    z_significant = z_pvalue < 0.05

    contingency = np.array(
        [
            [champ_approved, n_champ - champ_approved],
            [chall_approved, n_chall - chall_approved],
        ]
    )
    chi2, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)

    if chall_auc >= champ_auc and p_chall >= p_champ:
        recommendation = "promote_challenger"
        reason = (
            f"Challenger LightGBM (AUC={chall_auc:.4f}) matches or exceeds "
            f"champion XGBoost (AUC={champ_auc:.4f}) with comparable or better "
            f"approval rate ({p_chall:.4f} vs {p_champ:.4f})."
        )
    elif chall_auc >= champ_auc and z_significant:
        recommendation = "keep_champion"
        reason = (
            f"Challenger AUC ({chall_auc:.4f}) >= champion ({champ_auc:.4f}), "
            f"but approval rates differ significantly "
            f"(p={z_pvalue:.4f}). Investigate before promoting."
        )
    else:
        recommendation = "keep_champion"
        reason = (
            f"Champion XGBoost (AUC={champ_auc:.4f}) outperforms "
            f"challenger LightGBM (AUC={chall_auc:.4f}). "
            f"Approval rates: champion={p_champ:.4f}, "
            f"challenger={p_chall:.4f}."
        )

    analysis = {
        "test_date": datetime.now(UTC).isoformat(),
        "test_set": "2017 data, 169300 rows",
        "champion_model": "xgboost",
        "challenger_model": "lightgbm",
        "thresholds": {
            "champion": champion_threshold,
            "challenger": challenger_threshold,
        },
        "traffic_split": {"champion": CHAMPION_PCT, "challenger": 1 - CHAMPION_PCT},
        "sample_sizes": {"champion": n_champ, "challenger": n_chall},
        "approval_rates": {
            "champion": round(float(p_champ), 4),
            "challenger": round(float(p_chall), 4),
        },
        "avg_default_probability": {
            "champion": round(champ_avg_prob, 4),
            "challenger": round(chall_avg_prob, 4),
        },
        "auc": {
            "champion": round(champ_auc, 4),
            "challenger": round(chall_auc, 4),
        },
        "z_test": {
            "z_statistic": round(z_stat, 4),
            "p_value": round(z_pvalue, 6),
            "significant_at_0.05": z_significant,
        },
        "chi_square_test": {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(chi2_p), 6),
            "degrees_of_freedom": int(chi2_dof),
            "significant_at_0.05": bool(chi2_p < 0.05),
        },
        "recommendation": recommendation,
        "reason": reason,
    }

    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: {out_path}")

    print(f"  Champion AUC:       {champ_auc:.4f}")
    print(f"  Challenger AUC:     {chall_auc:.4f}")
    print(f"  Champion approval:  {p_champ:.4f}")
    print(f"  Challenger approval:{p_chall:.4f}")
    sig_label = "significant" if z_significant else "not significant"
    print(f"  Z-test p-value:     {z_pvalue:.6f} ({sig_label})")
    print(f"  Chi-square p-value: {chi2_p:.6f}")
    print(f"  Recommendation:     {recommendation}")
    print(f"  Reason: {reason}")

    return analysis


def generate_plots(
    results_df: pd.DataFrame,
    analysis: dict,
    out_dir: Path,
) -> None:
    print("\n=== Step 4: Generating plots ===")

    p_champ = analysis["approval_rates"]["champion"]
    p_chall = analysis["approval_rates"]["challenger"]
    n_champ = analysis["sample_sizes"]["champion"]
    n_chall = analysis["sample_sizes"]["challenger"]

    ci_champ = 1.96 * np.sqrt(p_champ * (1 - p_champ) / n_champ)
    ci_chall = 1.96 * np.sqrt(p_chall * (1 - p_chall) / n_chall)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        ["Champion\n(XGBoost)", "Challenger\n(LightGBM)"],
        [p_champ, p_chall],
        yerr=[ci_champ, ci_chall],
        capsize=8,
        color=["#2196F3", "#FF9800"],
        edgecolor="white",
        width=0.5,
    )
    for bar, rate in zip(bars, [p_champ, p_chall], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_ylabel("Approval Rate")
    ax.set_title("A/B Test: Approval Rate Comparison", fontweight="bold")
    ax.set_ylim(0, max(p_champ, p_chall) + 0.1)
    plt.tight_layout()
    path1 = out_dir / "ab_approval_rate_comparison.png"
    plt.savefig(path1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path1}")

    champ_mask = results_df["assignment"] == "champion"
    chall_mask = results_df["assignment"] == "challenger"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        results_df.loc[champ_mask, "prediction_proba"],
        bins=80,
        alpha=0.5,
        label="Champion (XGBoost)",
        color="#2196F3",
        density=True,
    )
    ax.hist(
        results_df.loc[chall_mask, "prediction_proba"],
        bins=80,
        alpha=0.5,
        label="Challenger (LightGBM)",
        color="#FF9800",
        density=True,
    )
    champ_thresh = analysis["thresholds"]["champion"]
    chall_thresh = analysis["thresholds"]["challenger"]
    ax.axvline(
        champ_thresh,
        color="#2196F3",
        linestyle="--",
        linewidth=1.5,
        label=f"Champion threshold ({champ_thresh})",
    )
    if chall_thresh != champ_thresh:
        ax.axvline(
            chall_thresh,
            color="#FF9800",
            linestyle="--",
            linewidth=1.5,
            label=f"Challenger threshold ({chall_thresh})",
        )
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(
        "Default Probability Distribution: Champion vs Challenger", fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    path2 = out_dir / "ab_default_prob_distribution.png"
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, mask, color in [
        ("Champion (XGBoost)", champ_mask, "#2196F3"),
        ("Challenger (LightGBM)", chall_mask, "#FF9800"),
    ]:
        y_true = results_df.loc[mask, "actual_default"]
        proba = results_df.loc[mask, "prediction_proba"]
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison: A/B Test", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path3 = out_dir / "ab_roc_comparison.png"
    plt.savefig(path3, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")


def main() -> None:
    print("=" * 60)
    print("BNPL A/B Test: Champion (XGBoost) vs Challenger (LightGBM)")
    print("=" * 60)

    train, val, test, feature_cols = load_data()
    print(f"Data loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Features: {len(feature_cols)}")

    challenger, challenger_threshold = train_challenger(
        train, val, test, feature_cols,
    )

    champion = joblib.load(MODELS_DIR / "champion_xgboost.pkl")
    print(f"\nChampion loaded: {MODELS_DIR / 'champion_xgboost.pkl'}")

    results_df = simulate_ab_test(
        test, feature_cols, champion, challenger,
        champion_threshold=CHAMPION_THRESHOLD,
        challenger_threshold=challenger_threshold,
    )

    analysis = run_statistical_analysis(
        results_df,
        REPORTS_DIR / "ab_test_results.json",
        champion_threshold=CHAMPION_THRESHOLD,
        challenger_threshold=challenger_threshold,
    )

    generate_plots(results_df, analysis, REPORTS_DIR)

    print("\n" + "=" * 60)
    print("A/B test complete!")
    print(f"  Recommendation: {analysis['recommendation']}")
    print(f"  Reason: {analysis['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
