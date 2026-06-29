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


def train_challenger(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> lgb.LGBMClassifier:
    print("=== Step 1: Training LightGBM challenger ===")

    X_train, y_train = train[feature_cols], train["default"]
    X_val, y_val = val[feature_cols], val["default"]
    X_test, y_test = test[feature_cols], test["default"]

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "challenger_lgbm.pkl")
    print(f"  Saved: {MODELS_DIR / 'challenger_lgbm.pkl'}")

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= CHAMPION_THRESHOLD).astype(int)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= CHAMPION_THRESHOLD).astype(int)

    metadata = {
        "model_type": "lightgbm",
        "trained_on": "2013 to 2015, train split",
        "validated_on": "2016, val split",
        "tested_on": "2017, test split",
        "feature_columns": feature_cols,
        "decision_threshold": CHAMPION_THRESHOLD,
        "threshold_selection_method": "same as champion for fair A/B comparison",
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": model.best_iteration_,
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
        },
        "validation_metrics": {
            "auc": float(roc_auc_score(y_val, val_proba)),
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
            "approval_rate_at_threshold": float((test_proba < CHAMPION_THRESHOLD).mean()),
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    with open(MODELS_DIR / "challenger_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {MODELS_DIR / 'challenger_metadata.json'}")

    print(f"  Validation AUC: {metadata['validation_metrics']['auc']:.4f}")
    print(f"  Test AUC:       {metadata['test_set_metrics']['auc']:.4f}")
    print(f"  Test approval:  {metadata['test_set_metrics']['approval_rate_at_threshold']:.4f}")

    return model


def simulate_ab_test(
    test: pd.DataFrame,
    feature_cols: list[str],
    champion_model: object,
    challenger_model: lgb.LGBMClassifier,
) -> pd.DataFrame:
    print("\n=== Step 2: Simulating A/B test on test set ===")

    X_test = test[feature_cols]
    y_test = test["default"]

    champion_proba = champion_model.predict_proba(X_test)[:, 1]
    challenger_proba = challenger_model.predict_proba(X_test)[:, 1]

    rng = np.random.RandomState(RANDOM_SEED)
    assignment = rng.choice(
        ["champion", "challenger"],
        size=len(test),
        p=[CHAMPION_PCT, 1 - CHAMPION_PCT],
    )

    champion_decision = np.where(champion_proba >= CHAMPION_THRESHOLD, "deny", "approve")
    challenger_decision = np.where(
        challenger_proba >= CHAMPION_THRESHOLD, "deny", "approve"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "ab_test_log.jsonl"
    now = datetime.now(UTC).isoformat()

    with open(log_path, "w", encoding="utf-8") as f:
        for i in range(len(test)):
            assigned = assignment[i]
            proba = float(
                champion_proba[i] if assigned == "champion" else challenger_proba[i]
            )
            decision = (
                champion_decision[i] if assigned == "champion" else challenger_decision[i]
            )
            entry = {
                "model_version": (
                    "champion_xgboost_v1"
                    if assigned == "champion"
                    else "challenger_lgbm_v1"
                ),
                "loan_amnt": float(test.iloc[i]["loan_amnt"]),
                "int_rate": float(test.iloc[i]["int_rate"]),
                "fico_range_low": float(test.iloc[i]["fico_range_low"]),
                "prediction_proba": round(proba, 4),
                "decision": decision,
                "timestamp": now,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Logged {len(test)} predictions to {log_path}")

    n_champ = (assignment == "champion").sum()
    n_chall = (assignment == "challenger").sum()
    print(f"  Champion: {n_champ} ({n_champ/len(test):.1%})")
    print(f"  Challenger: {n_chall} ({n_chall/len(test):.1%})")

    results_df = pd.DataFrame(
        {
            "assignment": assignment,
            "champion_proba": champion_proba,
            "challenger_proba": challenger_proba,
            "champion_decision": champion_decision,
            "challenger_decision": challenger_decision,
            "actual_default": y_test.values,
        }
    )
    return results_df


def run_statistical_analysis(
    results_df: pd.DataFrame,
    out_path: Path,
) -> dict:
    print("\n=== Step 3: Statistical analysis ===")

    champ_mask = results_df["assignment"] == "champion"
    chall_mask = results_df["assignment"] == "challenger"

    n_champ = int(champ_mask.sum())
    n_chall = int(chall_mask.sum())

    champ_approved = (results_df.loc[champ_mask, "champion_decision"] == "approve").sum()
    chall_approved = (
        results_df.loc[chall_mask, "challenger_decision"] == "approve"
    ).sum()

    p_champ = champ_approved / n_champ
    p_chall = chall_approved / n_chall

    champ_avg_prob = float(results_df.loc[champ_mask, "champion_proba"].mean())
    chall_avg_prob = float(results_df.loc[chall_mask, "challenger_proba"].mean())

    champ_auc = float(
        roc_auc_score(results_df["actual_default"], results_df["champion_proba"])
    )
    chall_auc = float(
        roc_auc_score(results_df["actual_default"], results_df["challenger_proba"])
    )

    p_pool = (champ_approved + chall_approved) / (n_champ + n_chall)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_champ + 1 / n_chall))
    z_stat = float((p_champ - p_chall) / se) if se > 0 else 0.0
    z_pvalue = float(2 * stats.norm.sf(abs(z_stat)))
    z_significant = z_pvalue < 0.05

    contingency = np.array(
        [
            [int(champ_approved), int(n_champ - champ_approved)],
            [int(chall_approved), int(n_chall - chall_approved)],
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
            f"Approval rates: champion={p_champ:.4f}, challenger={p_chall:.4f}."
        )

    analysis = {
        "test_date": datetime.now(UTC).isoformat(),
        "test_set": "2017 data, 169300 rows",
        "champion_model": "xgboost",
        "challenger_model": "lightgbm",
        "threshold": CHAMPION_THRESHOLD,
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
        "auc": {"champion": round(champ_auc, 4), "challenger": round(chall_auc, 4)},
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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        results_df["champion_proba"],
        bins=80,
        alpha=0.5,
        label="Champion (XGBoost)",
        color="#2196F3",
        density=True,
    )
    ax.hist(
        results_df["challenger_proba"],
        bins=80,
        alpha=0.5,
        label="Challenger (LightGBM)",
        color="#FF9800",
        density=True,
    )
    ax.axvline(
        CHAMPION_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({CHAMPION_THRESHOLD})",
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
    y_true = results_df["actual_default"]

    for name, proba, color in [
        ("Champion (XGBoost)", results_df["champion_proba"], "#2196F3"),
        ("Challenger (LightGBM)", results_df["challenger_proba"], "#FF9800"),
    ]:
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

    challenger = train_challenger(train, val, test, feature_cols)

    champion = joblib.load(MODELS_DIR / "champion_xgboost.pkl")
    print(f"\nChampion loaded: {MODELS_DIR / 'champion_xgboost.pkl'}")

    results_df = simulate_ab_test(test, feature_cols, champion, challenger)

    analysis = run_statistical_analysis(
        results_df,
        REPORTS_DIR / "ab_test_results.json",
    )

    generate_plots(results_df, analysis, REPORTS_DIR)

    print("\n" + "=" * 60)
    print("A/B test complete!")
    print(f"  Recommendation: {analysis['recommendation']}")
    print(f"  Reason: {analysis['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
