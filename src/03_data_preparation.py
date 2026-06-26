"""
Step 3: Data Preparation
- Data cleaning (dtype fixes, business-rule checks)
- Missing value treatment (median/mode imputation, >40% drop rule)
- Outlier treatment (1%-99% capping / Winsorization)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def get_feature_columns(df: pd.DataFrame) -> list:
    """All columns minus IDs/dates/target/leakage fields."""
    drop = set(cfg.EXCLUDE_COLS + [cfg.TARGET_COL])
    return [c for c in df.columns if c not in drop and c != "Vintage_Quarter"]


def missing_value_report(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rep = df[cols].isnull().mean().sort_values(ascending=False).reset_index()
    rep.columns = ["feature", "missing_pct"]
    return rep


def treat_missing(df: pd.DataFrame, cols: list, drop_threshold=None):
    """Drop vars with >threshold missing; median-impute numeric, mode-impute categorical."""
    drop_threshold = drop_threshold or cfg.MISSING_DROP_THRESHOLD
    df = df.copy()
    miss = df[cols].isnull().mean()
    to_drop = miss[miss > drop_threshold].index.tolist()
    keep_cols = [c for c in cols if c not in to_drop]

    for c in keep_cols:
        if df[c].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            mode = df[c].mode()
            df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "Missing")
    return df, keep_cols, to_drop


def winsorize(df: pd.DataFrame, cols: list, lower=None, upper=None):
    """Cap numeric features at the 1st/99th percentile (computed on TRAIN only)."""
    lower = lower if lower is not None else cfg.OUTLIER_LOWER_PCT
    upper = upper if upper is not None else cfg.OUTLIER_UPPER_PCT
    df = df.copy()
    bounds = {}
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        lo, hi = df[c].quantile(lower), df[c].quantile(upper)
        bounds[c] = (lo, hi)
        df[c] = df[c].clip(lo, hi)
    return df, bounds


def apply_bounds(df: pd.DataFrame, bounds: dict):
    """Apply train-derived winsorization bounds to test/OOT to avoid leakage."""
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)
    return df


def business_rule_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity checks a credit risk team would run before modelling."""
    issues = []
    if "Age" in df.columns:
        issues.append(("Age out of [18,100]", ((df["Age"] < 18) | (df["Age"] > 100)).sum()))
    if "Credit_Utilization_Ratio" in df.columns:
        issues.append(("Utilization Ratio <0 or >1.5", ((df["Credit_Utilization_Ratio"] < 0) | (df["Credit_Utilization_Ratio"] > 1.5)).sum()))
    if "LTV_Ratio" in df.columns:
        issues.append(("LTV Ratio <0 or >1.5", ((df["LTV_Ratio"] < 0) | (df["LTV_Ratio"] > 1.5)).sum()))
    return pd.DataFrame(issues, columns=["rule", "violations"])


def run():
    train = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "test.csv"))
    oot = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "oot.csv"))

    feature_cols = get_feature_columns(train)

    miss_rep = missing_value_report(train, feature_cols)
    miss_rep.to_csv(os.path.join(cfg.REPORTS_DIR, "03_missing_value_report.csv"), index=False)
    print("Top missing features:\n", miss_rep.head(10).to_string(index=False))

    rules = business_rule_checks(train)
    print("\nBusiness rule check:\n", rules.to_string(index=False))

    train_imp, keep_cols, dropped = treat_missing(train, feature_cols)
    train_imp = train_imp.drop(columns=dropped)
    print(f"\nDropped (>{cfg.MISSING_DROP_THRESHOLD*100:.0f}% missing): {dropped}")

    # Outlier bounds learned on TRAIN, applied everywhere (no leakage)
    train_clean, bounds = winsorize(train_imp, keep_cols)
    test_imp, _, _ = treat_missing(test, keep_cols, drop_threshold=1.0)  # keep same cols, just impute
    oot_imp, _, _ = treat_missing(oot, keep_cols, drop_threshold=1.0)
    test_imp = test_imp[[c for c in test_imp.columns if c not in dropped]]
    oot_imp = oot_imp[[c for c in oot_imp.columns if c not in dropped]]
    test_clean = apply_bounds(test_imp, bounds)
    oot_clean = apply_bounds(oot_imp, bounds)

    for name, d in [("train", train_clean), ("test", test_clean), ("oot", oot_clean)]:
        d.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, f"{name}_clean.csv"), index=False)

    print(f"\nFinal feature count after missing-value filter: {len(keep_cols)}")
    return train_clean, test_clean, oot_clean, keep_cols


if __name__ == "__main__":
    run()
