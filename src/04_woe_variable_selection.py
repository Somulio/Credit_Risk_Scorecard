"""
Steps 4-5: Segmentation, WOE/IV Binning & Variable Reduction
- optbinning-based WOE/IV for every candidate feature
- judgmental + K-Means statistical segmentation profile
- IV filter -> correlation filter -> VIF filter -> monotonicity check
"""
import os
import sys
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def compute_woe_iv(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Fit optimal (monotonic) binning per feature; return IV summary + fitted binners."""
    results, binners = [], {}
    for col in feature_cols:
        x = df[col].values
        y = df[target_col].values
        dtype = "categorical" if df[col].dtype == object else "numerical"
        try:
            ob = OptimalBinning(name=col, dtype=dtype, solver="cp", monotonic_trend="auto",
                                 max_n_bins=4, min_bin_size=0.10)
            ob.fit(x, y)
            iv = ob.binning_table.build().loc["Totals", "IV"]
            results.append({"feature": col, "iv": iv, "dtype": dtype, "n_bins": len(ob.splits) + 1 if dtype == "numerical" else len(ob.splits)})
            binners[col] = ob
        except Exception as e:
            results.append({"feature": col, "iv": np.nan, "dtype": dtype, "n_bins": np.nan, "error": str(e)})
    iv_df = pd.DataFrame(results).sort_values("iv", ascending=False)
    return iv_df, binners


def iv_filter(iv_df: pd.DataFrame, min_iv=None, max_iv=None):
    """
    Step 5B (IV filter) + Step 5F (Business Review) combined:
    - drop anything below the noise floor (IV_MIN_THRESHOLD)
    - anything above IV_SUSPICIOUS_THRESHOLD is auto-flagged for review
    - vars on BUSINESS_APPROVED_HIGH_IV_VARS are confirmed legitimate
      pre-default behavioral/bureau predictors and kept regardless of how
      high their IV is; everything else above the suspicious threshold is dropped
      pending SME sign-off (true post-default leakage never reaches this stage --
      it's already hard-excluded in config.EXCLUDE_COLS).
    """
    min_iv = min_iv or cfg.IV_MIN_THRESHOLD
    max_iv = max_iv or cfg.IV_SUSPICIOUS_THRESHOLD
    approved = set(cfg.BUSINESS_APPROVED_HIGH_IV_VARS)

    above_floor = iv_df[iv_df["iv"] >= min_iv]
    keep = above_floor[(above_floor["iv"] <= max_iv) | (above_floor["feature"].isin(approved))]
    pending_review = above_floor[(above_floor["iv"] > max_iv) & (~above_floor["feature"].isin(approved))]
    return keep["feature"].tolist(), pending_review


def transform_to_woe(df: pd.DataFrame, feature_cols: list, binners: dict) -> pd.DataFrame:
    woe_df = pd.DataFrame(index=df.index)
    for col in feature_cols:
        woe_df[col] = binners[col].transform(df[col].values, metric="woe")
    return woe_df


def correlation_filter(woe_df: pd.DataFrame, iv_df: pd.DataFrame, threshold=None):
    """Among pairs with |corr| > threshold, drop the one with lower IV."""
    threshold = threshold or cfg.CORR_DROP_THRESHOLD
    corr = woe_df.corr().abs()
    iv_map = iv_df.set_index("feature")["iv"].to_dict()
    to_drop = set()
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if c1 in to_drop or c2 in to_drop:
                continue
            if corr.loc[c1, c2] > threshold:
                drop = c1 if iv_map.get(c1, 0) < iv_map.get(c2, 0) else c2
                to_drop.add(drop)
    kept = [c for c in cols if c not in to_drop]
    return kept, sorted(to_drop), corr


def stepwise_forward_select(woe_df: pd.DataFrame, y: pd.Series, candidates: list, max_vars=12):
    """
    Greedy forward selection using 5-fold CV AUC (logistic regression on WOE).
    Stops when adding a variable no longer improves CV AUC, or max_vars is hit.
    This is the standard scorecard-team approach to control overfitting when
    the bad count is small relative to the candidate variable list.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    selected, remaining = [], list(candidates)
    best_auc = 0.0
    while remaining and len(selected) < max_vars:
        scores = {}
        for c in remaining:
            cols = selected + [c]
            X = woe_df[cols].values
            lr = LogisticRegression(solver="liblinear", C=0.5, random_state=cfg.RANDOM_STATE)
            try:
                cv_auc = cross_val_score(lr, X, y, cv=5, scoring="roc_auc").mean()
            except Exception:
                cv_auc = 0.0
            scores[c] = cv_auc
        best_var = max(scores, key=scores.get)
        if scores[best_var] <= best_auc + 0.002:  # negligible improvement -> stop
            break
        best_auc = scores[best_var]
        selected.append(best_var)
        remaining.remove(best_var)
    return selected, best_auc


def vif_filter(woe_df: pd.DataFrame, cols: list, max_vif=None):
    """Iteratively drop the highest-VIF feature until all remaining VIF < threshold."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    max_vif = max_vif or cfg.VIF_ACCEPTABLE_THRESHOLD
    cols = list(cols)
    dropped = []
    while True:
        X = woe_df[cols].values
        vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({"feature": cols, "vif": vifs}).sort_values("vif", ascending=False)
        if vif_df["vif"].iloc[0] < max_vif or len(cols) <= 2:
            return cols, dropped, vif_df
        worst = vif_df["feature"].iloc[0]
        cols.remove(worst)
        dropped.append(worst)


def kmeans_segmentation(df: pd.DataFrame, cols: list, k=4):
    """Statistical risk segmentation profile (K-Means) for portfolio reporting."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = df[cols].fillna(df[cols].median())
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=10).fit(Xs)
    seg = df.copy()
    seg["Segment"] = km.labels_
    profile = seg.groupby("Segment").agg(
        accounts=("Segment", "size"),
        bad_rate=(cfg.TARGET_COL, "mean"),
        avg_income=("Income_INR", "mean"),
        avg_credit_limit=("Total_Credit_Limit", "mean"),
    ).reset_index()
    profile["bad_rate"] = (profile["bad_rate"] * 100).round(2)
    return seg, profile


def run():
    train = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "train_clean.csv"))
    test = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "test_clean.csv"))
    oot = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "oot_clean.csv"))

    drop = set(cfg.EXCLUDE_COLS + [cfg.TARGET_COL])
    feature_cols = [c for c in train.columns if c not in drop]

    print(f"Candidate features: {len(feature_cols)}")
    iv_df, binners = compute_woe_iv(train, feature_cols, cfg.TARGET_COL)
    iv_df.to_csv(os.path.join(cfg.REPORTS_DIR, "04_iv_summary.csv"), index=False)
    print("\nTop 15 by IV:\n", iv_df.head(15)[["feature", "iv", "dtype"]].to_string(index=False))

    kept_iv, pending_review = iv_filter(iv_df)
    print(f"\nKept after IV filter + business review: {len(kept_iv)}")
    if len(pending_review):
        print("Pending SME review (high IV, not on approved list) -> dropped this cycle:\n", pending_review[["feature", "iv"]].to_string(index=False))

    woe_train = transform_to_woe(train, kept_iv, binners)
    kept_corr, dropped_corr, corr_matrix = correlation_filter(woe_train, iv_df[iv_df.feature.isin(kept_iv)])
    print(f"\nDropped on correlation >{cfg.CORR_DROP_THRESHOLD}: {dropped_corr}")

    kept_vif, dropped_vif, vif_table = vif_filter(woe_train, kept_corr)
    print(f"Dropped on VIF >{cfg.VIF_ACCEPTABLE_THRESHOLD}: {dropped_vif}")
    vif_table.to_csv(os.path.join(cfg.REPORTS_DIR, "05_vif_final.csv"), index=False)

    final_vars, cv_auc = stepwise_forward_select(woe_train, train[cfg.TARGET_COL], kept_vif, max_vars=12)
    print(f"\nStepwise CV-AUC with final set: {cv_auc:.4f}")
    kept_vif = final_vars

    print(f"\nFinal variable shortlist ({len(kept_vif)}): {kept_vif}")

    # Persist WOE-transformed datasets for modelling stage
    woe_test = transform_to_woe(test, kept_vif, binners)
    woe_oot = transform_to_woe(oot, kept_vif, binners)
    woe_train_final = woe_train[kept_vif]

    for name, d, raw in [("train", woe_train_final, train), ("test", woe_test, test), ("oot", woe_oot, oot)]:
        out = d.copy()
        out[cfg.TARGET_COL] = raw[cfg.TARGET_COL].values
        out.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, f"{name}_woe.csv"), index=False)

    import pickle
    with open(os.path.join(cfg.MODELS_DIR, "binners.pkl"), "wb") as f:
        pickle.dump({k: v for k, v in binners.items() if k in kept_vif}, f)

    # Segmentation profiling (reporting layer, not a modelling input)
    _, profile = kmeans_segmentation(train, [c for c in ["Income_INR", "Total_Credit_Limit", "Credit_Utilization_Ratio", "Age"] if c in train.columns])
    profile.to_csv(os.path.join(cfg.REPORTS_DIR, "06_kmeans_segment_profile.csv"), index=False)
    print("\nK-Means segment profile:\n", profile.to_string(index=False))

    return kept_vif, binners


if __name__ == "__main__":
    run()
