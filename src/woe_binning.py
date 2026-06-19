"""WOE / Information Value binning and transformation using optbinning."""
import json
import pandas as pd
from optbinning import BinningProcess
from . import config


def fit_binning(X: pd.DataFrame, y: pd.Series) -> BinningProcess:
    """Fit a monotonic WOE binning process across all features."""
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    bp = BinningProcess(
        variable_names=X.columns.tolist(),
        categorical_variables=cat_cols,
        min_bin_size=0.05,
        max_n_bins=6,
    )
    bp.fit(X, y)
    return bp


def iv_table(bp: BinningProcess) -> pd.DataFrame:
    """Return a sorted Information Value summary for every feature."""
    summary = bp.summary()
    summary = summary.rename(columns={"iv": "IV"}).sort_values("IV", ascending=False)
    return summary[["name", "dtype", "n_bins", "IV"]]


def select_features(iv_df: pd.DataFrame, top_n: int = 15) -> list:
    """Keep the top-N features within the IV sweet spot (predictive, not leaky)."""
    mask = (iv_df["IV"] >= config.IV_MIN_THRESHOLD) & (iv_df["IV"] <= config.IV_MAX_THRESHOLD)
    return iv_df.loc[mask, "name"].head(top_n).tolist()


def transform_woe(bp: BinningProcess, X: pd.DataFrame, features: list) -> pd.DataFrame:
    woe_full = bp.transform(X, metric="woe")
    woe_full.columns = X.columns
    return woe_full[features]


def save_iv_report(iv_df: pd.DataFrame, path):
    iv_df.to_csv(path, index=False)


def save_shortlist(features: list, path):
    with open(path, "w") as f:
        json.dump({"selected_features": features}, f, indent=2)
