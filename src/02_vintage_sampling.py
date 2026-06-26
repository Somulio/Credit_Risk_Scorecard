"""
Step 2: Target Variable Creation, Vintage Analysis & Sampling
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def vintage_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly vintage bad-rate table used for OOT cut-off selection."""
    df = df.copy()
    df["Vintage_Quarter"] = pd.to_datetime(df[cfg.DATE_COL]).dt.to_period("Q")
    vintage = (
        df.groupby("Vintage_Quarter")[cfg.TARGET_COL]
        .agg(accounts="count", bads="sum")
        .assign(bad_rate=lambda x: (x["bads"] / x["accounts"] * 100).round(2))
        .reset_index()
    )
    return vintage


def train_test_oot_split(df: pd.DataFrame):
    """
    Stratified 70:30 train/test split on the in-time population,
    plus a strict Out-of-Time holdout for the most recent vintage quarters.
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL])

    oot_mask = df[cfg.DATE_COL] >= pd.Timestamp(cfg.OOT_START_DATE)
    oot = df[oot_mask].reset_index(drop=True)
    in_time = df[~oot_mask].reset_index(drop=True)

    train, test = train_test_split(
        in_time,
        test_size=cfg.TRAIN_TEST_SPLIT_RATIO,
        stratify=in_time[cfg.TARGET_COL],
        random_state=cfg.RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), oot.reset_index(drop=True)


def run():
    master = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "01_merged_master.csv"))

    vintage = vintage_analysis(master)
    vintage.to_csv(os.path.join(cfg.REPORTS_DIR, "02_vintage_analysis.csv"), index=False)
    print("Vintage / quarterly bad-rate table:")
    print(vintage.to_string(index=False))

    train, test, oot = train_test_oot_split(master)
    print(f"\nTrain: {len(train)} ({train[cfg.TARGET_COL].mean()*100:.2f}% bad)")
    print(f"Test : {len(test)} ({test[cfg.TARGET_COL].mean()*100:.2f}% bad)")
    print(f"OOT  : {len(oot)} ({oot[cfg.TARGET_COL].mean()*100:.2f}% bad)  [>= {cfg.OOT_START_DATE}]")

    train.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "test.csv"), index=False)
    oot.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "oot.csv"), index=False)
    return train, test, oot


if __name__ == "__main__":
    run()
