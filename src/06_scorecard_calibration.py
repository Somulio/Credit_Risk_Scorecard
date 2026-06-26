"""
Step 7: Scorecard Development & Calibration
Converts the logistic regression's log-odds into a points-based scorecard
using the standard Base Score / PDO / Base Odds formulation.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def scaling_factors():
    factor = cfg.PDO / np.log(2)
    offset = cfg.BASE_SCORE - factor * np.log(cfg.BASE_ODDS)
    return factor, offset


def build_scorecard_points(model, woe_binners: dict, feature_cols: list):
    """Points table: every bin of every feature gets a score contribution."""
    factor, offset = scaling_factors()
    n = len(feature_cols)
    intercept = model.intercept_[0]
    coefs = dict(zip(feature_cols, model.coef_[0]))

    rows = []
    for feat in feature_cols:
        binning_table = woe_binners[feat].binning_table.build()
        beta = coefs[feat]
        for _, row in binning_table.iterrows():
            if row.name == "Totals":
                continue
            woe = row["WoE"]
            points = -(beta * woe + intercept / n) * factor + offset / n
            rows.append({
                "feature": feat, "bin": row.name, "count": row.get("Count", np.nan),
                "woe": woe, "points": round(points, 1),
            })
    return pd.DataFrame(rows), factor, offset


def score_from_probability(pd_prob: np.ndarray, factor: float, offset: float) -> np.ndarray:
    """Score = Offset + Factor * ln(odds), odds = (1-PD)/PD"""
    pd_prob = np.clip(pd_prob, 1e-6, 1 - 1e-6)
    odds = (1 - pd_prob) / pd_prob
    return offset + factor * np.log(odds)


def run():
    with open(os.path.join(cfg.MODELS_DIR, "logistic_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(cfg.MODELS_DIR, "binners.pkl"), "rb") as f:
        binners = pickle.load(f)

    train_woe = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "train_woe.csv"))
    feature_cols = [c for c in train_woe.columns if c != cfg.TARGET_COL]

    points_table, factor, offset = build_scorecard_points(model, binners, feature_cols)
    points_table.to_csv(os.path.join(cfg.REPORTS_DIR, "08_scorecard_points_table.csv"), index=False)
    print(f"Factor={factor:.3f}, Offset={offset:.3f}  (Base={cfg.BASE_SCORE}, PDO={cfg.PDO}, BaseOdds={cfg.BASE_ODDS}:1)")
    print(f"\nScorecard points table ({len(points_table)} bins):")
    print(points_table.head(15).to_string(index=False))

    for name in ["train", "test", "oot"]:
        woe = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, f"{name}_woe.csv"))
        X = woe[feature_cols]
        pd_pred = model.predict_proba(X)[:, 1]
        score = score_from_probability(pd_pred, factor, offset)
        out = woe[[cfg.TARGET_COL]].copy()
        out["PD"] = pd_pred
        out["Score"] = score.round(0).astype(int)
        out.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, f"{name}_scored.csv"), index=False)
        print(f"\n{name}: score range [{out['Score'].min()}, {out['Score'].max()}], mean PD {out['PD'].mean()*100:.2f}%")

    with open(os.path.join(cfg.MODELS_DIR, "scaling.pkl"), "wb") as f:
        pickle.dump({"factor": factor, "offset": offset}, f)


if __name__ == "__main__":
    run()
