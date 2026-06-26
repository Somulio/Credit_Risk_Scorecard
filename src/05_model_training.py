"""
Step 6: Model Creation
Logistic Regression on WOE-transformed features (industry-standard scorecard model),
plus an XGBoost challenger model for benchmarking / SHAP explainability.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def train_logistic(train_woe: pd.DataFrame, target_col: str):
    X = train_woe.drop(columns=[target_col])
    y = train_woe[target_col]
    model = LogisticRegression(solver="liblinear", C=0.5, random_state=cfg.RANDOM_STATE)
    model.fit(X, y)
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient")
    return model, coefs


def train_xgboost_challenger(train_raw: pd.DataFrame, feature_cols: list, target_col: str):
    import xgboost as xgb
    X = train_raw[feature_cols].select_dtypes(include=[np.number])
    y = train_raw[target_col]
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=cfg.RANDOM_STATE,
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def run():
    train_woe = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "train_woe.csv"))

    model, coefs = train_logistic(train_woe, cfg.TARGET_COL)
    print("Logistic Regression coefficients (sign should be negative -> higher WOE = lower risk):")
    print(coefs.to_string(index=False))

    neg_sign_ok = (coefs["coefficient"] < 0).mean() * 100
    print(f"\n% features with expected negative sign: {neg_sign_ok:.1f}%")

    with open(os.path.join(cfg.MODELS_DIR, "logistic_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    coefs.to_csv(os.path.join(cfg.REPORTS_DIR, "07_logistic_coefficients.csv"), index=False)

    # Challenger model on raw (non-WOE) numeric features for SHAP / benchmarking
    train_clean = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "train_clean.csv"))
    feature_cols = [c for c in train_woe.columns if c != cfg.TARGET_COL]
    xgb_model, xgb_cols = train_xgboost_challenger(train_clean, feature_cols, cfg.TARGET_COL)
    with open(os.path.join(cfg.MODELS_DIR, "xgb_challenger.pkl"), "wb") as f:
        pickle.dump({"model": xgb_model, "cols": xgb_cols}, f)
    print(f"\nXGBoost challenger trained on {len(xgb_cols)} raw numeric features.")

    return model, xgb_model


if __name__ == "__main__":
    run()
