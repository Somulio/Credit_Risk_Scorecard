"""
Bonus: Model Explainability (SHAP)
Generates SHAP summary values for the XGBoost challenger model so the
challenger's logic can be sanity-checked against the logistic scorecard's
WOE-driven coefficients (Step 6 of the methodology doc — "explainability"
is implicit in why banks prefer logistic regression, this module proves it
empirically by comparing against a black-box challenger).
"""
import os
import sys
import pickle
import pandas as pd
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def run():
    with open(os.path.join(cfg.MODELS_DIR, "xgb_challenger.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model, cols = bundle["model"], bundle["cols"]

    test_clean = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "test_clean.csv"))
    X_test = test_clean[cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    importance = pd.DataFrame({
        "feature": cols,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(os.path.join(cfg.REPORTS_DIR, "11_shap_feature_importance.csv"), index=False)

    print("XGBoost challenger — SHAP feature importance (mean |SHAP|):")
    print(importance.to_string(index=False))

    with open(os.path.join(cfg.MODELS_DIR, "test_shap_values.pkl"), "wb") as f:
        pickle.dump({"shap_values": shap_values, "X_test": X_test, "cols": cols}, f)

    return importance


if __name__ == "__main__":
    run()
