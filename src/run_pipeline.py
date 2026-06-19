"""
End-to-end Credit Risk PD Scorecard pipeline.
Run: python -m src.run_pipeline
"""
import json
import pandas as pd

from . import config, data_loader, woe_binning, model_training, evaluation, scorecard


def main():
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    print("[1/6] Loading train / test / OOT splits ...")
    (X_train, y_train), (X_test, y_test), (X_oot, y_oot) = data_loader.load_all()

    print("[2/6] Fitting WOE binning + computing Information Value ...")
    bp = woe_binning.fit_binning(X_train, y_train)
    iv_df = woe_binning.iv_table(bp)
    woe_binning.save_iv_report(iv_df, config.REPORTS_DIR / "iv_summary.csv")

    selected = woe_binning.select_features(iv_df)
    woe_binning.save_shortlist(selected, config.DATA_PROCESSED / "05_final_shortlist.json")
    print(f"    Selected {len(selected)} features (IV between "
          f"{config.IV_MIN_THRESHOLD} and {config.IV_MAX_THRESHOLD})")

    print("[3/6] Transforming features to WOE scale ...")
    X_train_woe = woe_binning.transform_woe(bp, X_train, selected)
    X_test_woe = woe_binning.transform_woe(bp, X_test, selected)
    X_oot_woe = woe_binning.transform_woe(bp, X_oot, selected)

    print("[4/6] Training Logistic Regression (scorecard) + XGBoost (challenger) ...")
    logit = model_training.train_logistic(X_train_woe, y_train)
    xgb = model_training.train_xgboost(X_train_woe, y_train)
    model_training.save_model(logit, config.MODELS_DIR / "logistic_scorecard.pkl")
    model_training.save_model(xgb, config.MODELS_DIR / "xgboost_challenger.pkl")
    model_training.save_model(bp, config.MODELS_DIR / "woe_binning_process.pkl")

    print("[5/6] Evaluating on train / test / OOT ...")
    results = {}
    for name, (Xw, y) in {
        "train": (X_train_woe, y_train),
        "test": (X_test_woe, y_test),
        "oot": (X_oot_woe, y_oot),
    }.items():
        p_logit = logit.predict_proba(Xw)[:, 1]
        p_xgb = xgb.predict_proba(Xw)[:, 1]
        results[name] = {
            "logistic": evaluation.evaluate(y, p_logit),
            "xgboost": evaluation.evaluate(y, p_xgb),
        }

    psi_score = evaluation.psi(
        pd.Series(logit.predict_proba(X_train_woe)[:, 1]),
        pd.Series(logit.predict_proba(X_oot_woe)[:, 1]),
    )
    results["stability"] = {"PSI_train_vs_oot": round(psi_score, 4)}

    with open(config.REPORTS_DIR / "model_performance.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[6/6] Building scorecard points table ...")
    points = scorecard.build_points_table(logit, selected)
    with open(config.REPORTS_DIR / "scorecard_points.json", "w") as f:
        json.dump(points, f, indent=2)

    print("\nDone. Results summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
