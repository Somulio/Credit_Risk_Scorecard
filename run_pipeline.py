"""
Master pipeline orchestrator — runs the full PD scorecard pipeline end-to-end.

Usage:
    python run_pipeline.py

Mirrors the 11-step industry methodology:
1. Data Source Identification & Merging      -> src/01_data_loader.py
2. Target / Vintage Analysis / Sampling      -> src/02_vintage_sampling.py
3. Data Preparation (missing/outlier)        -> src/03_data_preparation.py
4. Segmentation + WOE/IV + Variable Reduction -> src/04_woe_variable_selection.py
5. Model Creation (Logistic Regression)      -> src/05_model_training.py
6. Scorecard Development & Calibration       -> src/06_scorecard_calibration.py
7. Model Validation                          -> src/07_model_validation.py
8. SHAP Explainability (challenger check)    -> src/08_shap_explainability.py
"""
import importlib.util
import os
import sys
import time

STEPS = [
    "01_data_loader",
    "02_vintage_sampling",
    "03_data_preparation",
    "04_woe_variable_selection",
    "05_model_training",
    "06_scorecard_calibration",
    "07_model_validation",
    "08_shap_explainability",
]


def run_step(name):
    path = os.path.join("src", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run()


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    for step in STEPS:
        print("\n" + "=" * 80)
        print(f"STEP: {step}")
        print("=" * 80)
        t0 = time.time()
        run_step(step)
        print(f"\n[{step}] completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE — see reports/ for all outputs, models/ for trained artifacts")
    print("=" * 80)


if __name__ == "__main__":
    main()
