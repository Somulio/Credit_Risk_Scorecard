# Retail Credit PD Scorecard — End-to-End Pipeline

A production-style **Probability of Default (PD) scorecard** built from scratch on 10 linked banking source tables, following the retail credit-risk scorecard development methodology used by Indian bank credit risk teams (vintage analysis → WOE/IV → business-reviewed variable reduction → logistic regression → points-based scorecard → full validation suite).

This isn't a single Kaggle-style notebook — it's structured the way a model-development team would actually build and hand off a scorecard: numbered pipeline stages, a config file instead of hardcoded thresholds, a data-quality audit, a documented overfitting investigation, and a model validation report with a pilot/no-go verdict.

## Why this project is different

Most portfolio scorecards stop at "I got AUC 0.85." This one shows the *process* of getting there responsibly:
- An early iteration with all IV-surviving variables overfit badly (train Gini 83% → test Gini 10%). The fix (stepwise CV-based selection, regularization, capped bin complexity) is documented in [`notebooks/00_methodology_notes.md`](notebooks/00_methodology_notes.md).
- Several variables had suspiciously perfect separation between good/bad accounts — a giveaway of synthetic-data construction artifacts rather than real signal. They were flagged and excluded via an explicit business-review step instead of being kept just because the IV number looked good.
- The final model is benchmarked against an XGBoost challenger, with SHAP used as an independent sanity check that the simpler, regulator-friendly logistic model isn't missing signal a black-box model would find.

## Results

| Metric | Train | Test | OOT |
|---|---|---|---|
| AUC | 0.962 | 0.888 | 0.922 |
| Gini | 92.4% | 77.6% | 84.3% |
| KS | 84.3 | 65.6 | 75.5 |
| PSI (vs train) | — | 0.036 (Stable) | 0.098 (Stable) |

Rank ordering monotonic on test (bad rate 50% → 0% across deciles). Full write-up: [`reports/Model_Validation_Report.md`](reports/Model_Validation_Report.md).

## Pipeline

```
1. Data Sourcing & Merge        src/01_data_loader.py
2. Vintage Analysis & Sampling  src/02_vintage_sampling.py     (70:30 train/test + strict OOT)
3. Data Preparation             src/03_data_preparation.py     (missing values, winsorization, business rules)
4. WOE/IV + Segmentation +      src/04_woe_variable_selection.py
   Variable Reduction             (optbinning WOE, IV filter + business review, correlation, VIF, stepwise CV selection, K-Means segmentation)
5. Model Creation                src/05_model_training.py      (Logistic Regression + XGBoost challenger)
6. Scorecard Calibration         src/06_scorecard_calibration.py (Base 600 / PDO 20 / Odds 50:1)
7. Model Validation              src/07_model_validation.py    (AUC, Gini, KS, PSI, decile rank order, concordance)
8. Explainability                src/08_shap_explainability.py (SHAP cross-check vs logistic coefficients)
```

Run the whole thing:
```bash
pip install -r requirements.txt
python run_pipeline.py
```

Or step through `notebooks/PD_Scorecard_Walkthrough.ipynb` for the same pipeline with visualizations (vintage curves, IV ranking, score distributions, ROC curves, SHAP summary plot).

## Data

Synthetic retail lending dataset (1,000 accounts, 9.5% bad rate) spanning 10 linked tables: customer application, behavioral, bureau, loan account, transaction, repayment history, collateral, collection, NPA/default flags, and a pre-joined master modelling table. Mirrors the kind of feature-store output a bank's data engineering team would deliver to a model-dev team.

## Repo structure

```
config/                  Central config — all thresholds (IV, VIF, PSI, scorecard params) in one place
data/raw/                 10 source Excel files
data/processed/           Train/test/OOT splits, cleaned data, WOE-transformed data, scored output
src/                      Numbered pipeline modules (run standalone or via run_pipeline.py)
models/                   Trained logistic + XGBoost models, fitted WOE binners, scaling factors
reports/                  CSV outputs for every stage + Model_Validation_Report.md
notebooks/                Walkthrough notebook + methodology notes on the overfitting fix
run_pipeline.py           Orchestrates all 8 stages end-to-end
```

## Methodology reference

Full step-by-step technique list (WOE/IV formulas, scorecard scaling formulas, validation metric definitions and industry benchmarks) is in [`notebooks/00_methodology_notes.md`](notebooks/00_methodology_notes.md) and reflected directly in `config/config.py`.

## Tech stack

Python · pandas · scikit-learn · optbinning · XGBoost · SHAP · statsmodels

---
*Built as part of a Credit Risk Analyst portfolio — see also: IFRS 9 ECL engine, Credit Portfolio MIS, XGBoost Early Warning System (companion repos).*
