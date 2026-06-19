# Credit Risk PD Scorecard

End-to-end Probability-of-Default (PD) scorecard pipeline built on synthetic
retail-lending data, following standard Basel/IFRS 9 credit risk modelling
practice used in Indian banking (DPD/MOB-based default definition, WOE/IV
feature engineering, logistic-regression scorecard with an XGBoost
challenger, KS/Gini/AUC evaluation, PSI stability monitoring, and a
simplified IFRS 9 ECL staging engine).

## Pipeline stages

1. **Data ingestion** – customer application, behavioural, bureau, loan,
   transaction, repayment, collateral, collection and macro-economic data
   merged into a single Analytical Base Table (`data/processed/01_abt_raw.csv`).
2. **Train / Test / OOT split** – chronological out-of-time holdout for
   realistic validation (`02_train/test/oot.csv`).
3. **WOE / IV binning** – monotonic optimal binning via `optbinning`,
   Information Value used to shortlist predictive, non-leaky features
   (`reports/iv_summary.csv`).
4. **Modelling** – Logistic Regression on WOE scale (interpretable,
   points-based scorecard) plus an XGBoost challenger model.
5. **Evaluation** – AUC-ROC, Gini, KS statistic on train/test/OOT, and
   Population Stability Index (PSI) between train and OOT score
   distributions (`reports/model_performance.json`).
6. **Scorecard scaling** – PDO-based points allocation
   (`reports/scorecard_points.json`).
7. **IFRS 9 ECL** – DPD-based IRAC staging (Stage 1/2/3) and 12-month /
   lifetime Expected Credit Loss computation (`src/ifrs9_ecl.py`).

## Repository structure

```
.
├── data/
│   ├── raw/            # source extracts (customer, bureau, loan, txn, ...)
│   └── processed/      # ABT, train/test/OOT splits, WOE-prepped sets
├── src/
│   ├── config.py           # paths, target, thresholds
│   ├── data_loader.py       # load splits into X / y
│   ├── woe_binning.py       # WOE/IV binning + feature shortlist
│   ├── model_training.py    # Logistic Regression + XGBoost
│   ├── evaluation.py        # AUC, Gini, KS, PSI
│   ├── scorecard.py         # log-odds -> points scaling
│   ├── ifrs9_ecl.py         # IRAC staging + ECL
│   └── run_pipeline.py      # orchestrates steps 1-6 end to end
├── tests/                   # pytest unit tests for metrics/scorecard
├── models/                  # serialized model artefacts (git-ignored)
├── reports/                  # IV summary, performance, scorecard points
├── notebooks/                # exploratory analysis
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

This trains the scorecard and challenger model, evaluates them on train /
test / out-of-time data, and writes all artefacts to `models/` and
`reports/`.

Run tests:
```bash
pytest tests/ -q
```

## Results (synthetic demo dataset)

| Split | Model | AUC-ROC | Gini | KS |
|-------|-------|---------|------|----|
| Train | Logistic | ~0.95 | ~0.90 | ~0.79 |
| Train | XGBoost | ~0.99 | ~0.99 | ~0.95 |
| Test  | Logistic | ~0.60 | ~0.21 | ~0.19 |
| OOT   | Logistic | ~0.42 | -0.16 | ~0.31 |

**Note on results:** the bundled dataset is a small (≈400-row) synthetic
sample generated for portfolio/demo purposes, so train metrics are
optimistic and out-of-time performance degrades — exactly the kind of
overfitting and population drift a real validation/monitoring process
(PSI, KS-on-OOT) is designed to catch. The pipeline, metrics, and
governance artefacts are production-pattern; only the data volume is a
toy substitute for a real loan book.

## Tech stack

Python, pandas, scikit-learn, optbinning, XGBoost, pytest.

## License

MIT — see [LICENSE](LICENSE).
