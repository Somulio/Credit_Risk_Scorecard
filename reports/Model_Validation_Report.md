# PD Scorecard — Model Validation Report

**Model:** Retail Credit PD Scorecard (Logistic Regression on WOE-transformed variables)
**Portfolio:** Retail lending (synthetic dataset, 1,000 accounts, 10 linked source tables)
**Prepared as part of:** Credit Risk Analyst portfolio project
**Methodology:** Standard 8-stage retail scorecard development lifecycle used by Indian bank credit risk teams (Data Sourcing → Target/Vintage → Data Prep → Segmentation → Variable Reduction → Logistic Regression → Scorecard Calibration → Validation)

---

## 1. Population & Target Definition

- **Bad definition:** `Default = 1` where DPD > 90 days (Ever-90+ proxy), consistent with RBI/Basel default convention.
- **Overall bad rate:** 9.5% (95/1,000 accounts).
- **Vintage anchor:** Loan `Disbursal_Date` (2020-01 to 2022-09). Quarterly vintage bad-rate analysis showed an elevated-risk cohort around 2021Q2–2022Q1 (bad rate 11–16%), consistent with a COVID-recovery-period credit stress pattern, used to justify a strict OOT cut.

## 2. Sampling

| Sample | Accounts | Bad rate | Window |
|---|---|---|---|
| Train | 624 | 9.94% | Disbursed before 2022-06, 70% stratified split |
| Test | 268 | 10.07% | Disbursed before 2022-06, 30% stratified split |
| OOT | 108 | 5.56% | Disbursed ≥ 2022-06 (strict out-of-time holdout) |

## 3. Data Preparation

- 7 collateral-related fields dropped for >40% missingness (expected — unsecured loans have no collateral by design).
- Remaining missing values: median (numeric) / mode (categorical) imputation, fit on train only.
- Outlier treatment: 1st/99th percentile winsorization, bounds learned on train and applied to test/OOT (no leakage).
- Business rule checks (age range, utilization/LTV bounds) — zero violations found.

## 4. Variable Reduction — IV, Business Review, Correlation, VIF

54 candidate features were IV-ranked. **Six fields showed implausibly strong separation** between good/bad populations (IV 1.6–5.5, near non-overlapping distributions) — e.g. `Negative_Balance_Days` (bad-mean 17.4 vs good-mean 2.7), `Avg_Balance_3M`, bureau inquiry counts, and `DPD_30`/`Credit_Card_Utilization`. This pattern is typical of a synthetic-data construction artifact rather than a believable real-world relationship, so these were **flagged for SME review and excluded this development cycle** rather than auto-included on IV alone — the same judgment call a model-dev team would apply under Step 5F (Business Review) before signing off on a variable.

Three moderately strong fields (`Income_INR`, `Savings_Account_Balance`, `Outstanding_Loans`, IV 1.1–1.4) were reviewed and **approved** — their separation is meaningful but the distributions still overlap substantially, consistent with genuine pre-default financial-capacity signal rather than leakage.

- Correlation filter (|r| > 0.8 on WOE scale): no drops needed in the final set.
- VIF: all final variables < 2.0 (well under the 5/10 preferred/acceptable bank thresholds) — see `reports/05_vif_final.csv`.
- **Forward stepwise selection** (5-fold CV AUC, greedy, capped at 12 variables) was used instead of including all IV-surviving variables, since the bad-account count (≈95) is small relative to the candidate list — this controls overfitting that an unconstrained variable set produced in an earlier iteration (train Gini 92% / test Gini 10% before stepwise selection; see `notebooks/00_methodology_notes.md`).

**Final 12 variables:** `Savings_Account_Balance`, `Income_INR`, `Outstanding_Loans`, `No_of_Open_Accounts`, `Employment_Years`, `Avg_Monthly_Salary_Credit`, `Cash_Withdrawal_3M`, `Interest_Rate_Pct`, `Age`, `ECS_Mandate_Count`, `Newest_Trade_Open_Months`, `Total_EMI_Paid`.

## 5. Model

Logistic Regression on WOE-transformed variables, `C=0.5` (L2-regularized to control overfitting on the small bad-account count).

- **100% of coefficients carry the expected negative sign** (higher WOE / lower risk → lower log-odds of default) — a hard pass/fail check banks run before sign-off, since a wrong-signed variable usually indicates a binning or business-logic error.
- An XGBoost challenger model was trained in parallel on the same 12 raw features. SHAP feature importance ranks the same top-3 drivers (`Savings_Account_Balance`, `Income_INR`, `Outstanding_Loans`) as the logistic model's largest coefficients — a useful cross-check that the simpler, regulator-friendly model isn't missing the signal a black-box model would find.

## 6. Scorecard Calibration

| Parameter | Value |
|---|---|
| Base Score | 600 |
| PDO (points to double odds) | 20 |
| Base Odds | 50:1 |
| Factor | 28.85 |
| Offset | 487.12 |

Score range produced: **427–736** (train), centred close to the 600 base score with a long right tail for the lowest-risk accounts.

## 7. Validation Results

| Metric | Train | Test | OOT | Industry Benchmark |
|---|---|---|---|---|
| AUC | 0.962 | 0.888 | 0.922 | >0.70 Good |
| **Gini** | **92.4%** | **77.6%** | **84.3%** | >40% Good |
| **KS** | **84.3** | **65.6** | **75.5** | >30 Good, >40 Excellent |
| Concordance | — | 90.0% | — | — |

- **Rank ordering:** Bad rate falls monotonically from 50% in the lowest score decile to 0% in the top deciles on the test set — **PASS**.
- **PSI** (population stability): Train-vs-Test = 0.036, Train-vs-OOT = 0.098 — both **Stable** (<0.10 threshold), i.e. the scoring population hasn't drifted between development and the OOT window.
- Train→Test→OOT performance is consistent (no severe overfitting gap), which is the main credibility check for a 12-variable model on a 1,000-account sample.

## 8. Verdict

**Conditional Approval for Pilot** — model shows strong, stable, rank-ordered discrimination across train/test/OOT with no sign errors and acceptable VIF/PSI. Before production deployment on a real portfolio, recommend: (a) re-validate on a larger, non-synthetic sample where bad-rate separation is less extreme; (b) re-run the SME review on the six excluded high-IV variables with the data engineering team to confirm whether they are genuinely usable bureau/behavioral signals or generation artifacts; (c) quarterly PSI monitoring once live.
