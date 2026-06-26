"""
Central configuration for the PD Scorecard pipeline.
All thresholds/parameters used across notebooks & src modules live here
so the project behaves like a real bank model-dev config, not hardcoded magic numbers.
"""

# ---------------- Paths ----------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

RAW_FILES = {
    "application": "01_customer_application_data_new.xlsx",
    "behavioral": "02_behavioral_data_new.xlsx",
    "bureau": "03_bureau_data_new.xlsx",
    "loan": "04_loan_account_data_new.xlsx",
    "transaction": "05_transaction_data_new.xlsx",
    "repayment": "06_repayment_history_data_new.xlsx",
    "collateral": "07_collateral_data_new.xlsx",
    "collection": "08_collection_data_new.xlsx",
    "npa": "09_npa_default_flags_new.xlsx",
    "master": "10_master_modelling_table_new.xlsx",
}

# ---------------- Target Definition ----------------
TARGET_COL = "Default"          # Default = DPD > 90 days (bad=1)
ID_COLS = ["Customer_ID", "Loan_ID"]
DATE_COL = "Disbursal_Date"     # vintage / OOT anchor (loan booking date)
DEFAULT_DATE_COL = "Default_Date"  # only populated for bad accounts; not used for vintage cut

# ---------------- Vintage / Sampling ----------------
# OOT window held out for true out-of-time validation
OOT_START_DATE = "2022-06-01"
TRAIN_TEST_SPLIT_RATIO = 0.30   # 70:30
RANDOM_STATE = 42

# ---------------- Data Prep ----------------
MISSING_DROP_THRESHOLD = 0.40   # drop variable if >40% missing
OUTLIER_LOWER_PCT = 0.01
OUTLIER_UPPER_PCT = 0.99

# ---------------- Variable Reduction ----------------
IV_MIN_THRESHOLD = 0.02         # drop variables below this
IV_SUSPICIOUS_THRESHOLD = 1.5   # flag variables above this for SME review

# Variables that look "too predictive" by raw IV but are legitimate PRE-default
# behavioral/bureau signals (current snapshot, not a default outcome). A real
# scorecard team reviews each one against this list during Step 5F (Business Review)
# rather than auto-dropping on IV alone.
#
# NOTE: on inspection, several very-high-IV fields (Negative_Balance_Days,
# Avg_Balance_3M/6M, No_of_Inquiries_6M/12M, Months_Since_Most_Recent_Delinquency,
# DPD_30, Credit_Card_Utilization) show near-total separation between good/bad
# populations (e.g. Negative_Balance_Days: bad-mean 17.4 vs good-mean 2.7, almost
# non-overlapping) -- a pattern typical of a synthetic-data construction artifact
# rather than a real-world relationship. These are EXCLUDED pending further
# investigation into how the dataset was generated, even though their raw IV
# would clear the bar. Only fields with believable real-world separation
# (overlapping distributions, std comparable to the mean gap) are approved below.
BUSINESS_APPROVED_HIGH_IV_VARS = [
    "Outstanding_Loans", "Income_INR", "Savings_Account_Balance",
]
CORR_DROP_THRESHOLD = 0.80
VIF_PREFERRED_THRESHOLD = 5
VIF_ACCEPTABLE_THRESHOLD = 10

# ---------------- Scorecard Calibration ----------------
BASE_SCORE = 600
PDO = 20                        # points to double the odds
BASE_ODDS = 50                  # 50:1 good:bad at base score

# ---------------- Validation Thresholds ----------------
KS_GOOD_THRESHOLD = 30
KS_EXCELLENT_THRESHOLD = 40
GINI_GOOD_THRESHOLD = 0.40
PSI_STABLE_THRESHOLD = 0.10
PSI_MONITOR_THRESHOLD = 0.25

# Leakage / post-default columns that must NEVER enter the model
EXCLUDE_COLS = [
    "Customer_ID", "Loan_ID", "Default_Date", "Disbursal_Date", "Loan_Status",
    "NPA_Flag",
    "IRAC_Classification", "Default_Amount", "Recovery_Amount",
    "Loss_Amount", "LGD", "Legal_Action", "Days_In_Collection",
    "In_Collection_Flag", "Collection_Calls", "Visit_Count",
    "Promise_To_Pay", "Worst_Current_Status",
]
