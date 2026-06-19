"""Central configuration for the Credit Risk Scorecard pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "Bad_Flag"
ID_COLS = ["index", "Loan_ID", "Customer_ID", "Disbursal_Date", "Vintage_Q",
           "Loan_Status", "Default", "Bad_A", "Bad_B", "Good_Flag"]

TRAIN_FILE = DATA_PROCESSED / "03_train_prepped.csv"
TEST_FILE = DATA_PROCESSED / "03_test_prepped.csv"
OOT_FILE = DATA_PROCESSED / "03_oot_prepped.csv"

RANDOM_STATE = 42
IV_MIN_THRESHOLD = 0.02   # below this, feature is dropped
IV_MAX_THRESHOLD = 0.5    # above this, suspect of leakage
SCORECARD_PDO = 20        # points to double the odds
SCORECARD_BASE_SCORE = 600
SCORECARD_BASE_ODDS = 50
