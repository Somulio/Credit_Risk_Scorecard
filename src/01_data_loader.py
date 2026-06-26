"""
Step 1: Data Source Identification & Merging
Loads all 9 source tables, joins them on Customer_ID / Loan_ID into a single
modelling-ready dataframe, and performs basic quality checks (dupes, ID integrity).
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def _path(name):
    return os.path.join(cfg.RAW_DATA_DIR, cfg.RAW_FILES[name])


def load_raw_tables() -> dict:
    """Load every raw Excel source into a dict of DataFrames."""
    tables = {}
    for key, fname in cfg.RAW_FILES.items():
        fp = os.path.join(cfg.RAW_DATA_DIR, fname)
        tables[key] = pd.read_excel(fp)
    return tables


def quality_checks(tables: dict) -> pd.DataFrame:
    """Duplicate / null / row-count audit across all source tables."""
    rows = []
    for name, df in tables.items():
        rows.append({
            "table": name,
            "rows": len(df),
            "cols": df.shape[1],
            "dup_rows": df.duplicated().sum(),
            "dup_customer_id": df["Customer_ID"].duplicated().sum() if "Customer_ID" in df.columns else None,
            "null_cells": int(df.isnull().sum().sum()),
        })
    return pd.DataFrame(rows)


def build_master_table(tables: dict) -> pd.DataFrame:
    """
    Use the pre-joined master_modelling_table as the modelling base
    (this mirrors how a bank's IT/feature-store team would deliver a
    flattened Customer_ID + Loan_ID grain table), then enrich with a
    few extra fields from satellite tables that aren't already present
    (transaction behaviour, collateral, repayment depth).
    """
    master = tables["master"].copy()

    disbursal = tables["loan"][["Loan_ID", "Disbursal_Date", "Loan_Status"]]
    master = master.merge(disbursal, on="Loan_ID", how="left")

    txn_extra = tables["transaction"][[
        "Customer_ID", "Avg_Balance_3M", "Salary_Credit_Frequency",
        "Avg_Monthly_Salary_Credit", "Monthly_Spend", "Cash_Withdrawal_3M",
        "Merchant_Spend_Pct", "Negative_Balance_Days", "UPI_Txn_Count_3M",
        "ECS_Mandate_Count",
    ]]
    master = master.merge(txn_extra, on="Customer_ID", how="left")

    repay_extra = tables["repayment"][[
        "Loan_ID", "Paid_On_Time_Pct", "Late_Payment_Count_12M",
        "Max_DPD_3M", "Max_DPD_6M", "Current_DPD_Bucket",
        "Prev_DPD_Bucket", "Prepayment_Count", "Total_EMI_Paid",
    ]]
    master = master.merge(repay_extra, on="Loan_ID", how="left")

    collateral_extra = tables["collateral"][[
        "Loan_ID", "Collateral_Type", "Property_Type",
        "Vehicle_Age_Years", "Collateral_Location_Tier", "Collateral_Insurance",
    ]]
    master = master.merge(collateral_extra, on="Loan_ID", how="left")

    bureau_extra = tables["bureau"][[
        "Customer_ID", "No_of_Closed_Accounts", "No_of_Inquiries_12M",
        "Months_Since_Most_Recent_Delinquency", "Max_Credit_Exposure",
        "Newest_Trade_Open_Months",
    ]]
    master = master.merge(bureau_extra, on="Customer_ID", how="left")

    return master


def run():
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    tables = load_raw_tables()
    qc = quality_checks(tables)
    qc.to_csv(os.path.join(cfg.REPORTS_DIR, "01_data_quality_audit.csv"), index=False)

    master = build_master_table(tables)
    master.to_csv(os.path.join(cfg.PROCESSED_DATA_DIR, "01_merged_master.csv"), index=False)

    print(f"Merged master table: {master.shape[0]} rows x {master.shape[1]} cols")
    print(qc.to_string(index=False))
    return master


if __name__ == "__main__":
    run()
