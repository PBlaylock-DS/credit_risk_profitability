"""
combine_auto_finance_files.py

Creates auto_finance_combined_over_500k.csv from:
- financial_loan.csv
- automobile_loan_default_1.csv
- automobile_loan_default_2.csv

The script standardizes column names, creates engineered features, adds source tracking,
and uses bootstrap-based synthetic expansion to reach >500,000 records.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 527
TARGET_ROWS = 500_001
rng = np.random.default_rng(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent

# Support both original filenames and uploaded names with (1)
FINANCIAL_FILE = BASE_DIR / "financial_loan.csv"
AUTO1_FILE = BASE_DIR / "automobile_loan_default_1.csv"
AUTO2_FILE = BASE_DIR / "automobile_loan_default_2.csv"

if not FINANCIAL_FILE.exists():
    FINANCIAL_FILE = BASE_DIR / "financial_loan(1).csv"
if not AUTO1_FILE.exists():
    AUTO1_FILE = BASE_DIR / "automobile_loan_default_1(1).csv"
if not AUTO2_FILE.exists():
    AUTO2_FILE = BASE_DIR / "automobile_loan_default_2(1).csv"

OUTPUT_FILE = BASE_DIR / "auto_finance_combined_over_500k.csv"


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a messy numeric column to numeric values."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def term_to_months(series: pd.Series) -> pd.Series:
    """Extract numeric term months from values such as '36 months'."""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def standardize_financial_loan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["record_id"] = df.get("id")
    out["source"] = "financial_loan"
    out["record_origin"] = "original"
    out["income"] = clean_numeric(df.get("annual_income"))
    out["loan_amount"] = clean_numeric(df.get("loan_amount"))
    out["int_rate"] = clean_numeric(df.get("int_rate"))
    # If interest rate was represented as whole percent, convert to decimal.
    out.loc[out["int_rate"] > 1, "int_rate"] = out.loc[out["int_rate"] > 1, "int_rate"] / 100
    out["term_months"] = term_to_months(df.get("term"))
    out["installment"] = clean_numeric(df.get("installment"))
    out["monthly_payment_est"] = out["installment"]
    out["total_payment"] = clean_numeric(df.get("total_payment"))
    out["dti"] = clean_numeric(df.get("dti"))
    out["loan_status"] = df.get("loan_status").astype(str)
    default_statuses = ["charged off", "default", "late", "does not meet the credit policy"]
    out["default"] = out["loan_status"].str.lower().apply(lambda x: int(any(s in x for s in default_statuses)))
    out["home_ownership"] = df.get("home_ownership")
    out["employment_length"] = df.get("emp_length")
    out["purpose"] = df.get("purpose")
    out["grade"] = df.get("grade")
    out["credit_bureau"] = np.nan
    out["score_source_1"] = np.nan
    out["score_source_2"] = np.nan
    out["score_source_3"] = np.nan
    out["client_income_type"] = np.nan
    out["client_education"] = np.nan
    out["client_marital_status"] = np.nan
    out["client_gender"] = np.nan
    out["car_owned"] = np.nan
    out["bike_owned"] = np.nan
    out["active_loan"] = np.nan
    out["house_own"] = np.nan
    out["child_count"] = np.nan
    out["age_years"] = np.nan
    out["employed_years"] = np.nan
    return out


def standardize_vehicle_loan(path: Path, source_name: str, has_default: bool) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["record_id"] = df.get("ID")
    out["source"] = source_name
    out["record_origin"] = "original"
    out["income"] = clean_numeric(df.get("Client_Income"))
    out["loan_amount"] = clean_numeric(df.get("Credit_Amount"))
    out["int_rate"] = np.nan
    out["term_months"] = np.nan
    out["installment"] = clean_numeric(df.get("Loan_Annuity"))
    out["monthly_payment_est"] = out["installment"]
    out["total_payment"] = np.nan
    out["dti"] = np.nan
    out["loan_status"] = np.nan
    if has_default and "Default" in df.columns:
        out["default"] = clean_numeric(df["Default"]).fillna(0).clip(0, 1).astype(int)
    else:
        out["default"] = np.nan
    out["home_ownership"] = df.get("Client_Housing_Type")
    out["employment_length"] = np.nan
    out["purpose"] = "vehicle_loan"
    out["grade"] = np.nan
    out["credit_bureau"] = clean_numeric(df.get("Credit_Bureau"))
    out["score_source_1"] = clean_numeric(df.get("Score_Source_1"))
    out["score_source_2"] = clean_numeric(df.get("Score_Source_2"))
    # Score_Source_2 contains some extreme values in this dataset; cap to expected score range.
    out.loc[out["score_source_2"] > 1, "score_source_2"] = np.nan
    out["score_source_3"] = clean_numeric(df.get("Score_Source_3"))
    out["client_income_type"] = df.get("Client_Income_Type")
    out["client_education"] = df.get("Client_Education")
    out["client_marital_status"] = df.get("Client_Marital_Status")
    out["client_gender"] = df.get("Client_Gender")
    out["car_owned"] = clean_numeric(df.get("Car_Owned"))
    out["bike_owned"] = clean_numeric(df.get("Bike_Owned"))
    out["active_loan"] = clean_numeric(df.get("Active_Loan"))
    out["house_own"] = clean_numeric(df.get("House_Own"))
    out["child_count"] = clean_numeric(df.get("Child_Count"))
    out["age_years"] = (clean_numeric(df.get("Age_Days")).abs() / 365.25).round(1)
    out["employed_years"] = (clean_numeric(df.get("Employed_Days")).abs() / 365.25).round(1)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["income", "loan_amount", "int_rate", "term_months", "installment", "monthly_payment_est", "default"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill core numeric fields with robust medians.
    for col in ["income", "loan_amount", "installment"]:
        median = df[col].median(skipna=True)
        df[col] = df[col].fillna(median)

    df["int_rate"] = df["int_rate"].fillna(df["int_rate"].median(skipna=True))
    if df["int_rate"].isna().all():
        df["int_rate"] = 0.12
    df["term_months"] = df["term_months"].fillna(60)
    df["monthly_payment_est"] = df["monthly_payment_est"].fillna(df["loan_amount"] / df["term_months"].replace(0, np.nan))

    df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_to_income"].replace([np.inf, -np.inf], np.nan).fillna(df["loan_to_income"].median())

    # Create a default proxy only where source records did not have a default label.
    missing_default = df["default"].isna()
    if missing_default.any():
        risk_score = (
            0.08
            + 0.10 * (df.loc[missing_default, "loan_to_income"] > df["loan_to_income"].median()).astype(float)
            + 0.04 * (df.loc[missing_default, "active_loan"].fillna(0) == 1).astype(float)
            + 0.04 * (df.loc[missing_default, "credit_bureau"].fillna(0) > df["credit_bureau"].median(skipna=True)).astype(float)
        ).clip(0.03, 0.35)
        df.loc[missing_default, "default"] = rng.binomial(1, risk_score)
    df["default"] = df["default"].fillna(0).astype(int)

    income_median = df["income"].median()
    df["borrower_segment"] = np.where(df["income"] >= income_median, "High Income", "Low Income")
    df["risk_band"] = pd.cut(
        df["loan_to_income"],
        bins=[-np.inf, 0.20, 0.40, np.inf],
        labels=["Low Burden", "Moderate Burden", "Elevated Burden"],
    ).astype(str)
    return df


def create_synthetic_rows(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    if len(df) >= target_rows:
        return df.head(target_rows).copy()

    needed = target_rows - len(df)
    synth = df.sample(n=needed, replace=True, random_state=RANDOM_SEED).copy().reset_index(drop=True)
    synth["record_origin"] = "synthetic_bootstrap"
    synth["record_id"] = [f"SYN_{i+1:06d}" for i in range(needed)]

    # Add light noise to continuous columns to avoid exact duplicates while preserving distributions.
    noise_cols = ["income", "loan_amount", "installment", "monthly_payment_est", "int_rate", "loan_to_income"]
    for col in noise_cols:
        if col in synth.columns:
            values = pd.to_numeric(synth[col], errors="coerce")
            std = values.std(skipna=True)
            if pd.notna(std) and std > 0:
                noise = rng.normal(0, std * 0.015, size=len(synth))
                synth[col] = (values + noise).clip(lower=0)

    synth = add_engineered_features(synth)
    return pd.concat([df, synth], ignore_index=True)


def main():
    missing = [p.name for p in [FINANCIAL_FILE, AUTO1_FILE, AUTO2_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    financial = standardize_financial_loan(FINANCIAL_FILE)
    auto1 = standardize_vehicle_loan(AUTO1_FILE, "automobile_loan_default_1", has_default=True)
    auto2 = standardize_vehicle_loan(AUTO2_FILE, "automobile_loan_default_2", has_default=False)

    combined = pd.concat([financial, auto1, auto2], ignore_index=True, sort=False)
    combined = add_engineered_features(combined)
    combined = create_synthetic_rows(combined, TARGET_ROWS)

    # Reorder Streamlit-friendly core columns first.
    core_cols = [
        "record_id", "source", "record_origin", "income", "loan_amount", "int_rate",
        "term_months", "installment", "monthly_payment_est", "loan_to_income",
        "borrower_segment", "risk_band", "default", "loan_status", "dti",
        "total_payment", "home_ownership", "employment_length", "purpose", "grade",
        "credit_bureau", "score_source_1", "score_source_2", "score_source_3",
        "client_income_type", "client_education", "client_marital_status", "client_gender",
        "car_owned", "bike_owned", "active_loan", "house_own", "child_count",
        "age_years", "employed_years",
    ]
    other_cols = [c for c in combined.columns if c not in core_cols]
    combined = combined[core_cols + other_cols]

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns):,}")
    print(combined[["source", "record_origin"]].value_counts().to_string())


if __name__ == "__main__":
    main()
