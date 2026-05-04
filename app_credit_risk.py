"""
Streamlit Dashboard: Automotive Finance Portfolio Risk

This Streamlit app is based on DSC_527_RS_Topic3_Risk_GenSynData_Part_2.ipynb.
It intentionally DOES NOT load model.pkl. The dashboard is exploratory and
storytelling-focused, not dependent on a saved machine-learning model.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Automotive Finance Portfolio Risk",
    page_icon="🚗",
    layout="wide",
)

COLOR_SEQUENCE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]

AUTO_FILE = "financial_loan.csv"
VEHICLE_FILE_1 = "automobile_loan_default_1.csv"
VEHICLE_FILE_2 = "automobile_loan_default_2.csv"
FINAL_FILE = "auto_finance_combined_over_500k.csv"
TARGET_ROWS = 500_001

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def coerce_numeric_if_present(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_term_months(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def derive_default_from_status(series: pd.Series) -> pd.Series:
    status = series.astype(str).str.strip().str.lower()
    default_statuses = {
        "charged off", "default", "late (31-120 days)", "late (16-30 days)",
        "does not meet the credit policy. status:charged off"
    }
    non_default_statuses = {
        "fully paid", "current", "in grace period",
        "does not meet the credit policy. status:fully paid"
    }
    result = np.where(status.isin(default_statuses), 1, np.where(status.isin(non_default_statuses), 0, np.nan))
    return pd.Series(result, index=series.index).astype(float)


@st.cache_data(show_spinner="Building combined dataset...")
def load_and_prepare_data() -> pd.DataFrame:
    """Load source CSV files, harmonize schemas, and expand above 500K rows."""

    if os.path.exists(FINAL_FILE):
        df = pd.read_csv(FINAL_FILE, low_memory=False)
        return prepare_visual_fields(df)

    missing = [f for f in [AUTO_FILE, VEHICLE_FILE_1, VEHICLE_FILE_2] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing) +
            ". Place them in the same folder as app_credit_risk.py."
        )

    df_auto = pd.read_csv(AUTO_FILE, low_memory=False)
    df_vehicle_1 = pd.read_csv(VEHICLE_FILE_1, low_memory=False)
    df_vehicle_2 = pd.read_csv(VEHICLE_FILE_2, low_memory=False)

    df_vehicle = pd.concat([df_vehicle_1, df_vehicle_2], ignore_index=True)

    df_auto = standardize_columns(df_auto)
    df_vehicle = standardize_columns(df_vehicle)

    # Source flags before harmonization
    df_auto["source"] = "financial_loan"
    df_vehicle["source"] = "vehicle_loan_default"
    df_auto["record_origin"] = "original"
    df_vehicle["record_origin"] = "original"

    # Map financial_loan.csv columns to unified schema
    auto_map = {
        "annual_income": "income",
        "loan_status": "loan_status",
        "int_rate": "int_rate",
        "installment": "installment",
        "loan_amount": "loan_amount",
        "term": "term",
        "purpose": "purpose",
        "grade": "grade",
        "home_ownership": "home_ownership",
        "dti": "dti",
        "total_acc": "total_acc",
        "total_payment": "total_payment",
    }

    # Map vehicle default columns to unified schema
    vehicle_map = {
        "client_income": "income",
        "credit_amount": "loan_amount",
        "loan_annuity": "installment",
        "default": "default",
        "client_income_type": "employment_type",
        "client_education": "education",
        "client_marital_status": "marital_status",
        "client_gender": "gender",
        "loan_contract_type": "loan_contract_type",
        "client_housing_type": "housing_type",
        "score_source_1": "score_source_1",
        "score_source_2": "score_source_2",
        "score_source_3": "score_source_3",
        "credit_bureau": "credit_bureau",
        "active_loan": "active_loan",
        "house_own": "house_own",
        "car_owned": "car_owned",
        "bike_owned": "bike_owned",
    }

    df_auto = df_auto.rename(columns={k: v for k, v in auto_map.items() if k in df_auto.columns})
    df_vehicle = df_vehicle.rename(columns={k: v for k, v in vehicle_map.items() if k in df_vehicle.columns})

    # Derive unified default flag
    if "loan_status" in df_auto.columns:
        df_auto["default"] = derive_default_from_status(df_auto["loan_status"])
    if "default" not in df_vehicle.columns:
        df_vehicle["default"] = np.nan

    # Clean loan term from text like " 60 months"
    if "term" in df_auto.columns:
        df_auto["term"] = clean_term_months(df_auto["term"])
    if "term" not in df_vehicle.columns:
        # Use a practical default when vehicle source lacks explicit term.
        df_vehicle["term"] = 60

    numeric_candidates = [
        "income", "loan_amount", "int_rate", "installment", "term", "default",
        "dti", "total_acc", "total_payment", "score_source_1", "score_source_2",
        "score_source_3", "credit_bureau", "active_loan", "house_own", "car_owned", "bike_owned"
    ]
    df_auto = coerce_numeric_if_present(df_auto, numeric_candidates)
    df_vehicle = coerce_numeric_if_present(df_vehicle, numeric_candidates)

    # Align schemas and concatenate
    all_cols = sorted(set(df_auto.columns).union(set(df_vehicle.columns)))
    df_auto = df_auto.reindex(columns=all_cols)
    df_vehicle = df_vehicle.reindex(columns=all_cols)
    df_combined = pd.concat([df_auto, df_vehicle], ignore_index=True)

    # Missing value handling
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_combined.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df_combined[col].isna().any():
            median_val = df_combined[col].median()
            df_combined[col] = df_combined[col].fillna(0 if pd.isna(median_val) else median_val)

    for col in categorical_cols:
        if df_combined[col].isna().any():
            mode_val = df_combined[col].mode(dropna=True)
            df_combined[col] = df_combined[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    # Engineered fields
    df_combined = add_engineered_fields(df_combined)

    # Synthetic expansion using bootstrap sampling with small numeric noise
    if len(df_combined) < TARGET_ROWS:
        needed = TARGET_ROWS - len(df_combined)
        synthetic = df_combined.sample(n=needed, replace=True, random_state=42).copy()
        synthetic["record_origin"] = "synthetic_bootstrap"

        rng = np.random.default_rng(42)
        noise_cols = ["income", "loan_amount", "installment", "int_rate"]
        for col in noise_cols:
            if col in synthetic.columns:
                noise = rng.normal(loc=1.0, scale=0.02, size=len(synthetic))
                synthetic[col] = (synthetic[col].astype(float) * noise).clip(lower=0)

        synthetic = add_engineered_fields(synthetic)
        df_final = pd.concat([df_combined, synthetic], ignore_index=True)
    else:
        df_final = df_combined.copy()

    # Save for reuse on local runs; on Streamlit Cloud this may not persist across deploys.
    try:
        df_final.to_csv(FINAL_FILE, index=False)
    except Exception:
        pass

    return prepare_visual_fields(df_final)


def add_engineered_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"loan_amount", "income"}.issubset(df.columns):
        df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
        df["loan_to_income"] = df["loan_to_income"].replace([np.inf, -np.inf], np.nan).fillna(0)
    if {"loan_amount", "term"}.issubset(df.columns):
        df["monthly_payment_est"] = df["loan_amount"] / df["term"].replace(0, np.nan)
        df["monthly_payment_est"] = df["monthly_payment_est"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def prepare_visual_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["income", "loan_amount", "int_rate", "installment", "term", "default", "loan_to_income", "monthly_payment_est"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "loan_to_income" not in df.columns:
        df = add_engineered_fields(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    if "income" in df.columns:
        median_income = df["income"].median()
        df["borrower_segment"] = np.where(df["income"] >= median_income, "High Income", "Low Income")
        df["income_capped"] = df["income"].clip(upper=df["income"].quantile(0.99))
    if "loan_amount" in df.columns:
        df["loan_amount_capped"] = df["loan_amount"].clip(upper=df["loan_amount"].quantile(0.99))
    if "loan_to_income" in df.columns:
        df["loan_to_income_capped"] = df["loan_to_income"].clip(upper=df["loan_to_income"].quantile(0.99))
        df["risk_band"] = pd.cut(
            df["loan_to_income"],
            bins=[-np.inf, 0.20, 0.40, np.inf],
            labels=["Low Burden", "Moderate Burden", "Elevated Burden"],
        )
    if "default" in df.columns:
        df["default"] = df["default"].fillna(0).round().clip(0, 1).astype(int)
        df["default_label"] = df["default"].map({0: "No Default", 1: "Default"})
    else:
        df["default"] = 0
        df["default_label"] = "No Default"
    if "source" in df.columns:
        df["source_label"] = df["source"].astype(str).str.replace("_", " ").str.title()
    else:
        df["source_label"] = "Combined Dataset"
    return df

# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
st.title("🚗 Automotive Finance Portfolio Risk Dashboard")
st.caption("Segmented reveal dashboard based on the Part 2 notebook. No model.pkl is required.")

try:
    df_final = load_and_prepare_data()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")
source_options = sorted(df_final["source_label"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect("Data Source", source_options, default=source_options)

segment_options = sorted(df_final["borrower_segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect("Borrower Segment", segment_options, default=segment_options)

risk_options = sorted(df_final["risk_band"].dropna().astype(str).unique().tolist())
selected_risks = st.sidebar.multiselect("Risk Band", risk_options, default=risk_options)

max_income_default = int(min(df_final["income_capped"].max(), 500000)) if "income_capped" in df_final.columns else 250000
income_max = st.sidebar.slider("Cap displayed income", 10000, max(max_income_default, 10000), max_income_default)

plot_df = df_final[
    df_final["source_label"].isin(selected_sources)
    & df_final["borrower_segment"].isin(selected_segments)
    & df_final["risk_band"].astype(str).isin(selected_risks)
    & (df_final["income_capped"] <= income_max)
].copy()

if plot_df.empty:
    st.warning("No records match the selected filters. Adjust the sidebar selections.")
    st.stop()

# KPI cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Records", f"{len(plot_df):,}")
kpi2.metric("Default Rate", f"{plot_df['default'].mean() * 100:.2f}%")
kpi3.metric("Median Loan", f"${plot_df['loan_amount'].median():,.0f}")
kpi4.metric("Median LTI", f"{plot_df['loan_to_income'].median():.3f}")

st.markdown("---")
st.header("Segmented Reveal Storyboard")
st.info(
    "The dashboard moves from portfolio distribution, to borrower burden, to segment comparison, "
    "to layered borrower profiles, and then relationship validation."
)

# Reveal 1
st.subheader("Reveal 1: Portfolio Shape — Loan Amount Distribution")
st.write("Start with the overall portfolio shape to understand where most loan balances are concentrated.")
fig1 = px.histogram(
    plot_df,
    x="loan_amount_capped",
    color="borrower_segment",
    marginal="box",
    nbins=60,
    opacity=0.75,
    color_discrete_sequence=COLOR_SEQUENCE,
    labels={"loan_amount_capped": "Loan Amount (99th Percentile Cap)", "borrower_segment": "Borrower Segment"},
)
fig1.add_vline(x=plot_df["loan_amount_capped"].median(), line_dash="dash", line_color="#333333", annotation_text="Median loan amount")
fig1.update_layout(template="plotly_white", legend_title_text="Borrower Segment")
st.plotly_chart(fig1, use_container_width=True)

# Reveal 2
st.subheader("Reveal 2: Risk Concentration — Loan-to-Income Ratio")
st.write("The box plot shifts from volume to borrower burden and makes tail risk visible.")
fig2 = px.box(
    plot_df,
    x="borrower_segment",
    y="loan_to_income_capped",
    color="borrower_segment",
    points="outliers",
    color_discrete_sequence=COLOR_SEQUENCE,
    labels={"loan_to_income_capped": "Loan-to-Income Ratio (99th Percentile Cap)", "borrower_segment": "Borrower Segment"},
)
fig2.add_hline(y=0.40, line_dash="dash", line_color="#D55E00", annotation_text="Elevated burden reference: 0.40")
fig2.update_layout(template="plotly_white", showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# Reveal 3
st.subheader("Reveal 3: Small Multiples — Default Rate by Risk Band")
st.write("Small multiples compare borrower segments without crowding the chart.")
small_mult = (
    plot_df.dropna(subset=["risk_band", "default"])
    .groupby(["borrower_segment", "risk_band"], observed=True)
    .agg(default_rate=("default", "mean"), record_count=("default", "size"), avg_lti=("loan_to_income", "mean"))
    .reset_index()
)
small_mult["default_rate_pct"] = small_mult["default_rate"] * 100
fig3 = px.bar(
    small_mult,
    x="risk_band",
    y="default_rate_pct",
    color="risk_band",
    facet_col="borrower_segment",
    text=small_mult["default_rate_pct"].round(2).astype(str) + "%",
    color_discrete_sequence=COLOR_SEQUENCE,
    labels={"risk_band": "Loan-to-Income Risk Band", "default_rate_pct": "Default Rate (%)"},
    hover_data=["record_count", "avg_lti"],
)
fig3.update_traces(textposition="outside")
fig3.update_layout(template="plotly_white", legend_title_text="Risk Band")
st.plotly_chart(fig3, use_container_width=True)

# Reveal 4
st.subheader("Reveal 4: Slope Graph — Average Burden by Source and Segment")
st.write("The slope graph shows directional change in financial burden across income segments and source datasets.")
slope_df = (
    plot_df.dropna(subset=["loan_to_income", "borrower_segment"])
    .groupby(["source_label", "borrower_segment"], observed=True)
    .agg(avg_lti=("loan_to_income", "mean"), records=("loan_to_income", "size"))
    .reset_index()
)
fig4 = px.line(
    slope_df,
    x="borrower_segment",
    y="avg_lti",
    color="source_label",
    markers=True,
    color_discrete_sequence=COLOR_SEQUENCE,
    labels={"borrower_segment": "Borrower Segment", "avg_lti": "Average Loan-to-Income Ratio", "source_label": "Data Source"},
    hover_data=["records"],
)
fig4.update_layout(template="plotly_white", legend_title_text="Data Source")
st.plotly_chart(fig4, use_container_width=True)

# Reveal 5
st.subheader("Reveal 5: Layered Borrower Profile")
st.write("This layered chart combines income, loan amount, risk band, and default flag while preserving interpretability.")
scatter_df = plot_df.sample(n=min(30000, len(plot_df)), random_state=42)
fig5 = px.scatter(
    scatter_df,
    x="income_capped",
    y="loan_amount_capped",
    color="risk_band",
    symbol="default_label",
    opacity=0.55,
    color_discrete_sequence=COLOR_SEQUENCE,
    labels={"income_capped": "Borrower Income (Capped)", "loan_amount_capped": "Loan Amount (Capped)", "risk_band": "Risk Band", "default_label": "Default Flag"},
    hover_data=["borrower_segment", "loan_to_income_capped", "source_label"],
)
x_line = np.linspace(0, scatter_df["income_capped"].max(), 100)
fig5.add_trace(go.Scatter(x=x_line, y=0.40 * x_line, mode="lines", name="0.40 LTI reference", line=dict(color="#333333", dash="dash")))
fig5.update_layout(template="plotly_white", legend_title_text="Risk Story")
st.plotly_chart(fig5, use_container_width=True)

# Reveal 6
st.subheader("Reveal 6: Relationship Validation — Correlation Heatmap")
st.write("The heatmap closes the story by validating which financial and risk variables move together.")
corr_cols = ["income", "loan_amount", "int_rate", "monthly_payment_est", "loan_to_income", "default"]
corr_cols = [c for c in corr_cols if c in plot_df.columns]
corr_matrix = plot_df[corr_cols].replace([np.inf, -np.inf], np.nan).dropna().corr(numeric_only=True).round(2)
fig6 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, labels=dict(color="Correlation"))
fig6.update_layout(template="plotly_white")
st.plotly_chart(fig6, use_container_width=True)

# Mixed media/storytelling element
st.markdown("---")
st.header("Mixed Media: Storyboard Support")
st.markdown(
    """
    **Narrative flow for screencast:**  
    1. Start with portfolio shape.  
    2. Move to borrower burden.  
    3. Compare risk bands across segments.  
    4. Show directional burden by source.  
    5. Layer borrower profile variables.  
    6. Validate relationships with correlation.  
    """
)

st.header("Portfolio Risk Pulse Check")
quiz_lti = st.slider("Select a sample loan-to-income ratio", 0.0, 1.0, 0.25, 0.01)
if quiz_lti >= 0.40:
    st.error("Elevated burden: this profile would require closer review.")
elif quiz_lti >= 0.20:
    st.warning("Moderate burden: monitor the profile with other borrower variables.")
else:
    st.success("Lower burden: based on LTI alone, this appears more manageable.")
