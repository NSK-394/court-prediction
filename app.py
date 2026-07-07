"""
Streamlit dashboard for the NI Act 138 case survival analysis.

Refits KaplanMeierFitter and CoxPHFitter on load (cached via
st.cache_resource so refits only happen once per session) rather than
pickling, since the dataset is small enough that refitting is instant.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter

DATA_CSV = "cases_clean.csv"
MONTHS = [6, 12, 24, 36]
DAYS_PER_MONTH = 30.44

st.set_page_config(page_title="NI Act 138 Case Timeline Estimator", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_CSV)


@st.cache_resource
def fit_km(_df: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(_df["duration_days"], event_observed=_df["event_observed"])
    return kmf


@st.cache_resource
def fit_cox(_df: pd.DataFrame) -> CoxPHFitter:
    covariates = _df[["duration_days", "event_observed", "has_stay_order", "num_hearings", "court_name"]].copy()
    covariates["has_stay_order"] = covariates["has_stay_order"].astype(int)
    covariates = pd.get_dummies(covariates, columns=["court_name"], drop_first=True, dtype=int)

    cph = CoxPHFitter()
    cph.fit(covariates, duration_col="duration_days", event_col="event_observed")
    return cph


def build_covariate_row(court_name, has_stay_order, num_hearings, template_columns):
    row = pd.DataFrame([{
        "has_stay_order": int(has_stay_order),
        "num_hearings": num_hearings,
        "court_name": court_name,
    }])
    row = pd.get_dummies(row, columns=["court_name"], dtype=int)
    return row.reindex(columns=template_columns, fill_value=0)


def median_months(cph: CoxPHFitter, covariate_row: pd.DataFrame):
    median_days = cph.predict_median(covariate_row)
    median_days = float(median_days.iloc[0]) if hasattr(median_days, "iloc") else float(median_days)
    if median_days == float("inf"):
        return None
    return median_days / DAYS_PER_MONTH


df = load_data()
kmf = fit_km(df)
cph = fit_cox(df)

court_options = sorted(df["court_name"].unique())
baseline_court = df["court_name"].mode()[0]
median_hearings = int(df["num_hearings"].median())

st.title("NI Act 138 Case Timeline Estimator")
st.caption("Bangalore Rural District Court & Bangalore City Civil Court")

left, right = st.columns([1, 2])

with left:
    st.subheader("Case profile")
    court_name = st.selectbox("Court", court_options)
    has_stay_order = st.checkbox("Has stay order")
    num_hearings = st.number_input(
        "Expected number of hearings", min_value=0, max_value=100, value=median_hearings
    )
    submitted = st.button("Estimate timeline", type="primary")

with right:
    st.subheader("Kaplan-Meier baseline (all cases)")

    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Days since filing")
    ax.set_ylabel("Proportion of cases still pending")
    st.pyplot(fig)

    day_marks = [m * DAYS_PER_MONTH for m in MONTHS]
    pending_pct = kmf.predict(day_marks) * 100

    metric_cols = st.columns(len(MONTHS))
    for col, month, pct in zip(metric_cols, MONTHS, pending_pct):
        col.metric(f"Pending @ {month}mo", f"{pct:.1f}%")

    if submitted:
        st.subheader("Cox-adjusted expected timeline")

        template_columns = cph.params_.index

        baseline_row = build_covariate_row(
            baseline_court, has_stay_order=False, num_hearings=median_hearings,
            template_columns=template_columns,
        )
        user_row = build_covariate_row(
            court_name, has_stay_order=has_stay_order, num_hearings=num_hearings,
            template_columns=template_columns,
        )

        baseline_result = median_months(cph, baseline_row)
        user_result = median_months(cph, user_row)

        col_a, col_b = st.columns(2)
        col_a.metric(
            "Baseline case (typical profile)",
            f"{baseline_result:.1f} months" if baseline_result else "not reached in data",
        )

        if user_result is not None and baseline_result:
            pct_diff = (user_result - baseline_result) / baseline_result * 100
            direction = "longer" if pct_diff > 0 else "shorter"
            col_b.metric(
                "Your case profile",
                f"{user_result:.1f} months",
                f"{pct_diff:+.0f}% ({direction} than baseline)",
                delta_color="inverse",
            )
        else:
            col_b.metric("Your case profile", "not reached in data")

st.divider()
st.caption(
    "Trained on synthetic/sample data standing in for real eCourts data "
    "(no live court-api.kleopatra.io feed yet). num_hearings reflects "
    "associated procedural activity, not a proven causal driver of delay."
)
