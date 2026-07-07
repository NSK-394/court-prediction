"""
Streamlit dashboard for the NI Act 138 case survival analysis.

Refits KaplanMeierFitter and CoxPHFitter on load (cached via
st.cache_resource so refits only happen once per session) rather than
pickling, since the dataset is small enough that refitting is instant.

Design system: deep ink-navy surface, one brass accent for anchor figures,
oxblood/teal signal colors borrowed from courtroom stamp conventions
(overdue vs. cleared) for the risk-direction indicators. Source Serif 4
for headings, IBM Plex Sans for UI text, IBM Plex Mono for every number
on the page (KPIs, axis ticks, the risk ruler) so figures read as
measured output rather than decoration.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter

DATA_CSV = "cases_clean.csv"
MONTHS = [6, 12, 24, 36]
DAYS_PER_MONTH = 30.44

INK = "#12161D"
PANEL = "#1A212B"
HAIRLINE = "#2B3341"
TEXT = "#E7EAEE"
MUTED = "#8891A0"
BRASS = "#B98A44"
OXBLOOD = "#8C3B3B"
TEAL = "#4C7A6B"

st.set_page_config(page_title="NI Act 138 Case Timeline Estimator", layout="wide")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {INK};
    color: {TEXT};
    font-family: 'IBM Plex Sans', sans-serif;
}}
[data-testid="stHeader"] {{ background-color: transparent; }}
[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {HAIRLINE};
}}
h1, h2, h3 {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: {TEXT} !important;
    font-weight: 600 !important;
}}
.brand-title {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.1rem;
}}
.brand-subtitle {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {MUTED};
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    margin-bottom: 1.2rem;
}}
.kpi-band {{
    display: flex;
    border-top: 1px solid {HAIRLINE};
    border-bottom: 1px solid {HAIRLINE};
    padding: 1rem 0;
    margin-bottom: 1.6rem;
    gap: 2.5rem;
}}
.kpi-item {{ display: flex; flex-direction: column; }}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: {BRASS};
    line-height: 1.1;
}}
.kpi-label {{
    font-family: 'IBM Plex Sans', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.68rem;
    color: {MUTED};
    margin-top: 0.25rem;
}}
.section-label {{
    font-family: 'IBM Plex Sans', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.75rem;
    color: {MUTED};
    margin-bottom: 0.4rem;
}}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {MUTED};
}}
div.stButton > button {{
    background-color: {BRASS};
    color: {INK};
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    width: 100%;
}}
div.stButton > button:hover {{
    background-color: #cf9b52;
    color: {INK};
}}
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {TEXT} !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {MUTED} !important;
    font-size: 0.72rem !important;
}}
.ruler-caption {{
    font-family: 'IBM Plex Mono', monospace;
    color: {MUTED};
    font-size: 0.85rem;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


def render_survival_chart(kmf: KaplanMeierFitter):
    plt.rcParams["font.family"] = "monospace"

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)

    timeline = kmf.survival_function_.index.values
    survival = kmf.survival_function_.iloc[:, 0].values

    ax.step(timeline, survival, where="post", color=BRASS, linewidth=1.8)
    ax.fill_between(timeline, survival, step="post", color=BRASS, alpha=0.12)

    median_days = kmf.median_survival_time_
    if median_days != float("inf"):
        ax.axvline(median_days, color=MUTED, linestyle=(0, (1, 3)), linewidth=1)
        ax.axhline(0.5, color=MUTED, linestyle=(0, (1, 3)), linewidth=1)
        ax.text(
            median_days + 15, 0.52, f"median {median_days / DAYS_PER_MONTH:.1f}mo",
            color=MUTED, fontsize=9, fontfamily="monospace",
        )

    ax.set_xlabel("Days since filing", color=MUTED, fontsize=10)
    ax.set_ylabel("Proportion still pending", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(HAIRLINE)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, timeline.max())

    fig.tight_layout()
    return fig


def risk_ruler_svg(baseline_months, user_months):
    axis_x0, axis_x1 = 60, 700
    axis_y, top_y = 90, 42

    known = [v for v in (baseline_months, user_months) if v is not None]
    max_scale = max(40, math.ceil((max(known) * 1.25) / 5) * 5) if known else 40

    def x_for(month):
        return axis_x0 + (min(month, max_scale) / max_scale) * (axis_x1 - axis_x0)

    ticks = list(range(0, max_scale + 1, 6))
    tick_svg = "".join(
        f'<line x1="{x_for(t)}" y1="{axis_y}" x2="{x_for(t)}" y2="{axis_y + 6}" '
        f'stroke="{HAIRLINE}" stroke-width="1"/>'
        f'<text x="{x_for(t)}" y="{axis_y + 22}" fill="{MUTED}" font-family="IBM Plex Mono" '
        f'font-size="11" text-anchor="middle">{t}mo</text>'
        for t in ticks
    )
    axis_svg = (
        f'<line x1="{axis_x0}" y1="{axis_y}" x2="{axis_x1}" y2="{axis_y}" '
        f'stroke="{HAIRLINE}" stroke-width="1.5"/>'
    )

    baseline_x = x_for(baseline_months) if baseline_months is not None else None
    markers, bracket_svg = "", ""

    if baseline_x is not None:
        markers += (
            f'<circle cx="{baseline_x}" cy="{axis_y}" r="5" fill="{BRASS}"/>'
            f'<text x="{baseline_x}" y="{axis_y - 14}" fill="{BRASS}" font-family="IBM Plex Sans" '
            f'font-size="12" text-anchor="middle">Baseline</text>'
            f'<text x="{baseline_x}" y="{axis_y - 30}" fill="{TEXT}" font-family="IBM Plex Mono" '
            f'font-size="13" text-anchor="middle">{baseline_months:.1f}mo</text>'
        )

    if user_months is None:
        direction_color = OXBLOOD
        user_x = axis_x1 + 15
        delta_label = "beyond observed data"
        markers += (
            f'<line x1="{axis_x1}" y1="{axis_y}" x2="{user_x}" y2="{axis_y}" '
            f'stroke="{direction_color}" stroke-width="2" stroke-dasharray="4,3"/>'
        )
    else:
        user_x = x_for(user_months)
        direction_color = OXBLOOD if baseline_months is not None and user_months >= baseline_months else TEAL
        markers += (
            f'<circle cx="{user_x}" cy="{axis_y}" r="5" fill="{direction_color}"/>'
            f'<text x="{user_x}" y="{axis_y - 14}" fill="{direction_color}" font-family="IBM Plex Sans" '
            f'font-size="12" text-anchor="middle">Your Profile</text>'
            f'<text x="{user_x}" y="{axis_y - 30}" fill="{TEXT}" font-family="IBM Plex Mono" '
            f'font-size="13" text-anchor="middle">{user_months:.1f}mo</text>'
        )
        if baseline_months is not None:
            delta = user_months - baseline_months
            pct = (delta / baseline_months) * 100 if baseline_months else 0
            sign = "+" if delta >= 0 else ""
            word = "longer" if delta >= 0 else "shorter"
            delta_label = f"{sign}{delta:.1f}mo ({sign}{pct:.0f}%) {word}"
        else:
            delta_label = ""

    if baseline_x is not None and delta_label:
        bx0, bx1 = sorted([baseline_x, min(user_x, axis_x1 + 15)])
        bracket_svg = (
            f'<path d="M {bx0} {top_y + 10} L {bx0} {top_y} L {bx1} {top_y} L {bx1} {top_y + 10}" '
            f'stroke="{direction_color}" stroke-width="1.5" fill="none"/>'
            f'<text x="{(bx0 + bx1) / 2}" y="{top_y - 6}" fill="{direction_color}" '
            f'font-family="IBM Plex Mono" font-size="13" text-anchor="middle" font-weight="600">'
            f'{delta_label}</text>'
        )

    return f'<svg viewBox="0 0 760 140" style="width:100%; height:auto;">{bracket_svg}{axis_svg}{tick_svg}{markers}</svg>'


df = load_data()
kmf = fit_km(df)
cph = fit_cox(df)

court_options = sorted(df["court_name"].unique())
baseline_court = df["court_name"].mode()[0]
median_hearings = int(df["num_hearings"].median())

with st.sidebar:
    st.markdown('<div class="section-label">Case Profile</div>', unsafe_allow_html=True)
    court_name = st.selectbox("Court", court_options)
    has_stay_order = st.checkbox("Has stay order")
    num_hearings = st.number_input(
        "Expected number of hearings", min_value=0, max_value=100, value=median_hearings
    )
    submitted = st.button("Estimate timeline")

st.markdown('<div class="brand-title">NI Act 138 — Case Timeline Risk Estimator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-subtitle">Bangalore Rural District Court &middot; Bangalore City Civil Court</div>',
    unsafe_allow_html=True,
)

day_marks = [m * DAYS_PER_MONTH for m in MONTHS]
pending_pct = kmf.predict(day_marks) * 100
overall_median_months = (
    kmf.median_survival_time_ / DAYS_PER_MONTH if kmf.median_survival_time_ != float("inf") else None
)

kpi_html = '<div class="kpi-band">'
kpi_html += (
    f'<div class="kpi-item"><div class="kpi-value">'
    f'{f"{overall_median_months:.1f}mo" if overall_median_months else "n/a"}</div>'
    f'<div class="kpi-label">Median time to disposal</div></div>'
)
for month, pct in zip(MONTHS, pending_pct):
    kpi_html += (
        f'<div class="kpi-item"><div class="kpi-value">{pct:.1f}%</div>'
        f'<div class="kpi-label">Pending @ {month}mo</div></div>'
    )
kpi_html += (
    f'<div class="kpi-item"><div class="kpi-value">{len(df)}</div>'
    f'<div class="kpi-label">Cases in sample</div></div>'
)
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

st.markdown('<div class="section-label">Survival Curve — All Cases</div>', unsafe_allow_html=True)
st.pyplot(render_survival_chart(kmf))

if submitted:
    st.markdown('<div class="section-label">Cox-Adjusted Timeline Comparison</div>', unsafe_allow_html=True)

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

    st.markdown(risk_ruler_svg(baseline_result, user_result), unsafe_allow_html=True)
    st.markdown(
        '<div class="ruler-caption">Baseline = typical profile (no stay order, median hearing count, '
        'most common court). Your Profile = selected inputs.</div>',
        unsafe_allow_html=True,
    )

st.markdown(f'<div style="border-top:1px solid {HAIRLINE}; margin-top:2rem; padding-top:1rem;"></div>', unsafe_allow_html=True)
st.caption(
    "Trained on synthetic/sample data standing in for real eCourts data "
    "(no live court-api.kleopatra.io feed yet). num_hearings reflects "
    "associated procedural activity, not a proven causal driver of delay."
)
