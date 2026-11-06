# NI Act §138 Case Timeline Estimator

Survival analysis of how long Indian **Negotiable Instruments Act, Section 138** (cheque-bounce) cases take to resolve, with an interactive Streamlit estimator.

> **Data note:** this runs on **synthetic/mock data** (`generate_mock_data.py`) that mimics realistic case-duration distributions. It's a demonstration of the survival-analysis method, not a prediction of real court outcomes.

## What it does
- **Kaplan–Meier** estimator for the baseline "survival" curve — the probability a case is *still open* at 6 / 12 / 24 / 36 months
- **Cox Proportional Hazards** model for how case features (court type, amount, etc.) shift the hazard of resolution
- A **Streamlit app** that refits both models on load and gives a case-timeline estimate with risk-direction indicators

## Why survival analysis (not regression)
Many cases are still open when the data is collected — you know they've lasted *at least* N months, but not their final duration. That's **right-censoring**, and ordinary regression on "duration" would throw those cases away or bias the estimate. Kaplan–Meier and Cox models are built to use censored observations correctly.

## Run it
```bash
pip install -r requirements.txt
python generate_mock_data.py     # writes cases_raw.csv
python clean_data.py             # -> cases_clean.csv
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `generate_mock_data.py` | Synthetic case dataset |
| `clean_data.py` | Cleaning → `cases_clean.csv` |
| `survival_baseline.py` | Kaplan–Meier fit + curve |
| `cox_model.py` | Cox PH model + hazard ratios |
| `app.py` | Streamlit dashboard |

Built with [lifelines](https://lifelines.readthedocs.io/).
