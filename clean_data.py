"""
Cleans cases_raw.csv into cases_clean.csv, computing the duration and
censoring flag that lifelines needs for survival analysis.
"""

import pandas as pd
from datetime import date

INPUT_CSV = "cases_raw.csv"
OUTPUT_CSV = "cases_clean.csv"
TODAY = pd.Timestamp(date.today())


def main():
    df = pd.read_csv(INPUT_CSV)

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["disposal_date"] = pd.to_datetime(df["disposal_date"], errors="coerce")

    df = df.dropna(subset=["filing_date"])

    df["event_observed"] = df["disposal_date"].notna()
    df["duration_days"] = (
        df["disposal_date"].fillna(TODAY) - df["filing_date"]
    ).dt.days

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    print(df[["duration_days", "event_observed"]].describe(include="all"))


if __name__ == "__main__":
    main()
