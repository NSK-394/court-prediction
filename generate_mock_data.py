"""
Generates a synthetic dataset of NI Act Section 138 (cheque bounce) cases
standing in for the eCourts / court-api.kleopatra.io API, which currently
requires a paid API key and has no published endpoint documentation.

Mimics the shape of real district court data: a mix of disposed and still-
pending cases, varied hearing counts, occasional stay orders, and a
log-normal spread of disposal times consistent with known NI Act backlogs.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(42)

N_CASES = 750
COURTS = [
    ("Bangalore Rural District Court", ["Judge K. Ramesh", "Judge S. Nagaraj", "Judge P. Lakshmi"]),
    ("Bangalore City Civil Court", ["Judge A. Prakash", "Judge M. Shalini", "Judge R. Vasanth"]),
]
CASE_TYPE = "NI Act 138"
FILING_START = date(2020, 1, 1)
FILING_END = date(2023, 12, 31)
TODAY = date(2026, 7, 3)

PENDING_FRACTION = 0.35


def random_filing_date():
    span_days = (FILING_END - FILING_START).days
    return FILING_START + timedelta(days=int(np.random.randint(0, span_days + 1)))


def random_court_judge():
    court_name, judges = COURTS[np.random.randint(0, len(COURTS))]
    judge_name = judges[np.random.randint(0, len(judges))]
    return court_name, judge_name


def main():
    rows = []
    for i in range(1, N_CASES + 1):
        filing_date = random_filing_date()
        court_name, judge_name = random_court_judge()
        has_stay_order = bool(np.random.rand() < 0.15)
        is_pending = np.random.rand() < PENDING_FRACTION

        max_possible_duration = (TODAY - filing_date).days

        if is_pending or max_possible_duration <= 0:
            disposal_date = None
            current_status = "Stayed" if has_stay_order else "Pending"
            duration_so_far = max(max_possible_duration, 1)
            num_hearings = int(np.clip(np.random.poisson(duration_so_far / 60), 1, None))
        else:
            # log-normal duration centered around 12-18 months, capped at what's possible
            duration = int(np.random.lognormal(mean=6.0, sigma=0.5))
            duration = int(np.clip(duration, 30, max_possible_duration))
            disposal_date = filing_date + timedelta(days=duration)
            current_status = "Disposed"
            num_hearings = int(np.clip(np.random.poisson(duration / 45), 1, None))

        case_number = f"CC/{i:05d}/{filing_date.year}"

        rows.append({
            "case_number": case_number,
            "case_type": CASE_TYPE,
            "filing_date": filing_date.isoformat(),
            "disposal_date": disposal_date.isoformat() if disposal_date else None,
            "current_status": current_status,
            "court_name": court_name,
            "judge_name": judge_name,
            "num_hearings": num_hearings,
            "has_stay_order": has_stay_order,
        })

    df = pd.DataFrame(rows)
    df.to_csv("cases_raw.csv", index=False)
    print(f"Wrote {len(df)} rows to cases_raw.csv")
    print(df["current_status"].value_counts())


if __name__ == "__main__":
    main()
