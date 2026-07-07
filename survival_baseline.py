"""
Fits a Kaplan-Meier estimator to cases_clean.csv and reports how much of
the NI Act 138 case backlog is still pending at 6/12/24/36 months.
"""

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

INPUT_CSV = "cases_clean.csv"
MONTHS = [6, 12, 24, 36]
DAYS_PER_MONTH = 30.44


def main():
    df = pd.read_csv(INPUT_CSV)

    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_days"], event_observed=df["event_observed"])

    ax = kmf.plot_survival_function()
    ax.set_xlabel("Days since filing")
    ax.set_ylabel("Proportion of cases still pending")
    ax.set_title("NI Act 138 Case Survival Curve (Bangalore, 2020-2023)")
    plt.savefig("survival_curve.png", dpi=150, bbox_inches="tight")
    print("Saved survival_curve.png")

    day_marks = [m * DAYS_PER_MONTH for m in MONTHS]
    pending_pct = kmf.predict(day_marks) * 100

    print("\nEstimated % of cases still pending:")
    for month, pct in zip(MONTHS, pending_pct):
        print(f"  {month:>2} months: {pct:.1f}%")


if __name__ == "__main__":
    main()
