"""
Fits a Cox proportional hazards model on cases_clean.csv to identify which
case factors speed up or delay NI Act 138 disposal.

Note: court_name is dropped as a covariate because each judge in this
dataset sits in exactly one court (judge_name and court_name are perfectly
collinear) - keeping both would make the fit unstable/non-identifiable.
judge_name is kept since it's the finer-grained variable and implicitly
captures court-level effects.
"""

import pandas as pd
from lifelines import CoxPHFitter

INPUT_CSV = "cases_clean.csv"
SIGNIFICANCE_LEVEL = 0.05


def main():
    df = pd.read_csv(INPUT_CSV)

    covariates = df[["duration_days", "event_observed", "has_stay_order", "num_hearings", "judge_name"]].copy()
    covariates["has_stay_order"] = covariates["has_stay_order"].astype(int)

    covariates = pd.get_dummies(covariates, columns=["judge_name"], drop_first=True, dtype=int)

    cph = CoxPHFitter()
    cph.fit(covariates, duration_col="duration_days", event_col="event_observed")

    cph.print_summary()

    ax = cph.plot()
    ax.figure.savefig("cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
    print("\nSaved cox_hazard_ratios.png")

    summary = cph.summary
    significant = summary[summary["p"] < SIGNIFICANCE_LEVEL]

    print(f"\nStatistically significant covariates (p < {SIGNIFICANCE_LEVEL}):")
    if significant.empty:
        print("  None")
    else:
        for covariate, row in significant.iterrows():
            hr = row["exp(coef)"]
            direction = "speeds up disposal (higher hazard)" if hr > 1 else "delays disposal (lower hazard)"
            print(f"  {covariate}: HR={hr:.3f}, p={row['p']:.4f} -> {direction}")


if __name__ == "__main__":
    main()
