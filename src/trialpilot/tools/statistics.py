"""Statistical analysis tools for trial feasibility.

Provides Kaplan-Meier survival analysis and eligibility screening using
lifelines and scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.tools import tool
from lifelines import KaplanMeierFitter
from scipy import stats


@tool
def run_survival_analysis(patients: list[dict[str, Any]]) -> dict[str, Any]:
    """Run Kaplan-Meier survival analysis on a patient cohort.

    Args:
        patients: List of patient dicts, each must have 'survival_months' and 'eligible' keys.

    Returns:
        Dict with median survival, confidence intervals, and summary text.
    """
    eligible = [p for p in patients if p.get("eligible") is True]
    if len(eligible) < 5:
        return {
            "median_survival_months": None,
            "survival_ci_lower": None,
            "survival_ci_upper": None,
            "eligible_count": len(eligible),
            "total_count": len(patients),
            "kaplan_meier_summary": "Insufficient eligible patients for survival analysis.",
        }

    durations = np.array([p["survival_months"] for p in eligible])
    # Assume all events observed (no censoring in synthetic data)
    event_observed = np.ones(len(durations), dtype=bool)

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=event_observed, label="Eligible Cohort")

    median = kmf.median_survival_time_
    ci = kmf.confidence_interval_survival_function_

    # Survival at 12 months
    try:
        surv_12 = float(kmf.predict(12.0))
    except Exception:
        surv_12 = None

    summary_lines = [
        f"Kaplan-Meier Analysis (n={len(eligible)} eligible of {len(patients)} total)",
        f"  Median survival: {median:.1f} months",
        f"  12-month survival rate: {surv_12:.1%}" if surv_12 is not None else "",
        f"  Mean survival: {durations.mean():.1f} months (SD={durations.std():.1f})",
    ]

    return {
        "median_survival_months": round(float(median), 2),
        "survival_ci_lower": round(float(np.percentile(durations, 2.5)), 2),
        "survival_ci_upper": round(float(np.percentile(durations, 97.5)), 2),
        "eligible_count": len(eligible),
        "total_count": len(patients),
        "survival_at_12mo": round(surv_12, 4) if surv_12 is not None else None,
        "mean_survival": round(float(durations.mean()), 2),
        "std_survival": round(float(durations.std()), 2),
        "kaplan_meier_summary": "\n".join(line for line in summary_lines if line),
    }


@tool
def compute_probability_of_success(
    eligible_fraction: float,
    median_survival_months: float,
    historical_control_median: float = 10.0,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Estimate trial Probability of Success (PoS).

    Uses a simplified frequentist model comparing expected survival improvement
    over historical control, combined with enrollment feasibility.

    Args:
        eligible_fraction: Fraction of screening cohort expected to be eligible (0-1).
        median_survival_months: Projected median survival for eligible arm.
        historical_control_median: Historical comparator median survival (months).
        sample_size: Planned sample size (if None, will recommend one).

    Returns:
        Dict with PoS estimate, recommended sample size, and reasoning.
    """
    # Hazard ratio estimate
    if median_survival_months <= 0:
        return {"probability_of_success": 0.0, "notes": "Invalid survival estimate."}

    hr = historical_control_median / median_survival_months  # HR < 1 = improvement
    hr = min(hr, 2.0)  # cap

    # Log-rank power approximation (Schoenfeld formula)
    alpha = 0.05
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    desired_power = 0.80
    z_beta = stats.norm.ppf(desired_power)

    if hr >= 1.0:
        # No improvement over control
        pos = max(0.05, 0.5 * eligible_fraction * 0.3)
        recommended_n = None
        notes = "Projected survival does not improve over historical control."
    else:
        log_hr = np.log(hr)
        # Required events
        d = ((z_alpha + z_beta) ** 2) / (log_hr ** 2)
        d = int(np.ceil(d))
        recommended_n = int(np.ceil(d / 0.7))  # assume ~70% event rate

        # PoS = P(observing the effect given the enrollment feasibility)
        enrollment_factor = min(1.0, eligible_fraction / 0.15)  # 15% is typical screen rate
        effect_confidence = 1 - stats.norm.cdf(0, loc=-log_hr, scale=0.3)
        pos = round(float(enrollment_factor * effect_confidence * 0.85), 3)  # 0.85 = trial execution discount
        notes = (
            f"HR={hr:.2f}, required events={d}, recommended N={recommended_n}, "
            f"enrollment factor={enrollment_factor:.2f}"
        )

    if sample_size is not None:
        recommended_n = sample_size

    return {
        "probability_of_success": pos,
        "hazard_ratio": round(float(hr), 3),
        "recommended_sample_size": recommended_n,
        "notes": notes,
    }
