"""Synthetic patient data generator.

Generates realistic synthetic patient cohorts for trial feasibility analysis.
Uses statistical distributions grounded in epidemiological ranges rather than
real patient data, so no PHI concerns.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from langchain_core.tools import tool


def _generate_cohort(
    condition: str,
    n: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate *n* synthetic patient records for a given condition."""
    rng = np.random.default_rng(seed)

    sexes = ["Male", "Female"]
    races = [
        "White", "Black or African American", "Asian",
        "Hispanic or Latino", "American Indian or Alaska Native", "Other",
    ]
    race_weights = [0.40, 0.18, 0.12, 0.20, 0.03, 0.07]

    # Condition-aware comorbidity pool
    comorbidity_pool = [
        "Hypertension", "Type 2 Diabetes", "COPD", "Chronic Kidney Disease",
        "Coronary Artery Disease", "Atrial Fibrillation", "Heart Failure",
        "Obesity", "Hypothyroidism", "Anemia",
    ]

    prior_treatment_pool = [
        "Chemotherapy", "Radiation Therapy", "Immunotherapy",
        "Targeted Therapy", "Surgery", "Hormone Therapy",
    ]

    patients: list[dict[str, Any]] = []
    for _ in range(n):
        age = int(rng.normal(loc=62, scale=12))
        age = max(18, min(90, age))

        n_comorbidities = rng.poisson(lam=1.5)
        comorbidities = rng.choice(
            comorbidity_pool,
            size=min(n_comorbidities, len(comorbidity_pool)),
            replace=False,
        ).tolist()

        n_prior = rng.poisson(lam=1.0)
        prior_treatments = rng.choice(
            prior_treatment_pool,
            size=min(n_prior, len(prior_treatment_pool)),
            replace=False,
        ).tolist()

        # Biomarkers (example: PD-L1 for oncology, eGFR for renal)
        biomarkers = {
            "PD-L1_TPS": round(float(rng.beta(2, 5) * 100), 1),
            "eGFR": round(float(rng.normal(75, 25)), 1),
            "Hemoglobin": round(float(rng.normal(12.5, 2.0)), 1),
            "ALC": round(float(rng.lognormal(0.5, 0.6)), 2),  # abs lymphocyte count
        }
        biomarkers["eGFR"] = max(15.0, biomarkers["eGFR"])

        # Simulated survival (months) — exponential with hazard influenced by age & comorbidities
        base_hazard = 0.05
        hazard = base_hazard * (1 + 0.01 * (age - 60)) * (1 + 0.15 * len(comorbidities))
        survival_months = round(float(rng.exponential(1 / hazard)), 1)

        patients.append({
            "patient_id": f"SYN-{uuid.uuid4().hex[:8].upper()}",
            "age": age,
            "sex": rng.choice(sexes),
            "race": rng.choice(races, p=race_weights),
            "primary_diagnosis": condition,
            "comorbidities": comorbidities,
            "biomarkers": biomarkers,
            "prior_treatments": prior_treatments,
            "ecog_score": int(rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.40, 0.30, 0.10, 0.05])),
            "eligible": None,
            "survival_months": survival_months,
        })

    return patients


@tool
def generate_synthetic_patients(
    condition: str, n: int = 200, seed: int | None = 42
) -> list[dict[str, Any]]:
    """Generate a synthetic patient cohort for trial feasibility testing.

    Args:
        condition: The primary diagnosis / medical condition.
        n: Number of patients to generate (default 200).
        seed: Random seed for reproducibility.

    Returns:
        List of synthetic patient dicts.
    """
    n = min(n, 1000)  # safety cap
    return _generate_cohort(condition, n, seed)
