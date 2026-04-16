"""Patient Simulation Agent — "The Digital Twin".

Generates a diverse synthetic patient cohort and evaluates eligibility against
criteria extracted by the Librarian.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from trialpilot.config import get_settings
from trialpilot.schemas import PipelineState
from trialpilot.tools.patient_generator import generate_synthetic_patients

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Patient Simulation Agent ("The Digital Twin") in a clinical trial design system.

Your job:
1. Review the inclusion/exclusion criteria from the extracted trials.
2. Define concrete eligibility rules based on the criteria patterns.
3. Apply these rules to each synthetic patient to determine eligibility.
4. Report on the diversity and feasibility of the eligible sub-cohort.

You will receive:
- A list of extracted trials with their criteria
- A synthetic patient cohort

Return your analysis as a JSON object:
{
  "eligibility_rules": {
    "age_min": <int>,
    "age_max": <int>,
    "max_ecog": <int>,
    "min_egfr": <float or null>,
    "required_biomarkers": {},
    "excluded_comorbidities": [<list>],
    "other_rules": "<text>"
  },
  "patients": [
    {"patient_id": "...", "eligible": true/false, "reason": "..."},
    ...
  ],
  "cohort_summary": {
    "total": <int>,
    "eligible": <int>,
    "eligible_fraction": <float>,
    "demographic_breakdown": "<summary of demographics among eligible>",
    "diversity_concerns": "<any demographic imbalances noted>"
  }
}

Only return the first 30 patient eligibility results in the patients array to keep output manageable.
Focus on accurately determining eligibility and flagging diversity concerns.
"""


def simulation_node(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: run the Patient Simulation Agent."""
    settings = get_settings()
    condition = state["condition"]
    extracted_trials = state.get("extracted_trials", [])

    logger.info("[Digital Twin] Generating synthetic cohort for: %s", condition)

    # Generate synthetic patients
    cohort = generate_synthetic_patients.invoke({
        "condition": condition,
        "n": settings.synthetic_cohort_size,
        "seed": 42,
    })

    # Prepare trial criteria summary for the LLM
    criteria_summary = []
    for trial in extracted_trials[:10]:
        if isinstance(trial, dict):
            criteria_summary.append({
                "nct_id": trial.get("nct_id", ""),
                "title": trial.get("title", "")[:100],
                "inclusion_criteria": trial.get("inclusion_criteria", "")[:500],
                "exclusion_criteria": trial.get("exclusion_criteria", "")[:500],
                "outcome_result": trial.get("outcome_result", "UNKNOWN"),
            })

    # Use LLM to derive eligibility rules and apply them
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    # Send a sample of patients (not all 200) to keep prompt manageable
    patient_sample = cohort[:30]

    user_msg = (
        f"Condition: {condition}\n\n"
        f"Extracted trial criteria (focus on successful trials):\n"
        f"{json.dumps(criteria_summary, indent=2, default=str)}\n\n"
        f"Synthetic patient sample ({len(patient_sample)} of {len(cohort)} total):\n"
        f"{json.dumps(patient_sample, indent=2, default=str)}\n\n"
        "Please:\n"
        "1. Define concrete eligibility rules based on the trial criteria patterns.\n"
        "2. Evaluate each patient in the sample for eligibility.\n"
        "3. Extrapolate the eligible fraction to the full cohort.\n"
        "4. Flag any diversity concerns."
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    # Parse response
    content = response.content
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        analysis = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        logger.warning("[Digital Twin] Could not parse LLM JSON; using heuristic eligibility")
        analysis = {}

    # Apply eligibility rules to the full cohort
    rules = analysis.get("eligibility_rules", {})
    patient_decisions = {
        p.get("patient_id"): p.get("eligible", False)
        for p in analysis.get("patients", [])
    }

    # Apply rule-based eligibility to remaining patients
    for patient in cohort:
        pid = patient["patient_id"]
        if pid in patient_decisions:
            patient["eligible"] = patient_decisions[pid]
        else:
            # Heuristic application of rules
            eligible = True
            age_min = rules.get("age_min", 18)
            age_max = rules.get("age_max", 85)
            max_ecog = rules.get("max_ecog", 2)
            min_egfr = rules.get("min_egfr")
            excluded = rules.get("excluded_comorbidities", [])

            if not (age_min <= patient["age"] <= age_max):
                eligible = False
            if patient["ecog_score"] > max_ecog:
                eligible = False
            if min_egfr and patient["biomarkers"].get("eGFR", 100) < min_egfr:
                eligible = False
            if any(c in excluded for c in patient.get("comorbidities", [])):
                eligible = False

            patient["eligible"] = eligible

    eligible_count = sum(1 for p in cohort if p.get("eligible"))
    eligible_fraction = eligible_count / len(cohort) if cohort else 0

    cohort_summary = analysis.get("cohort_summary", {
        "total": len(cohort),
        "eligible": eligible_count,
        "eligible_fraction": round(eligible_fraction, 3),
    })

    log_msg = (
        f"[Digital Twin] Generated {len(cohort)} patients; "
        f"{eligible_count} eligible ({eligible_fraction:.1%}). "
        f"Diversity: {cohort_summary.get('diversity_concerns', 'N/A')}"
    )
    logger.info(log_msg)

    return {
        "synthetic_patients": cohort,
        "agent_logs": [log_msg],
    }
