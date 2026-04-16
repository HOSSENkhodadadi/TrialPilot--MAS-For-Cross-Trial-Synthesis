"""Biostatistical Reasoning Agent — "The Analyst".

Runs survival analysis on the synthetic cohort, estimates Probability of Success,
and recommends sample sizes for the proposed trial.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from trialpilot.config import get_settings
from trialpilot.schemas import PipelineState
from trialpilot.tools.statistics import compute_probability_of_success, run_survival_analysis

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Biostatistical Reasoning Agent ("The Analyst") in a clinical trial design system.

Your job:
1. Review the survival analysis results from the synthetic cohort.
2. Review the extracted trial landscape to determine historical control values.
3. Interpret the Probability of Success (PoS) estimation.
4. Propose a concrete trial design with:
   - Recommended sample size
   - Primary endpoint specification
   - Power analysis justification
   - Risk assessment

Return your analysis as a JSON object:
{
  "statistical_interpretation": "<detailed interpretation of survival & PoS>",
  "proposed_design": {
    "phase": "<recommended phase>",
    "primary_endpoint": "<e.g., Overall Survival, PFS>",
    "target_enrollment": <int>,
    "control_arm": "<description>",
    "treatment_arm": "<description>",
    "inclusion_criteria": ["<list of key criteria>"],
    "exclusion_criteria": ["<list of key exclusions>"],
    "estimated_pos": <float 0-1>,
    "estimated_duration_months": <int>
  },
  "risk_factors": ["<list of key risks>"],
  "recommendations": "<additional recommendations>"
}
"""


def biostatistics_node(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: run the Biostatistical Reasoning Agent."""
    settings = get_settings()
    condition = state["condition"]
    patients = state.get("synthetic_patients", [])
    extracted_trials = state.get("extracted_trials", [])

    logger.info("[Analyst] Running survival analysis on %d patients", len(patients))

    # Run survival analysis tool
    survival_results = run_survival_analysis.invoke({"patients": patients})

    # Compute PoS
    eligible_count = survival_results.get("eligible_count", 0)
    total_count = survival_results.get("total_count", len(patients))
    eligible_fraction = eligible_count / total_count if total_count > 0 else 0

    median_survival = survival_results.get("median_survival_months")
    if median_survival and median_survival > 0:
        pos_results = compute_probability_of_success.invoke({
            "eligible_fraction": eligible_fraction,
            "median_survival_months": median_survival,
            "historical_control_median": 10.0,
        })
    else:
        pos_results = {
            "probability_of_success": 0.0,
            "notes": "Unable to compute PoS — no valid survival data.",
        }

    # Feed results to LLM for interpretation and design proposal
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    # Summarize trial landscape for context
    trial_summary = []
    for t in extracted_trials[:10]:
        if isinstance(t, dict):
            trial_summary.append({
                "nct_id": t.get("nct_id", ""),
                "intervention": t.get("intervention", ""),
                "outcome_result": t.get("outcome_result", "UNKNOWN"),
                "enrollment": t.get("enrollment"),
            })

    user_msg = (
        f"Condition: {condition}\n\n"
        f"Survival Analysis Results:\n{json.dumps(survival_results, indent=2, default=str)}\n\n"
        f"Probability of Success Estimation:\n{json.dumps(pos_results, indent=2, default=str)}\n\n"
        f"Trial Landscape Summary:\n{json.dumps(trial_summary, indent=2, default=str)}\n\n"
        "Please interpret these results and propose a concrete trial design."
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    content = response.content
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        analysis = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        logger.warning("[Analyst] Could not parse LLM JSON response")
        analysis = {}

    # Build statistical analysis output
    statistical_analysis = {
        "survival_results": survival_results,
        "pos_results": pos_results,
        "llm_interpretation": analysis.get("statistical_interpretation", ""),
        "risk_factors": analysis.get("risk_factors", []),
        "recommendations": analysis.get("recommendations", ""),
    }

    # Build proposed design
    proposed = analysis.get("proposed_design", {})
    proposed_design = {
        "condition": condition,
        "phase": proposed.get("phase", "Phase III"),
        "intervention": proposed.get("treatment_arm", ""),
        "inclusion_criteria": proposed.get("inclusion_criteria", []),
        "exclusion_criteria": proposed.get("exclusion_criteria", []),
        "primary_endpoint": proposed.get("primary_endpoint", "Overall Survival"),
        "target_enrollment": proposed.get("target_enrollment"),
        "estimated_pos": pos_results.get("probability_of_success"),
        "statistical_summary": survival_results.get("kaplan_meier_summary", ""),
    }

    log_msg = (
        f"[Analyst] Survival: median={survival_results.get('median_survival_months')} mo, "
        f"PoS={pos_results.get('probability_of_success', 'N/A')}, "
        f"recommended N={pos_results.get('recommended_sample_size', 'N/A')}"
    )
    logger.info(log_msg)

    return {
        "statistical_analysis": statistical_analysis,
        "proposed_design": proposed_design,
        "agent_logs": [log_msg],
    }
