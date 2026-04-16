"""Regulatory Critic Agent — "The Ethicist".

Reviews the proposed trial design against FDA/EMA guidelines using
Chain-of-Thought reasoning. Flags biases, safety concerns, and ethical issues.
Can trigger a feedback loop back to the Extraction Agent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from trialpilot.config import get_settings
from trialpilot.schemas import PipelineState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Regulatory Critic Agent ("The Ethicist") in a clinical trial design system.
You review proposed trial designs against FDA and EMA regulatory guidelines.

You MUST use Chain-of-Thought reasoning. Think step by step through each aspect:

## Review Checklist:
1. **Demographic Bias**: Do the inclusion/exclusion criteria accidentally exclude or
   under-represent specific demographics (age groups, sex, race/ethnicity)?
   - FDA Guidance: "Enhancing the Diversity of Clinical Trial Populations"
   - Check if age ranges exclude elderly (>75) without medical justification
   - Check if criteria disproportionately exclude minority populations

2. **Safety Concerns**: Are there adequate safety monitoring provisions?
   - Stopping rules for adverse events
   - Dose-limiting toxicity criteria
   - Vulnerable population protections

3. **Efficacy Design**: Is the statistical design sound?
   - Appropriate primary endpoint
   - Adequate sample size and power
   - Relevant comparator arm

4. **Ethical Considerations**:
   - Informed consent adequacy
   - Risk-benefit ratio
   - Equipoise between arms

5. **Regulatory Compliance**:
   - ICH-GCP alignment
   - Required safety reporting timelines
   - Data monitoring committee provisions

## Decision:
- If you find HIGH-severity issues, set "approved" to false and "needs_revision" to true.
- If only LOW/MEDIUM issues, set "approved" to true, "needs_revision" to false, but still list findings.

Return your review as a JSON object:
{
  "chain_of_thought": "<your step-by-step reasoning>",
  "findings": [
    {
      "category": "BIAS|SAFETY|EFFICACY|ETHICAL|DESIGN",
      "severity": "HIGH|MEDIUM|LOW",
      "description": "<what the issue is>",
      "recommendation": "<how to fix it>"
    }
  ],
  "approved": <true/false>,
  "needs_revision": <true/false>,
  "suggested_criteria_changes": "<specific changes to inclusion/exclusion criteria>",
  "overall_assessment": "<brief overall assessment>"
}
"""


def regulatory_node(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: run the Regulatory Critic Agent."""
    settings = get_settings()
    proposed_design = state.get("proposed_design", {})
    statistical_analysis = state.get("statistical_analysis", {})
    patients = state.get("synthetic_patients", [])
    iteration = state.get("iteration", 0)

    logger.info("[Ethicist] Reviewing proposed trial design (iteration %d)", iteration)

    # Compute demographic summary of eligible patients
    eligible = [p for p in patients if p.get("eligible")]
    demo_summary = _demographic_summary(eligible) if eligible else "No eligible patients."

    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    user_msg = (
        f"## Proposed Trial Design\n"
        f"{json.dumps(proposed_design, indent=2, default=str)}\n\n"
        f"## Statistical Analysis Summary\n"
        f"{json.dumps(statistical_analysis, indent=2, default=str)}\n\n"
        f"## Eligible Cohort Demographics\n"
        f"{demo_summary}\n\n"
        f"## Current Iteration: {iteration}\n"
        f"Max iterations allowed: {state.get('max_iterations', 2)}\n\n"
        "Please perform a thorough regulatory review with Chain-of-Thought reasoning."
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
        review = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        logger.warning("[Ethicist] Could not parse LLM JSON; defaulting to approved")
        review = {"approved": True, "needs_revision": False, "findings": [], "chain_of_thought": content}

    regulatory_review = {
        "approved": review.get("approved", True),
        "findings": review.get("findings", []),
        "chain_of_thought": review.get("chain_of_thought", ""),
        "suggested_criteria_changes": review.get("suggested_criteria_changes", ""),
        "overall_assessment": review.get("overall_assessment", ""),
    }

    needs_revision = review.get("needs_revision", False)
    # Only allow revision if we haven't exceeded max iterations
    max_iter = state.get("max_iterations", 2)
    if iteration >= max_iter:
        needs_revision = False

    high_count = sum(
        1 for f in review.get("findings", []) if f.get("severity") == "HIGH"
    )

    log_msg = (
        f"[Ethicist] Review complete: approved={regulatory_review['approved']}, "
        f"findings={len(review.get('findings', []))} (HIGH={high_count}), "
        f"needs_revision={needs_revision}"
    )
    logger.info(log_msg)

    # Enrich proposed design with regulatory summary
    updated_design = dict(proposed_design)
    updated_design["regulatory_summary"] = review.get("overall_assessment", "")

    return {
        "regulatory_review": regulatory_review,
        "proposed_design": updated_design,
        "needs_revision": needs_revision,
        "iteration": iteration + 1,
        "agent_logs": [log_msg],
    }


def _demographic_summary(patients: list[dict[str, Any]]) -> str:
    """Compute a textual demographic summary of a patient list."""
    if not patients:
        return "No patients."

    n = len(patients)
    ages = [p.get("age", 0) for p in patients]
    sexes: dict[str, int] = {}
    races: dict[str, int] = {}
    ecog_scores: dict[int, int] = {}

    for p in patients:
        sex = p.get("sex", "Unknown")
        sexes[sex] = sexes.get(sex, 0) + 1
        race = p.get("race", "Unknown")
        races[race] = races.get(race, 0) + 1
        ecog = p.get("ecog_score", -1)
        ecog_scores[ecog] = ecog_scores.get(ecog, 0) + 1

    lines = [
        f"Total eligible: {n}",
        f"Age: mean={sum(ages)/n:.1f}, range=[{min(ages)}-{max(ages)}]",
        f"Sex: {', '.join(f'{k}={v} ({v/n:.0%})' for k, v in sorted(sexes.items()))}",
        f"Race: {', '.join(f'{k}={v} ({v/n:.0%})' for k, v in sorted(races.items()))}",
        f"ECOG: {', '.join(f'{k}={v}' for k, v in sorted(ecog_scores.items()))}",
    ]
    return "\n".join(lines)
