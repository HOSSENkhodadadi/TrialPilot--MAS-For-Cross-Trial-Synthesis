"""Extraction Agent — "The Librarian".

Crawls ClinicalTrials.gov for historical trials matching the target condition,
extracts structured records, and classifies outcomes as SUCCESS / FAILURE / UNKNOWN.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from trialpilot.config import get_settings
from trialpilot.schemas import PipelineState
from trialpilot.tools.clinical_trials_api import search_clinical_trials

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Extraction Agent ("The Librarian") in a clinical trial design system.

Your job:
1. Use the search_clinical_trials tool to find historical trials for the given condition.
2. Analyze each trial's status, outcomes, and eligibility criteria.
3. Classify each trial's outcome as SUCCESS, FAILURE, or UNKNOWN based on:
   - Completion status (COMPLETED with results → likely SUCCESS)
   - Termination or withdrawal → likely FAILURE
   - Ongoing or no results → UNKNOWN
4. Summarize the key inclusion/exclusion criteria patterns across successful trials.
5. Identify the most promising interventions and common eligibility criteria.

Return your analysis as a JSON object with the structure:
{
  "trials": [<list of enriched trial records with outcome_result field>],
  "common_inclusion_patterns": "<summary of patterns in successful trials>",
  "common_exclusion_patterns": "<summary of exclusion criteria patterns>",
  "promising_interventions": "<summary of interventions that correlate with success>",
  "synthesis_notes": "<overall synthesis of the landscape>"
}
"""


def extraction_node(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: run the Extraction Agent."""
    settings = get_settings()
    condition = state["condition"]
    research_goal = state.get("research_goal", "")

    logger.info("[Librarian] Searching trials for: %s", condition)

    # Call the ClinicalTrials.gov tool directly
    raw_trials = search_clinical_trials.invoke({"condition": condition, "max_results": settings.max_trials_to_fetch})

    if not raw_trials:
        logger.warning("[Librarian] No trials found for '%s'", condition)
        return {
            "extracted_trials": [],
            "agent_logs": [f"[Librarian] No trials found for '{condition}'."],
        }

    # Use LLM to classify outcomes and synthesize patterns
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    user_msg = (
        f"Condition: {condition}\n"
        f"Research Goal: {research_goal}\n\n"
        f"I found {len(raw_trials)} trials. Here are the records:\n\n"
        f"{json.dumps(raw_trials[:15], indent=2, default=str)}\n\n"
        "Please classify each trial's outcome and synthesize the landscape. "
        "Return your response as the specified JSON object."
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    # Parse LLM response
    content = response.content
    try:
        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        analysis = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        logger.warning("[Librarian] Could not parse LLM response as JSON, using raw trials")
        analysis = {"trials": raw_trials}

    # Merge LLM classifications back into trial records
    enriched_trials = analysis.get("trials", raw_trials)

    log_msg = (
        f"[Librarian] Extracted {len(enriched_trials)} trials for '{condition}'. "
        f"Synthesis: {analysis.get('synthesis_notes', 'N/A')[:200]}"
    )
    logger.info(log_msg)

    return {
        "extracted_trials": enriched_trials,
        "agent_logs": [log_msg],
    }
