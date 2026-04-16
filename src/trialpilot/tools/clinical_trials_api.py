"""ClinicalTrials.gov v2 API client.

Docs: https://clinicaltrials.gov/data-api/api
No API key required; respect rate limits (≤3 req/sec).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.tools import tool

from trialpilot.config import get_settings

logger = logging.getLogger(__name__)

_FIELDS = (
    "NCTId,BriefTitle,Condition,Phase,OverallStatus,EnrollmentCount,"
    "StartDate,CompletionDate,InterventionName,EligibilityCriteria,"
    "PrimaryOutcomeMeasure,BriefSummary"
)


def _parse_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten a ClinicalTrials.gov v2 study object into a simple dict."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    eligibility = proto.get("eligibilityModule", {})
    outcomes = proto.get("outcomesModule", {})
    desc = proto.get("descriptionModule", {})
    arms = proto.get("armsInterventionsModule", {})

    interventions = arms.get("interventions", [])
    intervention_names = [i.get("name", "") for i in interventions]

    primary_outcomes = outcomes.get("primaryOutcomes", [])
    primary_measures = [o.get("measure", "") for o in primary_outcomes]

    phases = design.get("phases", [])
    enrollment_info = design.get("enrollmentInfo", {})

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", ""),
        "condition": ", ".join(
            proto.get("conditionsModule", {}).get("conditions", [])
        ),
        "phase": ", ".join(phases) if phases else "",
        "status": status_mod.get("overallStatus", ""),
        "enrollment": enrollment_info.get("count"),
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
        "intervention": "; ".join(intervention_names),
        "inclusion_criteria": eligibility.get("eligibilityCriteria", ""),
        "exclusion_criteria": "",  # embedded in eligibilityCriteria text
        "primary_outcome": "; ".join(primary_measures),
        "summary": desc.get("briefSummary", ""),
    }


@tool
def search_clinical_trials(condition: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Search ClinicalTrials.gov for trials matching a given condition.

    Args:
        condition: Medical condition or search query (e.g. "Immunotherapy Stage IV Lung Cancer").
        max_results: Maximum number of trials to return (capped at 50).

    Returns:
        List of simplified trial record dicts.
    """
    settings = get_settings()
    max_results = min(max_results, 50)

    url = f"{settings.ctgov_base_url}/studies"
    params = {
        "query.cond": condition,
        "pageSize": max_results,
        "format": "json",
    }

    logger.info("Querying ClinicalTrials.gov: %s", condition)

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    data = resp.json()
    studies = data.get("studies", [])
    results = [_parse_study(s) for s in studies]
    logger.info("Retrieved %d trials for '%s'", len(results), condition)
    return results


@tool
def get_trial_details(nct_id: str) -> dict[str, Any]:
    """Fetch full details for a single trial by NCT ID.

    Args:
        nct_id: The NCT identifier (e.g. "NCT04000165").

    Returns:
        Detailed trial record dict.
    """
    settings = get_settings()
    url = f"{settings.ctgov_base_url}/studies/{nct_id}"
    params = {"format": "json"}

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    study = resp.json()
    return _parse_study(study)
