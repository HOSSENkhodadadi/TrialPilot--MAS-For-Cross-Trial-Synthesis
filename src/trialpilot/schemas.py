"""Pydantic models and LangGraph state schema for TrialPilot."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------


class TrialRecord(BaseModel):
    """A structured record extracted from a clinical trial."""

    nct_id: str = Field(description="ClinicalTrials.gov NCT identifier")
    title: str = ""
    condition: str = ""
    phase: str = ""
    status: str = ""
    enrollment: int | None = None
    start_date: str = ""
    completion_date: str = ""
    intervention: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    primary_outcome: str = ""
    outcome_result: str = Field(default="", description="SUCCESS / FAILURE / UNKNOWN")
    summary: str = ""


class SyntheticPatient(BaseModel):
    """A synthetic patient profile for trial feasibility testing."""

    patient_id: str
    age: int
    sex: str
    race: str
    primary_diagnosis: str = ""
    comorbidities: list[str] = Field(default_factory=list)
    biomarkers: dict[str, float] = Field(default_factory=dict)
    prior_treatments: list[str] = Field(default_factory=list)
    ecog_score: int = Field(default=1, ge=0, le=5)
    eligible: bool | None = Field(default=None, description="Set by eligibility check")
    survival_months: float | None = None


class StatisticalResult(BaseModel):
    """Output of the biostatistical analysis."""

    median_survival_months: float | None = None
    survival_ci_lower: float | None = None
    survival_ci_upper: float | None = None
    eligible_fraction: float = Field(description="Fraction of cohort that meets criteria")
    probability_of_success: float = Field(description="Estimated PoS for the trial")
    kaplan_meier_summary: str = ""
    sample_size_recommendation: int | None = None
    notes: str = ""


class RegulatoryFinding(BaseModel):
    """A single finding from the regulatory review."""

    category: str = Field(description="BIAS | SAFETY | EFFICACY | ETHICAL | DESIGN")
    severity: str = Field(description="HIGH | MEDIUM | LOW")
    description: str = ""
    recommendation: str = ""


class RegulatoryReview(BaseModel):
    """Collected output of the Regulatory Critic."""

    approved: bool = False
    findings: list[RegulatoryFinding] = Field(default_factory=list)
    chain_of_thought: str = ""
    suggested_criteria_changes: str = ""


class ProposedTrialDesign(BaseModel):
    """The proposed trial design assembled from agent outputs."""

    condition: str = ""
    phase: str = ""
    intervention: str = ""
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    primary_endpoint: str = ""
    target_enrollment: int | None = None
    estimated_pos: float | None = None
    statistical_summary: str = ""
    regulatory_summary: str = ""


# ---------------------------------------------------------------------------
# LangGraph Shared State
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Uses Annotated reducers so nodes append rather than overwrite list fields.
    """

    # Input
    condition: str
    research_goal: str

    # Extraction Agent outputs
    extracted_trials: Annotated[list[dict[str, Any]], operator.add]

    # Patient Simulation Agent outputs
    synthetic_patients: Annotated[list[dict[str, Any]], operator.add]

    # Biostatistical Agent outputs
    statistical_analysis: dict[str, Any]

    # Regulatory Critic outputs
    regulatory_review: dict[str, Any]

    # Assembled design
    proposed_design: dict[str, Any]

    # Control flow
    iteration: int
    max_iterations: int
    needs_revision: bool

    # Logging / trace
    agent_logs: Annotated[list[str], operator.add]
