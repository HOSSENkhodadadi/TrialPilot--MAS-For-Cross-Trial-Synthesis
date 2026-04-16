"""Tests for TrialPilot tools and pipeline components."""

from __future__ import annotations

import pytest

from trialpilot.schemas import (
    PipelineState,
    ProposedTrialDesign,
    RegulatoryFinding,
    StatisticalResult,
    SyntheticPatient,
    TrialRecord,
)


class TestSchemas:
    """Test that all Pydantic models validate correctly."""

    def test_trial_record_minimal(self):
        t = TrialRecord(nct_id="NCT00000001")
        assert t.nct_id == "NCT00000001"
        assert t.outcome_result == ""

    def test_synthetic_patient(self):
        p = SyntheticPatient(
            patient_id="SYN-001",
            age=55,
            sex="Female",
            race="White",
            primary_diagnosis="Lung Cancer",
            comorbidities=["Hypertension"],
            ecog_score=1,
        )
        assert p.eligible is None
        assert p.ecog_score == 1

    def test_statistical_result(self):
        s = StatisticalResult(
            eligible_fraction=0.35,
            probability_of_success=0.42,
        )
        assert 0 <= s.probability_of_success <= 1

    def test_regulatory_finding(self):
        f = RegulatoryFinding(
            category="BIAS",
            severity="HIGH",
            description="Age cutoff excludes elderly",
            recommendation="Raise age cutoff to 80",
        )
        assert f.severity == "HIGH"


class TestPatientGenerator:
    """Test synthetic patient generation tool."""

    def test_generate_patients(self):
        from trialpilot.tools.patient_generator import generate_synthetic_patients

        patients = generate_synthetic_patients.invoke({
            "condition": "Lung Cancer",
            "n": 10,
            "seed": 42,
        })
        assert len(patients) == 10
        for p in patients:
            assert 18 <= p["age"] <= 90
            assert p["sex"] in ("Male", "Female")
            assert "PD-L1_TPS" in p["biomarkers"]
            assert p["survival_months"] > 0

    def test_reproducible_with_seed(self):
        from trialpilot.tools.patient_generator import generate_synthetic_patients

        p1 = generate_synthetic_patients.invoke({"condition": "X", "n": 5, "seed": 99})
        p2 = generate_synthetic_patients.invoke({"condition": "X", "n": 5, "seed": 99})
        assert [p["age"] for p in p1] == [p["age"] for p in p2]


class TestStatistics:
    """Test statistical analysis tools."""

    def test_survival_analysis_insufficient(self):
        from trialpilot.tools.statistics import run_survival_analysis

        patients = [{"survival_months": 12, "eligible": True}]
        result = run_survival_analysis.invoke({"patients": patients})
        assert result["median_survival_months"] is None

    def test_survival_analysis_basic(self):
        from trialpilot.tools.patient_generator import generate_synthetic_patients
        from trialpilot.tools.statistics import run_survival_analysis

        patients = generate_synthetic_patients.invoke({"condition": "Test", "n": 50, "seed": 1})
        for p in patients:
            p["eligible"] = True

        result = run_survival_analysis.invoke({"patients": patients})
        assert result["median_survival_months"] is not None
        assert result["eligible_count"] == 50

    def test_pos_no_improvement(self):
        from trialpilot.tools.statistics import compute_probability_of_success

        result = compute_probability_of_success.invoke({
            "eligible_fraction": 0.3,
            "median_survival_months": 8.0,
            "historical_control_median": 10.0,
        })
        # median < control → no improvement path
        assert result["probability_of_success"] < 0.5

    def test_pos_with_improvement(self):
        from trialpilot.tools.statistics import compute_probability_of_success

        result = compute_probability_of_success.invoke({
            "eligible_fraction": 0.4,
            "median_survival_months": 18.0,
            "historical_control_median": 10.0,
        })
        assert result["probability_of_success"] > 0
        assert result["hazard_ratio"] < 1.0


class TestGraph:
    """Test graph construction (no LLM calls)."""

    def test_graph_compiles(self):
        from trialpilot.graph import build_graph

        graph = build_graph()
        assert graph is not None
