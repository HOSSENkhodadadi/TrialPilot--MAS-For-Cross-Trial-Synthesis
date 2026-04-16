"""LangGraph workflow definition for TrialPilot.

Orchestrates the four agents in a directed graph with a conditional feedback loop:

  Extraction → Simulation → Biostatistics → Regulatory
       ↑                                         │
       └──────── (if needs_revision) ────────────┘

"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from trialpilot.agents.biostatistics import biostatistics_node
from trialpilot.agents.extraction import extraction_node
from trialpilot.agents.regulatory import regulatory_node
from trialpilot.agents.simulation import simulation_node
from trialpilot.schemas import PipelineState

logger = logging.getLogger(__name__)


def _should_revise(state: PipelineState) -> Literal["extraction", "end"]:
    """Conditional edge: decide whether to loop back for revision."""
    if state.get("needs_revision", False):
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 2)
        if iteration < max_iter:
            logger.info("Regulatory review requested revision — looping back (iter %d)", iteration)
            return "extraction"
    return "end"


def build_graph() -> StateGraph:
    """Construct and compile the TrialPilot LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Register agent nodes
    graph.add_node("extraction", extraction_node)
    graph.add_node("simulation", simulation_node)
    graph.add_node("biostatistics", biostatistics_node)
    graph.add_node("regulatory", regulatory_node)

    # Define edges: linear pipeline with conditional loop-back
    graph.set_entry_point("extraction")
    graph.add_edge("extraction", "simulation")
    graph.add_edge("simulation", "biostatistics")
    graph.add_edge("biostatistics", "regulatory")

    # Conditional: Regulatory Critic can send back to Extraction
    graph.add_conditional_edges(
        "regulatory",
        _should_revise,
        {
            "extraction": "extraction",
            "end": END,
        },
    )

    return graph.compile()


def run_pipeline(
    condition: str,
    research_goal: str = "",
    max_iterations: int = 2,
) -> dict[str, Any]:
    """Execute the full TrialPilot pipeline.

    Args:
        condition: Medical condition to research (e.g., "Immunotherapy for Stage IV Lung Cancer").
        research_goal: Optional high-level goal description.
        max_iterations: Maximum feedback-loop iterations (default 2).

    Returns:
        Final pipeline state dict.
    """
    graph = build_graph()

    initial_state: PipelineState = {
        "condition": condition,
        "research_goal": research_goal or f"Design an optimal clinical trial for: {condition}",
        "extracted_trials": [],
        "synthetic_patients": [],
        "statistical_analysis": {},
        "regulatory_review": {},
        "proposed_design": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "needs_revision": False,
        "agent_logs": [],
    }

    logger.info("Starting TrialPilot pipeline for: %s", condition)
    final_state = graph.invoke(initial_state)
    logger.info("Pipeline complete after %d iteration(s)", final_state.get("iteration", 0))

    return final_state
