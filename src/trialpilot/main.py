"""CLI entry point for TrialPilot."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trialpilot.graph import run_pipeline

console = Console()


def _print_results(state: dict) -> None:
    """Pretty-print the pipeline results."""
    # Header
    console.print(Panel.fit(
        f"[bold green]TrialPilot Report[/bold green]\n"
        f"Condition: {state.get('condition', 'N/A')}\n"
        f"Iterations: {state.get('iteration', 0)}",
        border_style="green",
    ))

    # Extracted trials summary
    trials = state.get("extracted_trials", [])
    console.print(f"\n[bold]Extracted Trials:[/bold] {len(trials)} found")

    # Synthetic cohort summary
    patients = state.get("synthetic_patients", [])
    eligible = sum(1 for p in patients if p.get("eligible"))
    console.print(f"[bold]Synthetic Cohort:[/bold] {len(patients)} patients, {eligible} eligible "
                  f"({eligible/len(patients):.0%})" if patients else "No patients generated")

    # Statistical analysis
    stats = state.get("statistical_analysis", {})
    if stats:
        survival = stats.get("survival_results", {})
        pos = stats.get("pos_results", {})
        table = Table(title="Statistical Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Median Survival", f"{survival.get('median_survival_months', 'N/A')} months")
        table.add_row("12-mo Survival Rate", f"{survival.get('survival_at_12mo', 'N/A')}")
        table.add_row("Probability of Success", f"{pos.get('probability_of_success', 'N/A')}")
        table.add_row("Hazard Ratio", f"{pos.get('hazard_ratio', 'N/A')}")
        table.add_row("Recommended N", f"{pos.get('recommended_sample_size', 'N/A')}")
        console.print(table)

    # Regulatory review
    review = state.get("regulatory_review", {})
    if review:
        status = "[green]APPROVED[/green]" if review.get("approved") else "[red]NOT APPROVED[/red]"
        console.print(f"\n[bold]Regulatory Review:[/bold] {status}")
        findings = review.get("findings", [])
        if findings:
            ft = Table(title="Findings")
            ft.add_column("Severity", style="bold")
            ft.add_column("Category")
            ft.add_column("Description")
            ft.add_column("Recommendation")
            for f in findings:
                sev = f.get("severity", "")
                style = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(sev, "white")
                ft.add_row(
                    f"[{style}]{sev}[/{style}]",
                    f.get("category", ""),
                    f.get("description", "")[:80],
                    f.get("recommendation", "")[:80],
                )
            console.print(ft)

    # Proposed design
    design = state.get("proposed_design", {})
    if design:
        console.print(Panel(
            json.dumps(design, indent=2, default=str),
            title="[bold]Proposed Trial Design[/bold]",
            border_style="blue",
        ))

    # Agent logs
    logs = state.get("agent_logs", [])
    if logs:
        console.print("\n[bold]Agent Activity Log:[/bold]")
        for log in logs:
            console.print(f"  • {log}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrialPilot: Autonomous Multi-Agent System for Cross-Trial Synthesis",
    )
    parser.add_argument(
        "condition",
        help="Medical condition to research (e.g., 'Immunotherapy for Stage IV Lung Cancer')",
    )
    parser.add_argument(
        "--goal", "-g",
        default="",
        help="High-level research goal description",
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=2,
        help="Maximum revision loop iterations (default: 2)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    console.print(Panel.fit(
        "[bold cyan]TrialPilot[/bold cyan] — Multi-Agent Clinical Trial Synthesis\n"
        f"Condition: {args.condition}",
        border_style="cyan",
    ))

    try:
        result = run_pipeline(
            condition=args.condition,
            research_goal=args.goal,
            max_iterations=args.max_iterations,
        )
    except Exception as exc:
        console.print(f"[bold red]Pipeline Error:[/bold red] {exc}")
        sys.exit(1)

    _print_results(result)

    if args.output:
        # Serialize the result (strip non-serializable parts)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        console.print(f"\nResults saved to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
