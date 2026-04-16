# TrialPilot: Autonomous Multi-Agent System for Cross-Trial Synthesis

An AI-powered multi-agent system built with **LangGraph** that acts as a Virtual Clinical Research Associate. It analyzes historical clinical trials, simulates patient cohorts, runs biostatistical analyses, and reviews designs against regulatory guidelines — all autonomously.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LangGraph Pipeline                     │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Extraction   │───▶│  Simulation  │───▶│ Biostatistics│  │
│  │  "Librarian"  │    │ "Digital Twin"│    │  "Analyst"   │  │
│  └──────┬───────┘    └──────────────┘    └──────┬───────┘  │
│         ▲                                        │          │
│         │            ┌──────────────┐            │          │
│         └────────────│  Regulatory  │◀───────────┘          │
│    (if revision      │  "Ethicist"  │                       │
│     needed)          └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Role | Tools |
|-------|------|-------|
| **Extraction Agent** ("Librarian") | Crawls ClinicalTrials.gov, classifies trial outcomes (SUCCESS/FAILURE/UNKNOWN), identifies criteria patterns | ClinicalTrials.gov v2 API |
| **Patient Simulation Agent** ("Digital Twin") | Generates synthetic patient cohorts, applies eligibility criteria, flags demographic diversity issues | Synthetic data generator (NumPy) |
| **Biostatistical Agent** ("Analyst") | Runs Kaplan-Meier survival analysis, estimates Probability of Success (PoS), recommends sample sizes | lifelines, scipy |
| **Regulatory Critic** ("Ethicist") | Reviews design against FDA/EMA guidelines using Chain-of-Thought reasoning, flags biases | LLM reasoning (CoT) |

The Regulatory Critic can **reject** a design and send it back to the Extraction Agent for refinement, creating a feedback loop.

## Setup

```bash
# Clone and install
cd "TrialPilot- MAS For Cross Trial Synthesis"
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

```bash
# Basic run
trialpilot "Immunotherapy for Stage IV Lung Cancer"

# With custom goal and output
trialpilot "CAR-T therapy for Diffuse Large B-Cell Lymphoma" \
  --goal "Identify optimal inclusion criteria for first-line treatment" \
  --max-iterations 3 \
  --output results.json \
  --verbose
```

### Programmatic Usage

```python
from trialpilot.graph import run_pipeline

result = run_pipeline(
    condition="Immunotherapy for Stage IV Lung Cancer",
    research_goal="Design a Phase III trial with optimized eligibility criteria",
    max_iterations=2,
)

print(result["proposed_design"])
print(result["regulatory_review"])
```

## Testing

```bash
pytest -v
```

## Project Structure

```
src/trialpilot/
├── __init__.py
├── config.py            # Settings from environment
├── schemas.py           # Pydantic models & LangGraph state
├── graph.py             # LangGraph pipeline definition
├── main.py              # CLI entry point
├── agents/
│   ├── extraction.py    # Librarian — trial search & classification
│   ├── simulation.py    # Digital Twin — synthetic cohort generation
│   ├── biostatistics.py # Analyst — survival analysis & PoS
│   └── regulatory.py    # Ethicist — regulatory review & CoT
└── tools/
    ├── clinical_trials_api.py  # ClinicalTrials.gov v2 client
    ├── patient_generator.py    # Synthetic patient data
    └── statistics.py           # KM analysis & PoS computation
```

## Roadmap

- **Phase 2**: Uncertainty quantification — confidence intervals from all agents
- **Phase 3**: Human-in-the-loop dashboard for real-time doctor feedback
- **Phase 4**: Benchmarking paper on agentic workflow efficiency

## License

MIT
