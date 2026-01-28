# PersonaGym

A benchmark for evaluating LLM personalization through multi-session conversations with evolving user preferences.

## Overview

PersonaGym measures how well AI agents remember and proactively use user preferences across multiple conversation sessions. Preferences evolve over time due to life events, testing whether agents can:
- Recall and apply current preferences
- Avoid using stale/superseded preferences
- Minimize clarifying questions by leveraging known context

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PersonaGym Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│  INPUT: Persona description                                             │
│           ↓                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────┐   │
│  │ 1. DATA GENERATION              │→ │ 2. EVALUATION               │   │
│  │ - Generate life events          │  │ - Generate evaluation task  │   │
│  │ - Create preferences            │  │ - Run agent dialogue        │   │
│  │ - Evolve preferences over time  │  │ - Judge performance         │   │
│  │ - Generate conversations        │  │ - Compute scores            │   │
│  └─────────────────────────────────┘  └─────────────────────────────┘   │
│           ↓                                   ↓                         │
│  OUTPUT: MultiSessionOutput          Evaluation Scores                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/persona_gym.git
cd persona_gym

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Configuration

Create a `.env` file with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="gpt-4"
AZURE_OPENAI_API_VERSION="2025-03-01-preview"
```

Authenticate using Azure CLI:
```bash
az login
```

## Quick Start

### Generate Multi-Session Data

```python
from persona_gym.data_generators import MultiSessionGenerator

generator = MultiSessionGenerator(
    persona="A 32-year-old software engineer considering a career change...",
    num_sessions=2,
    num_preferences=3,  # 3 preferences per life event
)
result = generator.generate_multi_session()

print(f"Sessions: {len(result.sessions)}")
print(f"Total preferences: {len(result.timeline.preferences)}")
print(f"Active preferences: {len(result.timeline.get_active_preferences())}")
```

### Run Evaluation

```python
from persona_gym.evaluation_multisession import run_evaluation_from_file

# Run with full context (agent has access to preference history)
result = run_evaluation_from_file("outputs/test_multisession_output.json", agent_type="full")
print(f"Score: {result.final_score:.2f}")

# Run without context (baseline)
result = run_evaluation_from_file("outputs/test_multisession_output.json", agent_type="no_context")
print(f"Score: {result.final_score:.2f}")
```

Or use the test scripts:
```bash
# Generate test data
python test_multisession.py

# Run evaluation
python test_evaluation.py --full      # Full context agent
python test_evaluation.py --no-context  # No context baseline
```

## Project Structure

```
persona_gym/
├── pyproject.toml                # Package configuration
├── data/source/                  # Source persona data
├── outputs/                      # Generated outputs
├── logs/                         # Log files
├── test_multisession.py          # Test data generation
├── test_evaluation.py            # Test evaluation
└── persona_gym/                  # Main package
    ├── __init__.py
    ├── schemas.py                # All data models
    ├── client.py                 # Shared LLM client
    ├── data_generators/          # Data generation
    │   ├── base.py               # Base generator class
    │   └── multisession.py       # Multi-session generator
    ├── evaluation_multisession/  # Evaluation system
    │   ├── orchestrator.py       # Task generation
    │   ├── judge.py              # LLM judge
    │   ├── user_simulator.py     # User simulation
    │   └── runner.py             # Evaluation runner
    └── prompts/                  # YAML prompt templates
        ├── data_generation/      # Data gen prompts
        └── evaluation/           # Evaluation prompts
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Preference Score** | Did the agent proactively use current preferences? |
| **Efficiency Score** | Did the agent avoid unnecessary clarifying questions? |
| **Task Completion** | Did the agent complete the requested task? |
| **Final Score** | Weighted combination of all metrics |

## Key Concepts

### Preference Evolution
Preferences change over time due to life events. For example:
- "Prefers morning study sessions" → life event: "Started new job with early meetings" → "Prefers evening study sessions"

### Evaluation Task Design
Tasks are designed to **require** preference knowledge:
- Each task has `required_preferences` that must be applied
- Generic advice without preference awareness scores poorly
- Agent must demonstrate explicit knowledge of user context

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
python test_multisession.py  # Data generation
python test_evaluation.py --full  # Evaluation
```
