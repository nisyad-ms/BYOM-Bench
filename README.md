# PersonaGym

A benchmark for evaluating LLM personalization through multi-turn conversations with embedded user preferences.

## Overview

PersonaGym generates synthetic user-assistant conversations grounded in persona histories, then evaluates how well LLMs can recall and use those preferences in task-oriented dialogues.

### Key Components

1. **Data Generation** - Creating synthetic user-assistant conversations grounded in persona histories
2. **TOD Task Generation** - Generating task-oriented dialogue tasks from conversation data  
3. **Agent Evaluation** - Evaluating agents on their ability to recall and apply user preferences

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

### Azure OpenAI Setup

Create a `.env` file with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="gpt-4"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

Authenticate using Azure CLI:
```bash
az login
```

## Quick Start

### Generate Conversation Data

```bash
# Debug mode (~1 minute) - minimal for testing
python -m persona_gym.data_generation --topic travel --debug

# Quick mode (~2-3 minutes) - init + week steps
python -m persona_gym.data_generation --topic travel --quick

# Full generation (~15+ minutes) - all steps
python -m persona_gym.data_generation --topic travel
```

**Supported Topics:** travel, therapy, food, writing, email, coding, legal

### Generate Tasks

```bash
python -m persona_gym.task_generator \
    --input outputs/travel/sample_conversation_travel_persona0_sample0_debug_artifacts.json
```

### Evaluate Agents

```bash
python -m persona_gym.evaluation \
    --tasks outputs/travel/sample_conversation_travel_persona0_sample0_debug_tasks.jsonl \
    --context outputs/travel/sample_conversation_travel_persona0_sample0_debug_conversation.json \
    --agent no_context
```

## Project Structure

```
persona_gym/
├── pyproject.toml                # Package configuration
├── README.md                     # This file
├── data/source/                  # Source persona data
├── outputs/                      # Generated outputs
├── logs/                         # Log files
└── persona_gym/                  # Main package
    ├── __init__.py
    ├── config.yaml               # Configuration file
    ├── data_generation.py        # Data generation entry point
    ├── task_generator.py         # Task generation
    ├── evaluation.py             # Evaluation runner
    ├── metric.py                 # Metrics and scoring
    ├── agent.py                  # Agent implementations
    ├── tool_simulator.py         # Tool simulation
    └── personamem_core/          # Core generation modules
        ├── schemas.py            # Pydantic models
        ├── prompts.py            # LLM prompts
        ├── prepare_data.py       # Data preparation
        ├── query_llm.py          # LLM interface
        └── utils.py              # Utilities
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Preference Recall** | Did the agent proactively use known preferences? |
| **Turn Efficiency** | Did the agent avoid unnecessary clarification questions? |
| **Task Completion** | Did the agent successfully complete the task? |

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
pytest

# Format code
ruff format persona_gym/

# Lint
ruff check persona_gym/
```

## License

MIT License
