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
AZURE_OPENAI_API_VERSION="2025-03-01-preview"
```

Authenticate using Azure CLI:
```bash
az login
```

## Quick Start

### Generate Conversation Data

```python
from persona_gym.data_generators import PersonaMemV2Generator, MultiSessionGenerator

# V2: Token-budgeted generation with preference evolution
generator = PersonaMemV2Generator(
    topic="travel",
    token_budget=8000,
    num_preferences=5,
)
output = generator.generate()

# Multi-session: Life-event driven preference evolution
generator = MultiSessionGenerator(
    persona="Software engineer considering career change...",
    num_sessions=2,
    num_preferences=5,
)
output = generator.generate()
```

### Generate Tasks

```bash
python -m persona_gym.task_generator \
    --input outputs/travel/conversation_artifacts.json
```

### Evaluate Agents

```bash
python -m persona_gym.evaluation \
    --tasks outputs/travel/tasks.jsonl \
    --context outputs/travel/conversation.json \
    --agent context  # or no_context
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
    ├── schemas.py                # All data models
    ├── client.py                 # Shared LLM client
    ├── task_generator.py         # Task generation
    ├── evaluation.py             # Evaluation runner
    ├── metric.py                 # Metrics and scoring
    ├── agent.py                  # Agent implementations
    ├── tool_simulator.py         # Tool simulation
    ├── prompts/                  # Centralized prompt management
    ├── data_generators/          # Data generation strategies
    │   ├── personamem_v2.py      # Token-budgeted generation
    │   └── multisession.py       # Multi-session generation
    └── evaluation_multisession/  # Multi-session evaluation
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
