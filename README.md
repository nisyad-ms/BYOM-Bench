# PersonaGym

A benchmark for evaluating LLM personalization through multi-session conversations with evolving user preferences.

## Overview

PersonaGym measures how well AI agents remember and proactively use user preferences across multiple conversation sessions. Preferences evolve over time due to life events, testing whether agents can:

- **Recall and apply current preferences** proactively without being asked
- **Avoid using stale/superseded preferences** that have been replaced
- **Minimize clarifying questions** by leveraging known context

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PersonaGym Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │ 1. DATA GEN      │ →  │ 2. TASK GEN      │ →  │ 3. EVALUATION    │      │
│  │                  │    │                  │    │                  │      │
│  │ • Expand persona │    │ • Select prefs   │    │ • Run dialogue   │      │
│  │ • Life events    │    │ • Create event   │    │ • Judge prefs    │      │
│  │ • Preferences    │    │ • User prompt    │    │ • Judge turns    │      │
│  │ • Conversations  │    │                  │    │ • Compute scores │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│         ↓                        ↓                       ↓                 │
│   sessions.json            task_XX.json          eval_XX_agent.json        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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

### Azure OpenAI

Create a `.env` file with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2025-03-01-preview"
AZURE_OPENAI_DEPLOYMENTS="gpt-4-001,gpt-4-002,gpt-4-003"  # Comma-separated for parallel execution
```

### Azure AI Foundry (Optional)

For the Foundry Memory Agent:

```bash
AZURE_FOUNDRY_ENDPOINT="https://your-foundry-endpoint.services.ai.azure.com/api/projects/your-project"
```

### Authentication

```bash
az login
```

## Running Components

### Stage 1: Data Generation

Generates multi-session conversation data with evolving preferences.

```bash
uv run python test_data_generation.py
```

Output: `outputs/<timestamp>/sessions.json`

### Stage 2: Task Generation

Creates evaluation tasks from the generated session data.

```bash
# Generate a single task
uv run python test_task_generation.py

# Generate multiple tasks in parallel
uv run python test_task_generation.py --count 3

# Use a specific session
uv run python test_task_generation.py --session 2026-02-02_1414
```

Output: `outputs/<timestamp>/tasks/task_XX.json`

### Stage 3: Evaluation

Runs agent dialogue and scoring on generated tasks.

```bash
# Context agent (has full conversation history)
uv run python test_evaluation.py --agent context

# No-context agent (baseline, no memory)
uv run python test_evaluation.py --agent nocontext

# Foundry memory agent (Azure AI Foundry memory store)
uv run python test_evaluation.py --agent foundry

# Run all tasks in parallel
uv run python test_evaluation.py --task all --agent context

# Force recreate memory store (foundry only)
uv run python test_evaluation.py --agent foundry --no-cache
```

Output: `outputs/<timestamp>/evaluation/eval_XX_<agent>.json`

## End-to-End Test

Run the complete pipeline:

```bash
# 1. Generate session data
uv run python test_data_generation.py

# 2. Generate evaluation tasks
uv run python test_task_generation.py --count 3

# 3. Run evaluations for all agents
uv run python test_evaluation.py --task all --agent context
uv run python test_evaluation.py --task all --agent nocontext
```

## Evaluation Metrics

### Preference Score

Measures proactive preference recall:

```
preference_score = max(0, (proactive_count - stale_count) / total_required)
```

- **PROACTIVE**: Agent mentioned preference before user (+1)
- **STALE**: Agent used superseded preference (-1)

### Efficiency Score

Measures conversation efficiency:

```
efficiency_score = max(0, (agent_turns - 0.5 × clarifying - corrections) / agent_turns)
```

- **PRODUCTIVE**: Agent provided useful, preference-aware response
- **IGNORED**: User mentioned preference first (no penalty)
- **CLARIFYING**: Agent asked about known preference (-0.5)
- **CORRECTION**: Agent made wrong suggestion (-1.0)

## Project Structure

```
persona_gym/
├── configs/
│   ├── client_config.yaml      # LLM client settings
│   └── prompt_config.yaml      # Prompt version configuration
├── data/source/                # Source persona data
├── docs/                       # Documentation and examples
├── outputs/                    # Generated outputs
│   └── <timestamp>/
│       ├── sessions.json       # Stage 1 output
│       ├── tasks/              # Stage 2 output
│       │   └── task_XX.json
│       └── evaluation/         # Stage 3 output
│           └── eval_XX_<agent>.json
├── logs/                       # Log files (mirrors output structure)
├── persona_gym/                # Main package
│   ├── agents/                 # Agent implementations
│   │   ├── base.py             # ContextAwareAgent, NoContextAgent
│   │   └── foundry_agent.py    # FoundryMemoryAgent
│   ├── client.py               # Shared Azure OpenAI client
│   ├── data_generators/        # Stage 1: Data generation
│   │   └── multisession.py     # MultiSessionGenerator
│   ├── evaluation_multisession/# Stage 3: Evaluation
│   │   ├── judge.py            # Preference + efficiency judges
│   │   ├── runner.py           # Evaluation orchestration
│   │   └── user_simulator.py   # Simulated user for dialogue
│   ├── prompts/                # YAML prompt templates
│   │   ├── data_generation/
│   │   ├── evaluation/
│   │   └── task_generation/
│   ├── schemas.py              # All data models
│   └── task_generators/        # Stage 2: Task generation
│       └── evaluation_task.py
├── test_data_generation.py     # Stage 1 test script
├── test_task_generation.py     # Stage 2 test script
├── test_evaluation.py          # Stage 3 test script
├── utils.py                    # Shared utilities
└── pyproject.toml              # Package configuration
```

## Agent Types

| Agent | Description | Expected Scores |
|-------|-------------|-----------------|
| **context** | Full conversation history access | preference ~0.8-1.0 |
| **nocontext** | No past context (baseline) | preference ~0.0, efficiency ~1.0 |
| **foundry** | Azure AI Foundry memory store | Varies by memory quality |

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Linting
ruff check .
ruff format .

# Type checking
mypy persona_gym
```
