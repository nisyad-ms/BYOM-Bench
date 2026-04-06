# REAM-Bench

A benchmark for evaluating LLM personalization through multi-session conversations with evolving user preferences.

For a minimal setup guide, see [QUICK_START.md](QUICK_START.md).

## Overview

REAM-Bench measures how well AI agents remember and proactively use user preferences across multiple conversation sessions. Preferences evolve over time due to life events, testing whether agents can:

- **Recall and apply current preferences** proactively without being asked
- **Avoid using stale/superseded preferences** that have been replaced
- **Minimize clarifying questions** by leveraging known context

### How Evaluation Works

Unlike simple Q&A benchmarks, REAM-Bench evaluates agents through **multi-turn task completion**. At evaluation time, the agent is given a multi-turn task that requires proactively applying multiple user preferences to complete successfully. A simulated user interacts with the agent, and evaluation metrics are calculated based on the entire conversation—measuring whether the agent applied preferences before being asked, avoided stale preferences, and completed the task efficiently.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ream_bench.git
cd ream_bench

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .

# Optional memory stores (third-party)
pip install -e ".[mem0]"        # Mem0 open-source memory layer
pip install -e ".[zep]"         # Graphiti temporal knowledge graph (Kùzu)
pip install -e ".[hindsight]"   # Hindsight biomimetic memory (bundles torch + embedded PostgreSQL)
pip install -e ".[all]"         # All optional stores

# To add your own memory backend, see BRING_YOUR_OWN_MEMORY.md
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Azure OpenAI is required for all agents. Foundry, Google, and AWS credentials are only needed for their respective agents. See [`.env.example`](.env.example) for all available settings.

Pipeline behavior is controlled by YAML files in `configs/`:

| File | Description |
|------|-------------|
| `llm.yaml` | Shared LLM defaults — temperature, max tokens per call type, retry settings |
| `pipeline.yaml` | Data and task generation defaults — number of sessions, preferences per task, evolved preference ratio |
| `prompts.yaml` | Prompt version mappings — maps each prompt to a versioned template file |
| `agents/<agent>.yaml` | Per-agent configs — retry, timeouts, polling intervals, search limits |

Authentication:

```bash
az login                              # Azure
gcloud auth application-default login # Google (if using google agent)
```

## Using the Pre-built Dataset

The `datasets/` folder contains pre-generated conversation sessions and evaluation tasks, ready for evaluation. To evaluate an agent on the existing dataset, pass `datasets/v1` as the `--outputs-dir`:

```bash
# Run evaluation on the pre-built dataset
uv run python scripts/test_evaluation.py --outputs-dir datasets/v1 --session all --agent foundry

# Gather results
uv run python scripts/gather_results.py --outputs-dir datasets/v1
```

To generate your own data from scratch, see the pipeline stages below.

## Running the Pipeline

All scripts are in `scripts/`. Run from the repo root.

### Base Personas

The repo includes `data/base_personas.json` — a JSON object mapping domain names to arrays of persona descriptions:

```json
{
  "software development": [
    "A mid-career backend developer at a fintech startup",
    "A 24-year-old mobile app developer freelancing full-time"
  ],
  "finance and accounting": [
    "A 32-year-old mid-career accountant at a regional firm specializing in nonprofits",
    "A freelance financial consultant with clients across multiple industries"
  ]
}
```

### Stage 1: Data Generation

Generates multi-session conversation data with evolving preferences. A **session** is one complete multi-turn conversation between the user and an AI assistant. A complete dataset for a persona contains N sessions, where preferences evolve across sessions via life events.

This is the most time-consuming step of the pipeline. For quick testing, use the default `--sessions 2` which generates 2 sessions and takes a few minutes.

```bash
uv run python scripts/test_data_generation.py --outputs-dir <dir>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sessions N` | `2` | Number of sessions to generate per persona |
| `--persona VALUE` | 1 random | `"all"` or a domain name (e.g., `"software development"`). If not set, picks 1 random persona. |
| `--num N` | all | Use first N personas from the resolved list. Only valid with `--persona`. |

Output: `outputs/<timestamp>/sessions.json` (one folder per persona)

### Stage 2: Task Generation

Creates evaluation tasks from the generated session data.

```bash
uv run python scripts/test_task_generation.py --outputs-dir <dir>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session NAME` | latest | Session name — the `<timestamp>` folder created by Stage 1 (one per persona) |
| `--count N` | `1` | Number of tasks to generate (parallel if > 1) |
| `--version VERSION` | auto-increment | Task version (e.g., `v1`, `v2`) |

Output: `outputs/<timestamp>/tasks/<version>/task_XX.json`

### Stage 3: Evaluation

Runs agent dialogue and scoring on generated tasks.

```bash
uv run python scripts/test_evaluation.py --outputs-dir <dir> --session <name> --agent context
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session NAME` | *(required)* | `"all"` or a session name — the `<timestamp>` folder created by Stage 1 (one per persona) |
| `--agent TYPE` | `context` | Agent type: `context`, `nocontext`, `foundry`, `google`, `aws`, `mem0`, `zep`, `hindsight` |
| `--task TASKS` | `all` | `"all"` or comma-separated task numbers (e.g., `01,02,03`) |
| `--task-version VERSION` | latest | Task version (e.g., `v1`) |
| `--num-runs N` | `1` | Number of evaluation runs per task |
| `--max-agent-turns N` | `10` | Maximum agent turns in dialogue |
| `--eval-run TIMESTAMP` | new run | Resume into an existing eval run, skipping completed tasks |

Output: `outputs/<timestamp>/evaluations/<eval_timestamp>/eval_XX_<agent>.json`

### Gathering Results

Aggregates evaluation results into an Excel file.

```bash
uv run python scripts/gather_results.py --outputs-dir <dir>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--eval-run TIMESTAMP` | latest per session | Specific eval run timestamp (e.g., `2026-02-09_143022`) |

Output: `outputs/results_<timestamp>.xlsx`

## End-to-End Example

```bash
# 1. Generate session data (1 random persona, 10 sessions)
uv run python scripts/test_data_generation.py --outputs-dir <dir> --sessions 10

# 2. Generate evaluation tasks
uv run python scripts/test_task_generation.py --outputs-dir <dir> --count 3

# 3. Run evaluations
uv run python scripts/test_evaluation.py --outputs-dir <dir> --session all --agent context --task-version v1
uv run python scripts/test_evaluation.py --outputs-dir <dir> --session all --agent foundry --task-version v1

# 4. Gather results
uv run python scripts/gather_results.py --outputs-dir <dir>
```

## Evaluation Metrics

### Preference Recall

Measures proactive preference recall:

$$\text{preference recall} = \max\left(0, \frac{\text{proactive} - \text{stale}}{\text{total required}}\right)$$

- **PROACTIVE**: Agent mentioned/applied preference before the user did (+1)
- **IGNORED**: User mentioned the preference first (no contribution)
- **STALE**: Agent used an outdated/superseded preference (-1)

### Stale Recall Rate

Measures how often the agent uses outdated preferences that have been superseded by newer ones:

$$\text{stale recall rate} = \frac{\text{stale count}}{\text{evolved preference count}}$$

A lower rate is better. Only evolved preferences (those that replaced an older version) are considered.

### Task Completion

Binary metric: **1** if `preference_recall == 1.0` (all preferences recalled proactively with no stale usage), **0** otherwise.

## Project Structure

```
├── scripts/
│   ├── test_data_generation.py     # Stage 1 entry point
│   ├── test_task_generation.py     # Stage 2 entry point
│   ├── test_evaluation.py          # Stage 3 entry point
│   └── gather_results.py           # Result aggregation
├── ream_bench/                     # Main package
│   ├── agents/                     # Agent implementations
│   │   └── stores/                 # Memory store backends
│   ├── client.py                   # Azure OpenAI client (Responses API + tenacity retry)
│   ├── data_generators/            # Stage 1: Data generation
│   ├── evaluation_multisession/    # Stage 3: Evaluation (judge, runner, user simulator)
│   ├── formatting.py               # Preference history formatting and event summarization
│   ├── prompts/                    # YAML prompt templates
│   ├── schemas.py                  # All data models
│   ├── task_generators/            # Stage 2: Task generation
│   └── utils.py                    # Output path helpers and regex patterns
├── configs/                        # Configuration files
│   ├── agents/                     # Per-agent configs (foundry.yaml, google.yaml, etc.)
│   ├── llm.yaml                    # Shared LLM defaults
│   ├── pipeline.yaml               # Data/task generation defaults
│   └── prompts.yaml                # Prompt version mappings
├── datasets/                       # Pre-built evaluation datasets
│   └── v1/                         # 50 personas, 10 sessions each, 5 tasks per session
├── data/                           # Base personas and domain list
│   ├── base_personas.json          # Persona definitions keyed by domain
│   ├── domains.txt                 # Domain list
│   └── generate_personas.py        # Persona generation script
├── docs/                           # Pipeline documentation
├── outputs/                        # Generated outputs (gitignored)
│   └── <timestamp>/                # One per persona
│       ├── sessions.json
│       ├── tasks/<version>/task_XX.json
│       └── evaluations/<eval_timestamp>/eval_XX_<agent>.json
└── pyproject.toml
```

## Agent Types

| Agent | Description | Install |
|-------|-------------|---------|
| **context** | Full ground-truth preference list provided (upper bound) | core |
| **nocontext** | No past context provided (lower bound) | core |
| **foundry** | Azure AI Foundry memory store | core |
| **google** | Google Vertex AI Agent Engine memory | core |
| **aws** | AWS Bedrock AgentCore memory | core |
| **mem0** | Mem0 open-source memory layer | `.[mem0]` |
| **zep** | Graphiti temporal knowledge graph (Kùzu) | `.[zep]` |
| **hindsight** | Hindsight biomimetic memory | `.[hindsight]` |

To add a custom memory backend, see [BRING_YOUR_OWN_MEMORY.md](BRING_YOUR_OWN_MEMORY.md).

For the latest benchmark results, see [RESULTS.md](RESULTS.md).

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Set up pre-commit hooks (required — runs gitleaks on every commit)
pre-commit install

# Linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright

# Tests
uv run pytest tests/
```
