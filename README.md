# MemoryGym

A benchmark for evaluating LLM personalization through multi-session conversations with evolving user preferences.

## Overview

MemoryGym measures how well AI agents remember and proactively use user preferences across multiple conversation sessions. Preferences evolve over time due to life events, testing whether agents can:

- **Recall and apply current preferences** proactively without being asked
- **Avoid using stale/superseded preferences** that have been replaced
- **Minimize clarifying questions** by leveraging known context

### How Evaluation Works

Unlike simple Q&A benchmarks, MemoryGym evaluates agents through **multi-turn task completion**. At evaluation time, the agent is given a multi-turn task that requires proactively applying multiple user preferences to complete successfully. A simulated user interacts with the agent, and evaluation metrics are calculated based on the entire conversation—measuring whether the agent applied preferences before being asked, avoided stale preferences, and completed the task efficiently.

## Pipeline Architecture

```
│                          MemoryGym Pipeline                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
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
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

For a detailed explanation of the pipeline, see [docs/pipeline_overview.md](docs/pipeline_overview.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/memory_gym.git
cd memory_gym

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
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

## Running the Pipeline

All scripts are in `scripts/`. Run from the repo root.

### Base Personas

Before running the pipeline, you need a `data/` directory at the repo root with persona files (gitignored):

- `data/base_personas.json` — Full persona set, keyed by domain
- `data/base_personas_test.json` — Small test set for quick runs
- `data/base_personas_single.json` — Minimal set (3 personas) for quick end-to-end testing

Format — a JSON object mapping domain names to arrays of persona descriptions:

```json
{
  "software development": [
    "A mid-career backend developer at a fintech startup",
    "A 24-year-old mobile app developer freelancing full-time"
  ],
  "finance and accounting": [
    "A 32-year-old mid-career accountant at a regional firm specializing in nonprofits",
    "A freelance financial consultant with clients across multiple industries",
  ]
}
```

### Stage 1: Data Generation

Generates multi-session conversation data with evolving preferences. A **session** is one complete multi-turn conversation between the user and an AI assistant. A complete dataset for a persona contains N sessions, where preferences evolve across sessions via life events.

This is the most time-consuming step of the pipeline. For quick testing, use the default `--sessions 2` which generates 2 sessions and takes a few minutes.

```bash
uv run python scripts/test_data_generation.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sessions N` | `2` | Number of sessions to generate per persona |
| `--persona VALUE` | 1 random | `"single"`, `"test"`, `"all"`, or a domain name. If not set, picks 1 random persona from `base_personas.json`. |
| `--num N` | all | Use first N personas from the resolved list. Only valid with `--persona`. |

Output: `outputs/<timestamp>/sessions.json` (one folder per persona)

### Stage 2: Task Generation

Creates evaluation tasks from the generated session data.

```bash
uv run python scripts/test_task_generation.py
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
uv run python scripts/test_evaluation.py --session <name> --agent context
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session NAME` | *(required)* | `"all"` or a session name — the `<timestamp>` folder created by Stage 1 (one per persona) |
| `--agent TYPE` | `context` | Agent type: `context`, `nocontext`, `foundry`, `foundry_local`, `google`, `aws` |
| `--task TASKS` | `all` | `"all"` or comma-separated task numbers (e.g., `01,02,03`) |
| `--task-version VERSION` | latest | Task version (e.g., `v1`) |
| `--num-runs N` | `1` | Number of evaluation runs per task |
| `--max-agent-turns N` | `10` | Maximum agent turns in dialogue |
| `--eval-run TIMESTAMP` | new run | Resume into an existing eval run, skipping completed tasks |

Output: `outputs/<timestamp>/evaluations/<eval_timestamp>/eval_XX_<agent>.json`

### Gathering Results

Aggregates evaluation results into an Excel file.

```bash
uv run python scripts/gather_results.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--eval-run TIMESTAMP` | latest per session | Specific eval run timestamp (e.g., `2026-02-09_143022`) |

Output: `outputs/results_<timestamp>.xlsx`

## End-to-End Example

```bash
# 1. Generate session data (1 random persona, 10 sessions)
uv run python scripts/test_data_generation.py --sessions 10

# 2. Generate evaluation tasks
uv run python scripts/test_task_generation.py --count 3

# 3. Run evaluations
uv run python scripts/test_evaluation.py --session all --agent context --task-version v1
uv run python scripts/test_evaluation.py --session all --agent foundry_local --task-version v1

# 4. Gather results
uv run python scripts/gather_results.py
```

## Evaluation Metrics

### Preference Score

Measures proactive preference recall:

$$\text{preference\_score} = \max\left(0, \frac{\text{proactive} - \text{stale}}{\text{total\_required}}\right)$$

- **PROACTIVE**: Agent mentioned/applied preference before the user did (+1)
- **IGNORED**: User mentioned the preference first (no contribution)
- **STALE**: Agent used an outdated/superseded preference (-1)

### Efficiency Score

Measures conversation efficiency:

$$\text{penalty} = 0.5 \times \text{generic} + 0.5 \times \text{ignored} + 1.0 \times \text{correction}$$

$$\text{efficiency\_score} = \max\left(0, \frac{\text{agent\_turns} - \text{penalty}}{\text{agent\_turns}}\right)$$

If any **REPEATED_CORRECTION** occurs, the efficiency score is automatically **0.0**.

Turn classifications (evaluated using the user's next message as look-ahead):

- **PRODUCTIVE**: Agent applied a specific required preference and user accepted (no penalty)
- **GENERIC**: Helpful but unpersonalized advice, equally appropriate for any user (-0.5)
- **IGNORED**: Agent omitted a preference, user had to reveal or remind (-0.5)
- **CORRECTION**: Agent made a wrong suggestion that contradicts a preference (-1.0)
- **REPEATED_CORRECTION**: Agent violated the same preference again after being corrected (instant 0.0)

## Project Structure

```
├── scripts/
│   ├── test_data_generation.py     # Stage 1 entry point
│   ├── test_task_generation.py     # Stage 2 entry point
│   ├── test_evaluation.py          # Stage 3 entry point
│   └── gather_results.py           # Result aggregation
├── memory_gym/                     # Main package
│   ├── agents/                     # Agent implementations
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
├── data/                           # Base personas (gitignored, see above)
├── docs/                           # Pipeline documentation
├── outputs/                        # Generated outputs (gitignored)
│   └── <timestamp>/                # One per persona
│       ├── sessions.json
│       ├── tasks/<version>/task_XX.json
│       └── evaluations/<eval_timestamp>/eval_XX_<agent>.json
└── pyproject.toml
```

## Agent Types

| Agent | Description |
|-------|-------------|
| **context** | Full ground-truth preference list provided (upper bound) |
| **nocontext** | No past context provided (lower bound) |
| **foundry** | Azure AI Foundry memory store |
| **foundry_local** | Local LanceDB replicating Foundry pipeline |
| **google** | Google Vertex AI Agent Engine memory |
| **aws** | AWS Bedrock AgentCore memory |

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright
```
