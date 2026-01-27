# PersonaGym

A benchmark for evaluating LLM personalization through multi-turn conversations with embedded user preferences.

## Overview

PersonaGym measures how well AI agents remember and proactively use user preferences without requiring users to repeat themselves.

**Three-Stage Pipeline:**
1. **Data Generation** - Create synthetic conversations from personas with embedded preferences
2. **Task Generation** - Generate task-oriented dialogue (TOD) tasks that test preference recall
3. **Evaluation** - Score agents on preference recall, turn efficiency, and task completion

## Directory Structure

```
persona_gym/
├── __init__.py                   # Package entry
├── schemas.py                    # All data models + pipeline contracts
├── client.py                     # Shared LLMClient (Azure OpenAI Responses API)
├── task_generator.py             # Generate TOD tasks from conversations
├── evaluation.py                 # Evaluate agents on TOD tasks
├── metric.py                     # Scoring and metrics
├── agent.py                      # Agent implementations (ContextAware, NoContext)
├── tool_simulator.py             # Simulated tool responses for evaluation
├── prompts/                      # Centralized prompt management (YAML files)
│   ├── __init__.py               # render_prompt(), list_prompts()
│   ├── data_generation/          # Data generation prompts
│   ├── task_generation/          # TOD task generation prompts
│   └── evaluation/               # Judge and user simulator prompts
├── data_generators/              # Pluggable data generation strategies
│   ├── base.py                   # BaseDataGenerator ABC
│   ├── personamem_v2.py          # Token-budgeted generation with preference evolution
│   └── multisession.py           # Multi-session generation with life events
└── evaluation_multisession/      # Multi-session evaluation system
    ├── orchestrator.py           # Orchestrates evaluation task generation
    ├── judge.py                  # LLM judge for scoring dialogues
    ├── user_simulator.py         # Simulates user responses
    └── agents.py                 # FullContextAgent, NoContextAgent
```

Note: Log files are written to `logs/` in the project root.

## Usage

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

### Generate TOD Tasks

```python
from persona_gym.task_generator import generate_tasks_from_data

task_output = generate_tasks_from_data(data_output, num_tasks=3)
```

Or via CLI:
```bash
python -m persona_gym.task_generator \
    --input outputs/travel/conversation_artifacts.json \
    --output outputs/travel/tasks.jsonl
```

### Evaluate Agents

```python
from persona_gym.evaluation import evaluate_from_tasks

eval_output = evaluate_from_tasks(task_output, agent_type="context")
print(f"Final Score: {eval_output.aggregate.average_final_score:.2f}")
```

Or via CLI:
```bash
python -m persona_gym.evaluation \
    --tasks outputs/travel/tasks.jsonl \
    --context outputs/travel/conversation.json \
    --agent context  # or no_context
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Preference Recall | Proactive vs. prompted usage of preferences |
| Turn Efficiency | Penalizes clarifying questions and corrections |
| Task Completion | Binary success indicator |
| Final Score | Weighted combination of metrics |

## Agent Types

- **ContextAwareAgent** - Has access to user's conversation history (upper bound)
- **NoContextAgent** - No access to past conversations (baseline)

## Configuration

Set environment variables (or use `.env` file):

```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="gpt-4"
AZURE_OPENAI_API_VERSION="2025-03-01-preview"
```

Authentication uses `DefaultAzureCredential` from `azure-identity`.
