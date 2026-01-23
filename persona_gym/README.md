# PersonaMem Experimental Module

Experimental code for generating persona-grounded conversation data and evaluating Task-Oriented Dialogue (TOD) agents on memory and preference recall.

## Overview

This module provides an isolated sandbox for:
1. **Data Generation** - Creating synthetic user-assistant conversations grounded in persona histories
2. **TOD Task Generation** - Generating task-oriented dialogue tasks from conversation data
3. **Agent Evaluation** - Evaluating agents on their ability to recall and apply user preferences

## Directory Structure

```
experimental/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── sample_data_generation.py     # Main entry point for data generation
├── tod_task_generation.py        # Generate TOD tasks from conversations
├── tod_evaluation.py             # Evaluate agents on TOD tasks
├── tod_metric.py                 # Evaluation metrics and scoring
├── agent.py                      # Agent implementations (ContextAware, NoContext)
├── tool_simulator.py             # Simulated tool responses for evaluation
└── personamem_core/              # Core data generation library
    ├── __init__.py
    ├── schemas.py                # Pydantic models for structured data
    ├── prompts.py                # LLM prompt templates
    ├── prepare_data.py           # Data preparation and generation
    ├── query_llm.py              # LLM query interface
    └── utils.py                  # Utility functions
```

Note: Log files are written to `logs/` in the project root (outside the package).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Azure OpenAI Configuration

Set the following environment variables (or use a `.env` file):

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"           # Your deployment name
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

Authentication uses `DefaultAzureCredential` from `azure-identity`, so ensure you are logged in via:
- Azure CLI (`az login`)
- Visual Studio Code Azure extension
- Or have appropriate environment credentials configured

## Usage

### Generate Conversation Data

Generate synthetic conversations for a specific topic:

```bash
# Full generation (~15+ minutes) - recommended for actual data
python experimental/sample_data_generation.py --topic travel

# Quick mode (~2-3 minutes) - init + week steps, with preference updates
python experimental/sample_data_generation.py --topic travel --quick

# Debug mode (~1 minute) - minimal for testing core logic
python experimental/sample_data_generation.py --topic travel --debug

# With verbose output
python experimental/sample_data_generation.py --topic travel --verbose
```

**Run Modes:**
| Mode | Flag | Time | Use Case |
|------|------|------|----------|
| Full | (none) | ~15+ min | Production data with all history expansion + conversation reflection |
| Quick | `--quick` | ~2-3 min | Testing with preference evolution (init + week steps) |
| Debug | `--debug` | ~1 min | Verify core logic works (init steps only, no reflection) |

**Supported Topics:** travel, therapy, food, writing, email, coding, legal

**Output:** Files are saved to `data/output_sample/{topic}/`:
- `sample_conversation_{topic}_persona{N}_sample{M}[_quick|_debug]_artifacts.json` - Full artifacts
- `sample_conversation_{topic}_persona{N}_sample{M}[_quick|_debug]_conversation.json` - Processed conversation

### Generate TOD Tasks

Generate task-oriented dialogue tasks from conversation data:

```bash
python experimental/tod_task_generation.py \
    --input data/output_sample/travel/sample_conversation_travel_persona0_sample0_conversation.json \
    --output data/output_sample/travel/tod_tasks.jsonl
```

### Evaluate Agents

Run TOD evaluation with an agent:

```bash
python experimental/tod_evaluation.py \
    --tasks data/output_sample/travel/tod_tasks.jsonl \
    --context data/output_sample/travel/sample_conversation_travel_persona0_sample0_conversation.json \
    --output data/output_sample/travel/evaluation_results.json
```

## Core Components

### personamem_core/schemas.py

Pydantic models for structured outputs (OpenAI Structured Outputs compatible):

- **`SideNote`** - Metadata annotation linking user turns to persona facts
- **`ConversationTurn`** - Single turn with role, content, and optional side note
- **`GeneratedConversation`** - Complete multi-turn conversation (used as response_format schema)

### personamem_core/prompts.py

LLM prompt templates for:
- Persona expansion
- Conversation generation with structured JSON schema
- History generation

### personamem_core/query_llm.py

LLM query interface with:
- **`query_llm()`** - Standard LLM calls for text generation
- **`query_llm_structured()`** - Structured output calls with Pydantic schema validation

### Agent Types (agent.py)

- **`ContextAwareAgent`** - Has access to user's conversation history (upper bound)
- **`NoContextAgent`** - No access to past conversations (lower bound baseline)

### Evaluation Metrics (tod_metric.py)

- **Preference Recall** - Proactive vs. prompted usage of preferences
- **Turn Efficiency** - Penalizing unnecessary clarification/correction turns
- **Task Completion** - Overall success rate

## Key Features

### Structured Outputs (No Legacy Parsing)

The pipeline uses OpenAI Structured Outputs to guarantee valid conversation format:

```python
# All conversation generation uses structured outputs
conversation = LLM.query_llm_structured(
    prompt=prompt,
    response_schema=GeneratedConversation,  # Pydantic model
)
# Returns validated GeneratedConversation - no regex/JSON repair needed
```

Benefits:
- Type-safe output guaranteed by the API
- No `repair_json` or regex parsing for conversations
- Explicit role values (`"user"`, `"assistant"`) enforced by schema
- Side notes always present (null or object) - no guessing

### Tool Simulation

The `tool_simulator.py` generates realistic tool outputs without knowledge of user preferences, enabling proper evaluation of preference-aware filtering by agents.

## Output Format

### Conversation JSON

```json
{
  "turns": [
    {
      "role": "user",
      "content": "I need to book a flight to Paris.",
      "side_note": {
        "event": "Prefers window seats on flights",
        "date": "03/15/2024"
      }
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help you book a flight to Paris..."
    }
  ],
  "topic": "travel",
  "period": "INIT"
}
```

### TOD Task JSONL

```json
{
  "task_id": "uuid",
  "description": "Book a flight from NYC to Paris for April 15th",
  "tools": ["search_flights", "book_flight", "select_seat"],
  "preferences": [
    {"category": "flight", "value": "window seat", "source": "conversation"}
  ],
  "expected_behaviors": ["Should proactively select window seat"]
}
```

## Development Notes

- All logs are written to `logs/` in the project root
- The pipeline is designed to be modular - each component can be used independently
- Legacy format support (`["User: content", "Assistant: response"]`) is maintained for compatibility

## Troubleshooting

### Azure Content Filter Errors

If you see `content_filter` errors in logs, the retry logic will handle them automatically. If generation fails completely, check:
1. Your deployment's content filter settings
2. The input persona for potentially problematic content

### Missing Assistant Turns

With structured outputs, the `GeneratedConversation` schema validates the conversation at generation time. If you still see issues:
1. Ensure you're using `query_llm_structured()` with the `GeneratedConversation` schema
2. Check that prompts include role alternation instructions
