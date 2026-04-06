# REAM-Bench Pipeline Overview

End-to-end synthetic benchmark for evaluating agentic memory in LLMs.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REAM-Bench Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │ 1. DATA GEN      │ →  │ 2. TASK GEN      │ →  │ 3. EVALUATION    │      │
│  │                  │    │                  │    │                  │      │
│  │ • Expand persona │    │ • Select 6 prefs │    │ • Run dialogue   │      │
│  │ • Life events    │    │ • Create event   │    │ • Preference     │      │
│  │ • 25 preferences │    │ • User prompt    │    │   judge          │      │
│  │ • 10 sessions    │    │ (50% evolved)    │    │ • Efficiency     │      │
│  │ • Conversations  │    │                  │    │   judge          │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│         ↓                        ↓                       ↓                 │
│   sessions.json            task_XX.json          eval_XX_agent.json        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Data Generation

**Module:** `ream_bench/data_generators/multisession.py`

Generates multi-session conversation data with evolving preferences.

### Flow

```
Input: Base persona description
         ↓
┌─────────────────────────────────────┐
│ 1. Expand Persona                   │
│    - Name, demographics, traits     │
│    - 25 baseline preferences        │
│      (5 per domain)                 │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 2. Generate Life Events             │
│    - Timeline of significant events │
│    - Each event evolves preferences │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3. Generate Sessions (10 total)     │
│    - Natural conversations          │
│    - Preferences revealed gradually │
│    - Some preferences superseded    │
└─────────────────────────────────────┘
         ↓
Output: sessions.json
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Baseline Preferences** | 25 initial preferences across 5 domains |
| **Life Events** | Significant events that cause preference changes |
| **Evolved Preferences** | New preferences that supersede old ones |
| **PreferenceTimeline** | Tracks active vs stale preferences over time |

### Output Structure

```json
{
  "persona": "Expanded persona description...",
  "sessions": [
    {
      "session_id": 0,
      "date": "01/15/2024",
      "life_event": "Started new job with early meetings",
      "conversation": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
  ],
  "timeline": {
    "preferences": [...],
    "active_preference_ids": [...],
    "superseded_map": {"pref_047": "pref_009"}
  }
}
```

---

## Stage 2: Task Generation

**Module:** `ream_bench/task_generators/evaluation_task.py`

Creates evaluation tasks that require proactive preference application.

### Flow

```
Input: sessions.json
         ↓
┌─────────────────────────────────────┐
│ 1. Select 6 Required Preferences    │
│    - 50% must be evolved (recent)   │
│    - Coherent theme for task        │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 2. Generate Evaluation Event        │
│    - Complex task description       │
│    - Requires multiple preferences  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3. Generate User Prompt             │
│    - Preference-neutral request     │
│    - Does not mention preferences   │
└─────────────────────────────────────┘
         ↓
Output: task_XX.json
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Required Preferences** | 6 preferences agent must apply proactively |
| **Evolved Preference** | Preference that supersedes an older one |
| **User Prompt** | Initial message that doesn't reveal preferences |
| **Task Internal** | Detailed breakdown of what agent should do |

### Output Structure

```json
{
  "task_id": "eval_abc123",
  "evaluation_event": {
    "event": "Help design a week-long meal plan...",
    "user_prompt": "I need help with meal planning...",
    "task_internal": "1. Apply Mediterranean diet... 2. Avoid caffeine..."
  },
  "rubric": {
    "required_preferences": [
      {
        "id": "pref_006",
        "fact": "Follows Mediterranean diet"
      },
      {
        "id": "pref_047",
        "fact": "Eliminates caffeine after noon",
        "supersedes": {
          "id": "pref_009",
          "fact": "Limits caffeine to morning espresso"
        }
      }
    ]
  }
}
```

---

## Stage 3: Evaluation

**Module:** `ream_bench/evaluation_multisession/`

Runs agent dialogue and scores with LLM judges.

### Flow

```
Input: sessions.json + task_XX.json
         ↓
┌─────────────────────────────────────┐
│ 1. Initialize Agent                 │
│    - ContextAwareAgent (full hist)  │
│    - NoContextAgent (baseline)      │
│    - FoundryMemoryAgent (memory)    │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 2. Run Dialogue Loop                │
│    - User simulator sends messages  │
│    - Agent responds                 │
│    - Max 10 agent turns             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3. Preference Judge                 │
│    - First-mention analysis         │
│    - Who mentioned preference first?│
│    - Agent (PROACTIVE) or User?     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 4. Efficiency Judge                 │
│    - Classify each agent turn       │
│    - PRODUCTIVE, IGNORED, etc.      │
└─────────────────────────────────────┘
         ↓
Output: eval_XX_agent.json
```

### Agent Types

| Agent | Context | Expected Scores |
|-------|---------|-----------------|
| **ContextAwareAgent** | Full conversation history | preference ~0.8-1.0 |
| **NoContextAgent** | None (baseline) | preference ~0.0, efficiency ~1.0 |
| **FoundryMemoryAgent** | Azure AI Foundry memory store | Varies |

### Scoring

#### Preference Score

Measures proactive preference recall:

```
preference_score = max(0, (proactive_count - stale_count) / total_required)
```

| Classification | Description | Impact |
|----------------|-------------|--------|
| **PROACTIVE** | Agent mentioned preference before user | +1 |
| **IGNORED** | User mentioned preference first | 0 |
| **STALE** | Agent used superseded preference | -1 |

#### Efficiency Score

Measures conversation efficiency:

```
efficiency_score = max(0, (agent_turns - 0.5×clarifying - corrections) / agent_turns)
```

| Turn Type | Description | Penalty |
|-----------|-------------|---------|
| **PRODUCTIVE** | Useful, preference-aware response | 0 |
| **IGNORED** | User revealed preference agent missed | 0 |
| **CLARIFYING** | Asked about known preference | 0.5 |
| **CORRECTION** | Wrong suggestion, user corrected | 1.0 |

### Output Structure

```json
{
  "task_id": "eval_abc123",
  "scores": {
    "preference_score": 0.83,
    "efficiency_score": 0.90
  },
  "conversation": [...],
  "preference_scoring": {
    "proactive_count": 5,
    "stale_count": 0,
    "preference_verdicts": [...]
  },
  "efficiency_scoring": {
    "total_turns": 10,
    "productive_turns": 7,
    "ignored_turns": 3,
    "turn_classifications": [...]
  },
  "rubric": {
    "required_preferences": [...]
  }
}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `ream_bench/schemas.py` | All data models (MultiSessionOutput, EvaluationTask, etc.) |
| `ream_bench/client.py` | Azure OpenAI client with retry logic |
| `ream_bench/data_generators/multisession.py` | Stage 1: MultiSessionGenerator |
| `ream_bench/task_generators/evaluation_task.py` | Stage 2: EvaluationTaskGenerator |
| `ream_bench/evaluation_multisession/runner.py` | Stage 3: Evaluation orchestration |
| `ream_bench/evaluation_multisession/judge.py` | Preference + efficiency judges |
| `ream_bench/evaluation_multisession/user_simulator.py` | Simulated user for dialogue |
| `ream_bench/agents/base.py` | ContextAwareAgent, NoContextAgent |
| `ream_bench/agents/foundry_agent.py` | FoundryMemoryAgent (Azure AI Foundry) |

---

## Running the Pipeline

```bash
# Stage 1: Generate session data
uv run python test_data_generation.py

# Stage 2: Generate evaluation tasks
uv run python test_task_generation.py --count 3

# Stage 3: Run evaluations
uv run python test_evaluation.py --agent context
uv run python test_evaluation.py --agent nocontext
uv run python test_evaluation.py --agent foundry --no-cache
```

---

## Output Directory Structure

```
outputs/
  2026-02-03_1430/
    sessions.json           # Stage 1 output
    tasks/
      task_01.json          # Stage 2 output
      task_02.json
      task_03.json
    evaluation/
      eval_01_context.json  # Stage 3 output
      eval_01_nocontext.json
      eval_01_foundry.json
```
