# PersonaGym Pipeline Overview

This document provides a comprehensive overview of the PersonaGym pipeline, covering data generation, task generation, and evaluation.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    PersonaGym Pipeline                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ╔═══════════════════════════════╗                                                                  │
│  ║     INPUT: Persona Hub        ║                                                                  │
│  ║   (200K persona descriptions) ║                                                                  │
│  ╚═══════════════╤═══════════════╝                                                                  │
│                  │                                                                                  │
│                  ▼                                                                                  │
│  ┌───────────────────────────────┐     ┌───────────────────────────────┐     ┌─────────────────────┐│
│  │    1. DATA GENERATION         │     │    2. TASK GENERATION         │     │   3. EVALUATION     ││
│  │    ════════════════════       │     │    ═══════════════════        │     │   ════════════      ││
│  │                               │     │                               │     │                     ││
│  │  ┌─────────────────────────┐  │     │  ┌─────────────────────────┐  │     │  ┌───────────────┐  ││
│  │  │ 1. Expand Persona       │  │     │  │ 1. Extract Preferences  │  │     │  │ Agent Types:  │  ││
│  │  │    (name, traits, bio)  │  │     │  │    from Side Notes      │  │     │  │ • ContextAware│  ││
│  │  ├─────────────────────────┤  │     │  ├─────────────────────────┤  │     │  │ • NoContext   │  ││
│  │  │ 2. Generate History     │  │     │  │ 2. Select Task Template │  │     │  │ • (Future:    │  ││
│  │  │    (events over years)  │  │     │  │    (flight, hotel, etc) │  │     │  │   MemoryAgent)│  ││
│  │  ├─────────────────────────┤  │     │  ├─────────────────────────┤  │     │  └───────────────┘  ││
│  │  │ 3. Create Conversation  │  │     │  │ 3. Map Preferences to   │  │     │         │           ││
│  │  │    (User ↔ Assistant)   │  │     │  │    Task Requirements    │  │     │         ▼           ││
│  │  ├─────────────────────────┤  │     │  ├─────────────────────────┤  │     │  ┌───────────────┐  ││
│  │  │ 4. Reflect & Expand     │  │     │  │ 4. Generate "Trap"      │  │     │  │ Dialogue Loop │  ││
│  │  │    (improve quality)    │  │     │  │    Results (test memory)│  │     │  │ Agent ↔ User  │  ││
│  │  ├─────────────────────────┤  │     │  ├─────────────────────────┤  │     │  │ + Tool Calls  │  ││
│  │  │ 5. Parse & Validate     │  │     │  │ 5. Define Expected      │  │     │  └───────────────┘  ││
│  │  │    (enforce alternation)│  │     │  │    Agent Behaviors      │  │     │         │           ││
│  │  └─────────────────────────┘  │     │  └─────────────────────────┘  │     │         ▼           ││
│  │               │               │     │               │               │     │  ┌───────────────┐  ││
│  │               ▼               │     │               ▼               │     │  │ LLM-as-Judge  │  ││
│  │  ┌─────────────────────────┐  │     │  ┌─────────────────────────┐  │     │  │ Turn Classify │  ││
│  │  │ OUTPUT:                 │  │     │  │ OUTPUT:                 │  │     │  └───────────────┘  ││
│  │  │ • Conversation JSON     │  │     │  │ • TOD Tasks (JSONL)     │  │     │         │           ││
│  │  │   (turns + side_notes)  │  │     │  │   - task description    │  │     │         ▼           ││
│  │  │ • Artifacts JSON        │  │     │  │   - tool schemas        │  │     │  ┌───────────────┐  ││
│  │  │   (persona, history,    │  │     │  │   - relevant prefs      │  │     │  │ METRICS:      │  ││
│  │  │    preferences)         │  │     │  │                         │  │     │  │ • Pref Recall │  ││
│  │  └─────────────────────────┘  │     │  └─────────────────────────┘  │     │  │ • Turn Effic. │  ││
│  │                               │     │                               │     │  │ • Task Compl. │  ││
│  └───────────────┬───────────────┘     └───────────────┬───────────────┘     │  └───────────────┘  ││
│                  │                                     │                     │                     ││
│                  │         ┌───────────────────────────┘                     └──────────┬──────────┘│
│                  │         │                                                            │           │
│                  ▼         ▼                                                            ▼           │
│  ╔═══════════════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  DATA FLOW:  Persona ──▶ Conversation + Preferences ──▶ TOD Tasks ──▶ Agent Evaluation Scores ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Data Generation — Creating Realistic User Profiles & Conversations

**Goal:** Generate synthetic but realistic user-assistant conversations where users naturally reveal their preferences, habits, and life events over time.

| Step | What Happens | Output |
|------|--------------|--------|
| **1.1 Load Persona** | Sample a base persona from Persona Hub (200K personas) | Raw persona description |
| **1.2 Expand Persona** | LLM enriches persona with name, demographics, personality traits | Detailed persona profile |
| **1.3 Generate Personal History** | Create timeline of life events, preferences, likes/dislikes spanning years | Dated events (Long-term & Short-term) |
| **1.4 Elaborate Topic** | Define what aspects of the topic (e.g., travel) the user might discuss | Topic elaboration |
| **1.5 Generate Conversation** | Create multi-turn dialogue where user naturally mentions their history | Raw conversation with Side Notes |
| **1.6 Reflect & Expand** | Improve conversation quality, ensure all events are covered, expand sections | Refined conversation |
| **1.7 Parse & Validate** | Convert to structured format, enforce User↔Assistant alternation | Final conversation JSON |

**Key Artifacts:**
- **Personal History Events:** Dated entries like "08/17/1954: Launched local radio show..." with Long-Term/Short-Term categories
- **Preferences:** Likes (window seats, vegetarian food), Dislikes (cruises, adventure sports), Habits (planned travel, writing journals)
- **Side Notes:** Annotations linking each user turn to the original persona fact being revealed

---

### Stage 2: Task Generation — Creating Evaluation Tasks from User Profiles

**Goal:** Generate realistic Task-Oriented Dialogue (TOD) tasks that test whether an agent can recall and apply the user's preferences from their conversation history.

**Inputs:**
- Processed conversation with embedded preferences (from Stage 1)
- Topic-specific task templates (flight booking, hotel search, etc.)

| Step | What Happens | Output |
|------|--------------|--------|
| **2.1 Extract Preferences** | Parse Side Notes to identify all user preferences from conversation | List of preferences with categories |
| **2.2 Select Task Template** | Choose appropriate task type for the topic (e.g., "Book a flight") | Task template with tool schemas |
| **2.3 Map Relevant Preferences** | Identify which preferences apply to this specific task | Relevant preference list |
| **2.4 Define Expected Behaviors** | Specify how agent should handle each preference (proactive use) | Expected usage patterns |
| **2.5 Generate Task & Tools** | Create task description and OpenAI-style tool schemas | Complete TOD task definition |

**Key Outputs:**
- **Task Description:** "Book a flight from NYC to Paris for April 15th"
- **Tool Schemas:** OpenAI-style function definitions (search_flights, book_flight, select_seat)
- **Relevant Preferences:** User prefers window seats, avoids red-eye flights, likes specific airlines
- **Expected Behaviors:** "Should proactively request window seat", "Should avoid suggesting red-eye flights"

---

### Stage 3: Evaluation — Measuring Agent Memory & Preference Recall

**Goal:** Assess how well an agent remembers and proactively applies user preferences during task completion, without requiring the user to repeat themselves.

**Inputs:**
- TOD Tasks (from Stage 2)
- Conversation History (context for ContextAwareAgent)
- Agent to evaluate (ContextAware vs NoContext baseline)

| What is Evaluated | Metric | What It Measures |
|-------------------|--------|------------------|
| **Preference Usage** | **Preference Recall** | Did the agent proactively use known preferences without being reminded? |
| **Dialogue Efficiency** | **Turn Efficiency** | Did the agent avoid unnecessary clarification questions about stated preferences? |
| **Task Success** | **Task Completion** | Did the agent successfully complete the requested task? |
| **Error Handling** | **Correction Penalty** | How often did the user need to correct the agent's mistakes? |

**Metric Details:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Preference Recall** | `(# proactively used) / (# relevant preferences)` | 1.0 = perfect memory, 0.0 = forgot everything |
| **Turn Efficiency** | `1 - (penalty_turns / total_turns)` | 1.0 = no wasted turns, lower = inefficient dialogue |
| **Task Completion** | Binary (0 or 1) | Did the task get completed successfully? |
| **Overall Score** | `0.5×recall + 0.3×efficiency + 0.2×completion` | Weighted combination favoring memory |

**Turn Classification (LLM-as-Judge):**
- ✅ **Productive (+1.0):** Agent advances task, correctly applies preferences
- ⚠️ **Justified Clarification (+0.5):** Agent asks about genuinely ambiguous/conflicting preferences
- ❌ **Unnecessary Clarification (-0.25):** Agent asks about clearly stated preference (forgot it)
- ❌ **Correction (-0.5):** User had to remind agent of forgotten preference
- ❌ **Repeated Correction (-1.0):** Agent ignores correction or makes same mistake twice

---

---

## Part 1: Data Generation

**Entry Point:** `data_generators/`  
**Core Modules:** `personamem_v2.py`, `multisession.py`

The data generation pipeline creates synthetic user-assistant conversations grounded in persona histories. The goal is to produce conversations where users naturally reveal their preferences over time.

### Data Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA GENERATION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD & EXPAND PERSONA                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Persona_Hub_200000.jsonl                                                  │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ Raw Persona:                                      │                     │
│   │ "A charismatic talk show host who advocates       │                     │
│   │  for natural remedies..."                         │                     │
│   └──────────────────────────────────────────────────┘                     │
│          │                                                                  │
│          │  LLM: expand_persona                                             │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ Expanded Persona:                                 │                     │
│   │ "Name: Carlos Rivera, Hispanic male, born 1936,   │                     │
│   │  known for warm presence, values traditional      │                     │
│   │  wisdom and natural remedies..."                  │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ELABORATE TOPIC                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Topic: "travel"                                                           │
│          │                                                                  │
│          │  LLM: elaborate_topic                                            │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ Topic Elaboration:                                │                     │
│   │ - Favorite destinations                           │                     │
│   │ - Travel experiences & stories                    │                     │
│   │ - Travel tips & planning                          │                     │
│   │ - Food and culture                                │                     │
│   │ - Personal growth through travel                  │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: GENERATE PERSONAL HISTORY (Longitudinal)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │    INIT     │──▶│   +WEEK     │──▶│   +MONTH    │──▶│   +YEAR     │    │
│   │  (10 events)│   │  (expand)   │   │  (expand)   │   │  (expand)   │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ Personal History Events:                          │                     │
│   │ {                                                 │                     │
│   │   "08/17/1954": {                                 │                     │
│   │     "Event": "Launched local radio show...",      │                     │
│   │     "Category": "Long-Term"                       │                     │
│   │   },                                              │                     │
│   │   "09/02/1954": {                                 │                     │
│   │     "Event": "Prepared herbal tea at studio...",  │                     │
│   │     "Category": "Short-Term"                      │                     │
│   │   },                                              │                     │
│   │   ...                                             │                     │
│   │ }                                                 │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GENERATE CONVERSATION (Structured Output)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Personal History + Topic                                                  │
│          │                                                                  │
│          │  LLM.query_llm_structured(response_schema=GeneratedConversation) │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ GeneratedConversation (Pydantic):                 │                     │
│   │ {                                                 │                     │
│   │   "turns": [                                      │                     │
│   │     {                                             │                     │
│   │       "role": "user",                             │                     │
│   │       "content": "I've been thinking...",         │                     │
│   │       "side_note": {                              │                     │
│   │         "event": "Launched radio show",           │                     │
│   │         "date": "08/17/1954"                      │                     │
│   │       }                                           │                     │
│   │     },                                            │                     │
│   │     {                                             │                     │
│   │       "role": "assistant",                        │                     │
│   │       "content": "That sounds wonderful!",        │                     │
│   │       "side_note": null                           │                     │
│   │     }                                             │                     │
│   │   ],                                              │                     │
│   │   "topic": "travel",                              │                     │
│   │   "period": "INIT"                                │                     │
│   │ }                                                 │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
│   ✓ Guaranteed valid JSON (API enforced)                                    │
│   ✓ Type-safe: role is exactly "user" or "assistant"                        │
│   ✓ side_note always present (null or object) - no guessing                 │
│   ✓ No regex parsing or json_repair needed                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT FILES                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   data/output_sample/{topic}/                                               │
│   ├── sample_conversation_{topic}_persona{N}_sample{M}_artifacts.json       │
│   │   └── Full artifacts (persona, raw conversation, preferences)           │
│   │                                                                         │
│   └── sample_conversation_{topic}_persona{N}_sample{M}_conversation.json    │
│       └── Processed conversation (ready for TOD task generation)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|

---

## Part 2: Task Generation

**Entry Point:** `tod_task_generation.py`

The task generation pipeline creates Task-Oriented Dialogue (TOD) evaluation tasks from the generated conversation data.

### Task Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TASK GENERATION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD CONVERSATION DATA                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   sample_conversation_{topic}_conversation.json                             │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ Conversation with embedded preferences:           │                     │
│   │ - User mentions window seat preference            │                     │
│   │ - User reveals vegetarian dietary needs           │                     │
│   │ - User shares fear of flying over water           │                     │
│   │ - User prefers boutique hotels                    │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: EXTRACT PREFERENCES                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Parse side_notes from conversation turns                                  │
│          │                                                                  │
│          │  LLM: extract_preferences                                        │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ Extracted Preferences:                            │                     │
│   │ [                                                 │                     │
│   │   {                                               │                     │
│   │     "category": "flight",                         │                     │
│   │     "preference": "Always prefers window seat",   │                     │
│   │     "source_turn": 12,                            │                     │
│   │     "strength": "strong"                          │                     │
│   │   },                                              │                     │
│   │   {                                               │                     │
│   │     "category": "food",                           │                     │
│   │     "preference": "Vegetarian, avoids meat",      │                     │
│   │     "source_turn": 24,                            │                     │
│   │     "strength": "strict"                          │                     │
│   │   }                                               │                     │
│   │ ]                                                 │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: SELECT TASK TEMPLATE                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Topic-specific templates:                                                 │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  TRAVEL TEMPLATES:                                               │      │
│   │  • Book a flight from {origin} to {destination}                  │      │
│   │  • Find and book a hotel in {location}                           │      │
│   │  • Plan a trip itinerary for {duration}                          │      │
│   │  • Reserve a rental car                                          │      │
│   │                                                                  │      │
│   │  THERAPY TEMPLATES:                                              │      │
│   │  • Schedule a therapy session                                    │      │
│   │  • Find a support group                                          │      │
│   │  • Set up self-care reminders                                    │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GENERATE TOD TASK                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Template + Preferences + LLM                                              │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────┐                     │
│   │ TOD Task:                                         │                     │
│   │ {                                                 │                     │
│   │   "task_id": "uuid-1234",                         │                     │
│   │   "description": "Book a flight from NYC to       │                     │
│   │                   Paris for April 15th",          │                     │
│   │                                                   │                     │
│   │   "tools": [                                      │                     │
│   │     {                                             │                     │
│   │       "name": "search_flights",                   │                     │
│   │       "schema": {...}                             │                     │
│   │     },                                            │                     │
│   │     {                                             │                     │
│   │       "name": "book_flight",                      │                     │
│   │       "schema": {...}                             │                     │
│   │     }                                             │                     │
│   │   ],                                              │                     │
│   │                                                   │                     │
│   │   "relevant_preferences": [                       │                     │
│   │     {                                             │                     │
│   │       "item": "window seat",                      │                     │
│   │       "expected_usage": "proactive"               │                     │
│   │     }                                             │                     │
│   │   ],                                              │                     │
│   │                                                   │                     │
│   │   "expected_behaviors": [                         │  ◀── How agent      │
│   │     "Should proactively request window seat",     │      should act     │
│   │     "Should avoid suggesting red-eye flights",    │                     │
│   │     "Should complete task efficiently"            │                     │
│   │   ]                                               │                     │
│   │ }                                                 │                     │
│   └──────────────────────────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: tod_tasks.jsonl                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   One task per line (JSONL format):                                         │
│   {"task_id": "...", "description": "Book flight...", ...}                  │
│   {"task_id": "...", "description": "Find hotel...", ...}                   │
│   {"task_id": "...", "description": "Reserve car...", ...}                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Relevant Preferences** | Preferences from conversation history that apply to this task |
| **Expected Usage** | How the agent should use the preference (proactive vs. prompted) |
| **Expected Behaviors** | Specific actions the agent should take based on user preferences |
| **Tool Schemas** | OpenAI-style function schemas for each tool |

---

## Part 3: Evaluation

**Entry Points:** `tod_evaluation.py`, `agent.py`, `tool_simulator.py`, `tod_metric.py`

The evaluation pipeline assesses how well agents recall and apply user preferences during task-oriented dialogues.

### Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD INPUTS                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐         ┌─────────────────┐                          │
│   │  TOD Tasks      │         │  Conversation   │                          │
│   │  (tod_tasks.    │         │  History        │                          │
│   │   jsonl)        │         │  (context.json) │                          │
│   └────────┬────────┘         └────────┬────────┘                          │
│            │                           │                                    │
│            └───────────┬───────────────┘                                    │
│                        ▼                                                    │
│            ┌─────────────────────┐                                          │
│            │   Evaluation        │                                          │
│            │   Orchestrator      │                                          │
│            └─────────────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: INITIALIZE AGENT                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                        AGENT TYPES                               │      │
│   │                                                                  │      │
│   │  ┌────────────────────────┐    ┌────────────────────────┐       │      │
│   │  │   ContextAwareAgent    │    │    NoContextAgent      │       │      │
│   │  │   (Upper Bound)        │    │    (Lower Bound)       │       │      │
│   │  ├────────────────────────┤    ├────────────────────────┤       │      │
│   │  │ • Has full access to   │    │ • No access to past    │       │      │
│   │  │   conversation history │    │   conversations        │       │      │
│   │  │ • Can retrieve all     │    │ • Must rely on user    │       │      │
│   │  │   user preferences     │    │   explicitly stating   │       │      │
│   │  │ • Expected to apply    │    │   preferences          │       │      │
│   │  │   preferences          │    │ • Baseline for         │       │      │
│   │  │   proactively          │    │   comparison           │       │      │
│   │  └────────────────────────┘    └────────────────────────┘       │      │
│   │                                                                  │      │
│   │  Future: MemoryAgent (agent with custom memory retrieval)        │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: RUN DIALOGUE SIMULATION                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each TOD task:                                                        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    DIALOGUE LOOP                                 │      │
│   │                                                                  │      │
│   │    ┌──────────┐      ┌──────────┐      ┌──────────────┐         │      │
│   │    │ Simulated│      │  Agent   │      │    Tool      │         │      │
│   │    │   User   │─────▶│ Response │─────▶│  Simulator   │         │      │
│   │    └──────────┘      └──────────┘      └──────────────┘         │      │
│   │         ▲                                     │                  │      │
│   │         │                                     │                  │      │
│   │         └─────────────────────────────────────┘                  │      │
│   │                                                                  │      │
│   │    Continue until task complete or max turns reached             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    TOOL SIMULATOR                                │      │
│   │                                                                  │      │
│   │  • Generates realistic tool results                              │      │
│   │  • NO knowledge of user preferences                              │      │
│   │  • Returns diverse options (some violate preferences)            │      │
│   │  • Tests agent's preference-aware filtering                      │      │
│   │                                                                  │      │
│   │  Example: search_flights returns mix of window/middle/aisle      │      │
│   │           Agent should filter for user's window preference       │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: CLASSIFY TURNS                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LLM Judge analyzes each dialogue turn:                                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      TURN TYPES                                  │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ PRODUCTIVE                                              │     │      │
│   │  │ Agent advances task, uses preferences correctly         │     │      │
│   │  │ Score: +1.0                                             │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ JUSTIFIED_CLARIFICATION                                 │     │      │
│   │  │ Agent asks about genuinely ambiguous/conflicting prefs  │     │      │
│   │  │ Score: +0.5                                             │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ UNNECESSARY_CLARIFICATION                               │     │      │
│   │  │ Agent asks about clearly stated preference              │     │      │
│   │  │ Score: -0.25                                            │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ CORRECTION                                              │     │      │
│   │  │ User had to remind agent of forgotten preference        │     │      │
│   │  │ Score: -0.5                                             │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ REPEATED_CORRECTION                                     │     │      │
│   │  │ Agent ignores correction or repeats same mistake        │     │      │
│   │  │ Score: -1.0                                             │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: CALCULATE METRICS                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      KEY METRICS                                 │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ PREFERENCE RECALL                                       │     │      │
│   │  │                                                         │     │      │
│   │  │ For each relevant preference:                           │     │      │
│   │  │ • PROACTIVE: Agent used it without prompting = 1.0      │     │      │
│   │  │ • IGNORED: Agent didn't use it / needed reminder = 0.0  │     │      │
│   │  │                                                         │     │      │
│   │  │ Score = (# PROACTIVE) / (# relevant preferences)        │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ TURN EFFICIENCY                                         │     │      │
│   │  │                                                         │     │      │
│   │  │ Penalizes unnecessary turns:                            │     │      │
│   │  │ Score = 1 - (penalty_turns / total_turns)               │     │      │
│   │  │                                                         │     │      │
│   │  │ Where penalty_turns = corrections + unnecessary_clarifs │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ TASK COMPLETION                                         │     │      │
│   │  │                                                         │     │      │
│   │  │ Binary: Did the agent complete the task?                │     │      │
│   │  │ • All required tool calls made                          │     │      │
│   │  │ • User satisfied (no abandonment)                       │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   │                                                                  │      │
│   │  ┌────────────────────────────────────────────────────────┐     │      │
│   │  │ OVERALL SCORE                                           │     │      │
│   │  │                                                         │     │      │
│   │  │ Weighted combination:                                   │     │      │
│   │  │ • preference_recall × 0.5                               │     │      │
│   │  │ • turn_efficiency × 0.3                                 │     │      │
│   │  │ • task_completion × 0.2                                 │     │      │
│   │  └────────────────────────────────────────────────────────┘     │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: evaluation_results.json                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   {                                                                         │
│     "agent_type": "ContextAwareAgent",                                      │
│     "tasks_evaluated": 10,                                                  │
│     "metrics": {                                                            │
│       "preference_recall": 0.85,                                            │
│       "turn_efficiency": 0.92,                                              │
│       "task_completion": 0.90,                                              │
│       "overall_score": 0.88                                                 │
│     },                                                                      │
│     "per_task_results": [...]                                               │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `tod_evaluation.py` | Main evaluation orchestrator |
| `agent.py` | Agent implementations (ContextAware, NoContext) |
| `tool_simulator.py` | Generates realistic tool responses |
| `tod_metric.py` | Metric definitions and scoring logic |

---

## End-to-End Example

```bash
# Step 1: Generate conversation data
python sample_data_generation.py --topic travel --verbose

# Step 2: Generate TOD tasks
python tod_task_generation.py \
    --input data/output_sample/travel/sample_conversation_travel_persona0_sample0_conversation.json \
    --output data/output_sample/travel/tod_tasks.jsonl

# Step 3: Run evaluation
python tod_evaluation.py \
    --tasks data/output_sample/travel/tod_tasks.jsonl \
    --context data/output_sample/travel/sample_conversation_travel_persona0_sample0_conversation.json \
    --agent context_aware \
    --output data/output_sample/travel/evaluation_results.json
```

---

## Design Principles

1. **Strict Alternation**: Every User turn MUST be followed by an Assistant turn
2. **Preference Grounding**: All preferences traced back to conversation history
3. **Fair Evaluation**: Tool simulator is preference-agnostic
4. **LLM-as-Judge**: Uses few-shot examples for calibrated turn classification
5. **Modular Design**: Each component can be used independently
