# BYOM-Bench: A Benchmark for Evaluating LLM Personalization Memory

## Motivation

Large language models are increasingly deployed as persistent personal assistants, yet there is no standard benchmark for measuring how well they remember and apply user preferences over time. Real users accumulate preferences across dozens of conversations — dietary restrictions, work-hour constraints, tool choices — and expect their assistant to recall these without being reminded. Preferences also *evolve*: a job change shifts someone's schedule, a health diagnosis alters their diet. An effective memory system must not only recall current preferences but also avoid applying stale ones that have been superseded.

BYOM-Bench addresses this gap. It is a fully synthetic, end-to-end benchmark that generates realistic multi-session conversation histories with evolving user preferences, creates evaluation tasks that demand proactive preference application, and scores agents on **preference recall** — did the agent proactively remember and correctly apply the user's current preferences?

## Pipeline Overview

The benchmark is a three-stage pipeline: **Data Generation** produces synthetic user histories, **Task Generation** creates evaluation scenarios from those histories, and **Evaluation** runs an agent against a simulated user and scores the resulting dialogue. Each stage is fully automated and driven by LLM calls.

### Stage 1: Data Generation

The pipeline begins with a short base persona description (e.g., "a mid-career backend developer at a fintech startup"). An LLM expands this into a detailed persona with demographics, life facts, and **25 baseline preferences** spanning five lifestyle domains.

Next, a sequence of life events is generated — significant changes such as a career shift, a move to a new city, or a health diagnosis. Each life event triggers preference evolution: some existing preferences are superseded by new ones (e.g., "morning espresso" becomes "eliminates caffeine after noon"), while others are newly created. A **PreferenceTimeline** tracks which preferences are active versus stale at each point in time.

Finally, 10 multi-turn conversation sessions are generated chronologically. Each session is grounded in a life event, and preferences are revealed naturally through dialogue — the user mentions their dietary needs when discussing meal planning, or their schedule constraints when booking meetings. The output is a single `sessions.json` file containing the full conversation history, the expanded persona, and the preference timeline.

### Stage 2: Task Generation

Task generation takes the completed session data and produces evaluation scenarios. Each task selects **6 required preferences** from the timeline: 2 evolved preferences (ones that superseded earlier versions) and 4 baseline preferences. Including evolved preferences ensures the benchmark tests not just recall but also the ability to distinguish current from outdated information.

An LLM generates a realistic evaluation event — a complex request such as planning a week-long meal plan or organizing a team offsite — that naturally requires applying all 6 selected preferences. Crucially, the **user prompt** is preference-neutral: it describes the task without mentioning any specific preferences (e.g., "I need help with meal planning for next week"). An ideal agent must proactively apply remembered preferences without being told. A rubric is produced alongside each task, mapping each required preference to its ID, description, and (if applicable) the stale preference it supersedes.

### Stage 3: Evaluation

Evaluation runs a dialogue between the agent under test and an LLM-powered **user simulator**. The agent is initialized with its memory system (or, for baselines, with full/no context). The user simulator sends the preference-neutral opening message, then responds naturally to the agent's suggestions. It maintains a scratchpad tracking which required preferences remain uncovered, and will gradually reveal them if the agent fails to apply them proactively — simulating a real user who eventually fills in the gaps when the assistant misses the mark.

The dialogue runs for up to 10 agent turns, after which the conversation is passed to a **Preference Judge**:

**Preference Judge.** For each of the 6 required preferences, the judge performs first-mention analysis: did the agent apply the preference before the user mentioned it (**PROACTIVE**, +1), or did the user have to reveal it first (**IGNORED**, 0)? If the agent applied a superseded version of an evolved preference, it is marked **STALE**, -1). Two metrics are derived:

**Preference Recall** measures proactive preference application:

$$\text{preference\_score} = \max\!\left(0,\;\frac{\text{proactive} - \text{stale}}{\text{total\_required}}\right)$$

**Stale Recall Rate** measures how often the agent applies outdated preferences that have been superseded:

$$\text{stale\_recall\_rate} = \frac{\text{stale\_count}}{\text{total\_stale\_preferences}}$$

A well-performing agent should maximize preference recall while minimizing stale recall rate — high recall alone is insufficient if the agent cannot distinguish current preferences from superseded ones.

### Agent Types and Baselines

BYOM-Bench evaluates agents along a spectrum. Two deterministic baselines bracket the range: a **context** agent that receives the ground-truth list of current preferences (upper bound) and a **nocontext** agent with no past information (lower bound). Between these sit the memory-backed agents — **Azure AI Foundry**, **Google Vertex AI Agent Engine**, **AWS Bedrock AgentCore**, **Mem0**, **Zep/Graphiti** — each using its own retrieval and storage strategy. All memory agents share a common tool-calling loop where the agent can invoke a memory-search tool during conversation, and retrieved memories are injected into subsequent turns.

### Scoring and Aggregation

Scores are macro-averaged hierarchically: per-run scores are averaged within a task, per-task scores within a session (persona), and per-session scores across the full benchmark. This ensures each persona receives equal weight regardless of the number of evaluation runs. 
