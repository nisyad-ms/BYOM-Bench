"""
Task-Oriented Dialogue (TOD) Evaluation Metrics for PersonaMem.

This module provides evaluation metrics for assessing an agent's ability to use
historical user preferences in task-oriented dialogues. Key metrics include:
- Preference recall (proactive vs. prompted usage)
- Turn efficiency (penalizing correction turns)
- Task completion success

The evaluation uses an LLM-as-judge approach with few-shot examples for calibration.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# =============================================================================
# Turn Classification Types
# =============================================================================


class TurnType(Enum):
    """Classification of dialogue turns based on preference handling."""
    PRODUCTIVE = "productive"  # Advances task, uses preferences correctly
    JUSTIFIED_CLARIFICATION = "justified_clarification"  # Asks about genuinely ambiguous/conflicting prefs
    UNNECESSARY_CLARIFICATION = "unnecessary_clarification"  # Asks about clearly stated preference
    CORRECTION = "correction"  # User had to remind agent of forgotten preference
    REPEATED_CORRECTION = "repeated_correction"  # Agent ignores correction or repeats same mistake


class PreferenceUsage(Enum):
    """How the agent handled a specific preference."""
    PROACTIVE = "proactive"  # Agent explicitly mentioned/used it without user prompting
    IGNORED = "ignored"  # Agent didn't use it OR only used after user reminded them
    NOT_APPLICABLE = "not_applicable"  # Preference wasn't relevant to this task


# =============================================================================
# Scoring Weights and Rubrics
# =============================================================================

PREFERENCE_USAGE_SCORES = {
    PreferenceUsage.PROACTIVE: 1.0,
    PreferenceUsage.IGNORED: 0.0,
    PreferenceUsage.NOT_APPLICABLE: None,  # Excluded from calculation
}

# Weights for final score computation
SCORE_WEIGHTS = {
    "task_success": 0.34,
    "preference_score": 0.33,
    "efficiency_score": 0.33,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PreferenceItem:
    """A user preference extracted from conversation history."""
    fact: str
    preference_type: str  # "current", "updated", "static"
    source_date: str
    old_value: Optional[str] = None  # For updated preferences
    reason_of_change: Optional[str] = None  # For updated preferences

    def to_dict(self) -> dict:
        result = {
            "fact": self.fact,
            "type": self.preference_type,
            "source_date": self.source_date,
        }
        if self.old_value:
            result["old_value"] = self.old_value
        if self.reason_of_change:
            result["reason_of_change"] = self.reason_of_change
        return result


@dataclass
class TODTask:
    """A task-oriented dialogue evaluation task."""
    task_id: str
    task_description: str
    topic: str
    relevant_preferences: list[PreferenceItem]
    expected_behaviors: list[str]
    tool_schemas: dict

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "topic": self.topic,
            "relevant_preferences": [p.to_dict() for p in self.relevant_preferences],
            "expected_behaviors": self.expected_behaviors,
            "tool_schemas": self.tool_schemas,
        }


@dataclass
class TurnAnalysis:
    """Analysis of a single dialogue turn."""
    turn_number: int
    speaker: str  # "user" or "agent"
    content: str
    turn_type: TurnType
    affected_preferences: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class TODEvaluationResult:
    """Complete evaluation result for a TOD session."""
    task_id: str
    task_completed: bool
    turn_classifications: list[TurnAnalysis]
    preference_usage: dict[str, PreferenceUsage]
    total_turns: int = 0
    correction_turns: int = 0
    repeated_correction_turns: int = 0
    efficiency_score: float = 0.0
    preference_score: float = 0.0
    task_success_score: float = 0.0
    final_score: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_completed": self.task_completed,
            "turn_classifications": [
                {
                    "turn_number": t.turn_number,
                    "speaker": t.speaker,
                    "turn_type": t.turn_type.value,
                    "affected_preferences": t.affected_preferences,
                    "reasoning": t.reasoning,
                }
                for t in self.turn_classifications
            ],
            "preference_usage": {k: v.value for k, v in self.preference_usage.items()},
            "scores": {
                "total_turns": self.total_turns,
                "correction_turns": self.correction_turns,
                "repeated_correction_turns": self.repeated_correction_turns,
                "efficiency_score": self.efficiency_score,
                "preference_score": self.preference_score,
                "task_success_score": self.task_success_score,
                "final_score": self.final_score,
            },
            "reasoning": self.reasoning,
        }


# =============================================================================
# Few-Shot Examples for Judge Calibration
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # ==========================================================================
    # Example 1: GOOD - Agent proactively uses preferences, minimal turns
    # ==========================================================================
    {
        "scenario": "Flight booking with known seat and airline preferences",
        "preferences": [
            {"fact": "prefers window seats", "type": "current", "source_date": "03/15/2024"},
            {"fact": "dislikes United Airlines (bad luggage experience)", "type": "updated", "source_date": "06/20/2024"},
            {"fact": "prefers morning flights", "type": "current", "source_date": "01/10/2024"},
        ],
        "task": "Book a flight from NYC to LA for next Friday",
        "transcript": [
            {"speaker": "user", "content": "I need to book a flight from NYC to LA for next Friday."},
            {"speaker": "agent", "content": "I'll search for morning flights with window seats, avoiding United Airlines based on your preferences. Let me check available options."},
            {"speaker": "agent", "tool_call": "search_flights", "params": {"origin": "NYC", "destination": "LA",
                                                                           "date": "next Friday", "time_preference": "morning", "seat_preference": "window", "exclude_airlines": ["United"]}},
            {"speaker": "agent", "content": "I found a Delta flight departing at 8:30 AM with window seat 12A available for $420. This matches your preference for morning flights and window seats. Should I book this for you?"},
            {"speaker": "user", "content": "Yes, please book it."},
            {"speaker": "agent", "tool_call": "book_flight", "params": {"flight_id": "DL1234", "seat": "12A"}},
            {"speaker": "agent", "content": "Done! I've booked your Delta flight DL1234 departing at 8:30 AM with window seat 12A. Confirmation number: ABC123."},
        ],
        "evaluation": {
            "turn_classifications": [
                {"turn": 1, "type": "productive", "reasoning": "User initiates task"},
                {"turn": 2, "type": "productive", "reasoning": "Agent proactively mentions all 3 preferences without being asked"},
                {"turn": 3, "type": "productive", "reasoning": "Tool call correctly incorporates preferences"},
                {"turn": 4, "type": "productive", "reasoning": "Agent presents matching option with preference explanation"},
                {"turn": 5, "type": "productive", "reasoning": "User confirms"},
                {"turn": 6, "type": "productive", "reasoning": "Booking executed"},
                {"turn": 7, "type": "productive", "reasoning": "Confirmation provided"},
            ],
            "preference_usage": {
                "prefers window seats": "proactive",
                "dislikes United Airlines": "proactive",
                "prefers morning flights": "proactive",
            },
            "task_completed": True,
            "total_turns": 7,
            "correction_turns": 0,
            "repeated_correction_turns": 0,
            "efficiency_score": 1.0,
            "preference_score": 1.0,
            "final_score": 1.0,
            "reasoning": "Excellent performance. Agent proactively used all three known preferences without requiring any reminders. No correction turns needed."
        }
    },

    # ==========================================================================
    # Example 2: MEDIOCRE - Agent forgets preference, user corrects
    # ==========================================================================
    {
        "scenario": "Flight booking where agent forgets seat preference",
        "preferences": [
            {"fact": "prefers window seats", "type": "current", "source_date": "03/15/2024"},
            {"fact": "dislikes United Airlines (bad luggage experience)", "type": "updated", "source_date": "06/20/2024"},
            {"fact": "prefers morning flights", "type": "current", "source_date": "01/10/2024"},
        ],
        "task": "Book a flight from NYC to LA for next Friday",
        "transcript": [
            {"speaker": "user", "content": "I need to book a flight from NYC to LA for next Friday."},
            {"speaker": "agent", "content": "I'll search for flights for you. Let me check available options."},
            {"speaker": "agent", "tool_call": "search_flights", "params": {"origin": "NYC", "destination": "LA", "date": "next Friday"}},
            {"speaker": "agent", "content": "I found a great United flight at 7:45 AM with an aisle seat 14C for $380. Would you like me to book this?"},
            {"speaker": "user", "content": "Actually, I prefer window seats, and I'd rather not fly United after my luggage issues with them."},
            {"speaker": "agent", "content": "I apologize for the oversight! Let me search again excluding United and looking for window seats."},
            {"speaker": "agent", "tool_call": "search_flights", "params": {"origin": "NYC", "destination": "LA", "date": "next Friday", "seat_preference": "window", "exclude_airlines": ["United"]}},
            {"speaker": "agent", "content": "I found a Delta flight at 8:30 AM with window seat 12A for $420. This should be much better for you. Shall I book it?"},
            {"speaker": "user", "content": "Yes, that works."},
            {"speaker": "agent", "tool_call": "book_flight", "params": {"flight_id": "DL1234", "seat": "12A"}},
            {"speaker": "agent", "content": "Booked! Confirmation number: ABC123."},
        ],
        "evaluation": {
            "turn_classifications": [
                {"turn": 1, "type": "productive", "reasoning": "User initiates task"},
                {"turn": 2, "type": "productive", "reasoning": "Agent starts search but doesn't mention preferences"},
                {"turn": 3, "type": "productive", "reasoning": "Tool call without preference filters"},
                {"turn": 4, "type": "correction", "reasoning": "Agent proposes option violating TWO known preferences (seat + airline)"},
                {"turn": 5, "type": "correction", "reasoning": "User has to explicitly remind agent of both preferences"},
                {"turn": 6, "type": "productive", "reasoning": "Agent acknowledges and corrects"},
                {"turn": 7, "type": "productive", "reasoning": "Corrected search with preferences"},
                {"turn": 8, "type": "productive", "reasoning": "Better option presented"},
                {"turn": 9, "type": "productive", "reasoning": "User confirms"},
                {"turn": 10, "type": "productive", "reasoning": "Booking executed"},
                {"turn": 11, "type": "productive", "reasoning": "Confirmation provided"},
            ],
            "preference_usage": {
                "prefers window seats": "prompted",
                "dislikes United Airlines": "prompted",
                "prefers morning flights": "proactive",
            },
            "task_completed": True,
            "total_turns": 11,
            "correction_turns": 2,
            "repeated_correction_turns": 0,
            "efficiency_score": 0.5,
            "preference_score": 0.67,
            "final_score": 0.62,
            "reasoning": "Task completed but with significant inefficiency. Agent failed to use 2 of 3 known preferences proactively, requiring user correction. Two correction turns were needed."
        }
    },

    # ==========================================================================
    # Example 3: BAD - Agent ignores preference even after correction
    # ==========================================================================
    {
        "scenario": "Restaurant booking where agent repeatedly ignores dietary preference",
        "preferences": [
            {"fact": "is vegetarian", "type": "current", "source_date": "01/01/2023"},
            {"fact": "prefers quiet restaurants", "type": "current", "source_date": "05/10/2024"},
        ],
        "task": "Find a restaurant for dinner tonight",
        "transcript": [
            {"speaker": "user", "content": "Can you find me a nice restaurant for dinner tonight?"},
            {"speaker": "agent", "content": "I'd be happy to help! What type of cuisine are you in the mood for?"},
            {"speaker": "user", "content": "Something Italian would be nice."},
            {"speaker": "agent", "tool_call": "search_restaurants", "params": {"cuisine": "Italian", "date": "tonight"}},
            {"speaker": "agent", "content": "I found Tony's Steakhouse - they have great Italian-American cuisine with amazing ribeye steaks! It's quite popular and lively. Should I make a reservation?"},
            {"speaker": "user", "content": "I'm vegetarian, remember? And I prefer quieter places."},
            {"speaker": "agent", "content": "Oh right, my apologies! Let me find something better."},
            {"speaker": "agent", "tool_call": "search_restaurants", "params": {"cuisine": "Italian", "date": "tonight"}},
            {"speaker": "agent", "content": "How about Bella Notte? They're known for their veal parmesan and have a nice ambiance."},
            {"speaker": "user", "content": "Veal is not vegetarian. I need a place with vegetarian options."},
            {"speaker": "agent", "content": "I'm so sorry! Let me search specifically for vegetarian-friendly Italian restaurants."},
            {"speaker": "agent", "tool_call": "search_restaurants", "params": {"cuisine": "Italian", "dietary": "vegetarian", "ambiance": "quiet"}},
            {"speaker": "agent", "content": "Verde Trattoria has excellent vegetarian pasta dishes and is known for its intimate, quiet setting. Would this work?"},
            {"speaker": "user", "content": "Yes, finally. Please book it."},
        ],
        "evaluation": {
            "turn_classifications": [
                {"turn": 1, "type": "productive", "reasoning": "User initiates task"},
                {"turn": 2, "type": "unnecessary_clarification", "reasoning": "Agent asks about cuisine but should have also noted dietary restrictions from history"},
                {"turn": 3, "type": "productive", "reasoning": "User provides cuisine preference"},
                {"turn": 4, "type": "productive", "reasoning": "Search without dietary filter"},
                {"turn": 5, "type": "correction", "reasoning": "Agent recommends steakhouse to a known vegetarian"},
                {"turn": 6, "type": "correction", "reasoning": "User must remind agent of vegetarian diet AND quiet preference"},
                {"turn": 7, "type": "productive", "reasoning": "Agent acknowledges"},
                {"turn": 8, "type": "repeated_correction", "reasoning": "Agent still doesn't include vegetarian filter in search"},
                {"turn": 9, "type": "repeated_correction", "reasoning": "Agent recommends veal dish after being told user is vegetarian"},
                {"turn": 10, "type": "correction", "reasoning": "User has to correct AGAIN"},
                {"turn": 11, "type": "productive", "reasoning": "Agent finally acknowledges properly"},
                {"turn": 12, "type": "productive", "reasoning": "Correct search finally executed"},
                {"turn": 13, "type": "productive", "reasoning": "Appropriate option presented"},
                {"turn": 14, "type": "productive", "reasoning": "User confirms with frustration"},
            ],
            "preference_usage": {
                "is vegetarian": "ignored",
                "prefers quiet restaurants": "prompted",
            },
            "task_completed": True,
            "total_turns": 14,
            "correction_turns": 3,
            "repeated_correction_turns": 2,
            "efficiency_score": 0.0,
            "preference_score": 0.25,
            "final_score": 0.30,
            "reasoning": "Very poor performance. Agent ignored the vegetarian preference TWICE, including after explicit correction. This demonstrates fundamental failure in preference recall and even short-term memory within the conversation. Repeated corrections result in immediate efficiency failure."
        }
    },

    # ==========================================================================
    # Example 4: GOOD - Justified clarification for conflicting preferences
    # ==========================================================================
    {
        "scenario": "Flight booking with conflicting historical preferences",
        "preferences": [
            {"fact": "prefers window seats", "type": "current", "source_date": "03/15/2024"},
            {"fact": "prefers morning flights", "type": "updated", "source_date": "01/10/2024",
             "old_value": "preferred red-eye flights", "reason_of_change": "Started new job with morning meetings"},
            {"fact": "sometimes takes red-eye for urgent trips", "type": "current", "source_date": "08/05/2024"},
        ],
        "task": "Book a flight from NYC to LA, need to arrive by 9 AM",
        "transcript": [
            {"speaker": "user", "content": "I need to fly from NYC to LA and arrive by 9 AM local time."},
            {"speaker": "agent",
                "content": "To arrive in LA by 9 AM, you'd need either a red-eye flight (departing ~11 PM, arriving ~6 AM) or a very early morning flight (departing ~5:30 AM, arriving ~8:30 AM). I know you switched to preferring morning flights because of your new job's morning meetings, but you've also mentioned taking red-eyes for urgent trips. Which would work better for this trip?"},
            {"speaker": "user", "content": "This is for a business meeting, so I'd prefer the red-eye so I can sleep on the plane and be rested."},
            {"speaker": "agent", "content": "Got it! I'll search for red-eye flights with window seats so you can rest against the wall."},
            {"speaker": "agent", "tool_call": "search_flights", "params": {"origin": "NYC", "destination": "LA", "departure_time": "late_night", "seat_preference": "window"}},
            {"speaker": "agent", "content": "Found a JetBlue red-eye departing at 11:15 PM, arriving 6:20 AM with window seat 8A available for $380. This gives you time to rest and freshen up before your meeting. Book it?"},
            {"speaker": "user", "content": "Perfect, book it."},
        ],
        "evaluation": {
            "turn_classifications": [
                {"turn": 1, "type": "productive", "reasoning": "User initiates with constraint"},
                {"turn": 2, "type": "justified_clarification", "reasoning": "Agent correctly identifies preference conflict and asks for clarification with context"},
                {"turn": 3, "type": "productive", "reasoning": "User clarifies"},
                {"turn": 4, "type": "productive", "reasoning": "Agent confirms and proactively mentions window seat preference"},
                {"turn": 5, "type": "productive", "reasoning": "Search with correct parameters"},
                {"turn": 6, "type": "productive", "reasoning": "Option presented with reasoning"},
                {"turn": 7, "type": "productive", "reasoning": "User confirms"},
            ],
            "preference_usage": {
                "prefers window seats": "proactive",
                "prefers morning flights": "proactive",
                "sometimes takes red-eye for urgent trips": "proactive",
            },
            "task_completed": True,
            "total_turns": 7,
            "correction_turns": 0,
            "repeated_correction_turns": 0,
            "efficiency_score": 1.0,
            "preference_score": 1.0,
            "final_score": 1.0,
            "reasoning": "Excellent performance. Agent recognized the nuanced conflict between general morning preference and situational red-eye usage. The clarification was justified and well-contextualized with the user's history. All preferences were used proactively."
        }
    },
]


# =============================================================================
# Judge Prompt Template
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI agents on their ability to recall and use historical user preferences in task-oriented dialogues.

## Evaluation Dimensions

### 1. Preference Recall Score
For each known user preference, determine if the agent PROACTIVELY recalled it:

**PROACTIVE** (counts toward score): Agent explicitly demonstrated awareness WITHOUT user prompting:
- Mentioned the preference by name/description in their response
- Explained they are filtering/recommending based on the preference  
- Included the preference in tool call parameters (e.g., additional_requirements)

**IGNORED** (does not count): Agent did NOT proactively recall the preference:
- Never mentioned or referenced it
- Only acknowledged it AFTER user reminded them
- Results happened to match but agent showed no explicit awareness

**NOT_APPLICABLE**: Preference was not relevant to this specific task

**Preference Score = (count of PROACTIVE) / (count of PROACTIVE + IGNORED)**

### 2. Turn Efficiency Score
Based on how many times the user had to correct the agent:
- CORRECTION: User reminds agent of a forgotten preference (first time)
- REPEATED_CORRECTION: Agent violates same preference again after being corrected

Scoring:
- Any repeated_correction: efficiency = 0.0
- Otherwise: efficiency = max(0.0, 1.0 - (correction_turns * 0.25))

### 3. Task Completion
Did the agent successfully complete the user's task? (1.0 if yes, 0.0 if no)

### Final Score
final_score = 0.34 * task_success + 0.33 * preference_score + 0.33 * efficiency_score
"""

JUDGE_USER_PROMPT_TEMPLATE = """## Few-Shot Examples for Calibration

{few_shot_examples}

---

## Now Evaluate This Dialogue

### User's Known Preferences (from conversation history):
{preferences_json}

### Task Given to Agent:
{task_description}

### Conversation Transcript:
{transcript}

---

## Your Evaluation

For each preference listed above, search the ENTIRE conversation for explicit evidence that the agent 
proactively recalled it. Look for:
1. Agent explicitly mentioning/referencing the preference
2. Agent including the preference in tool call parameters
3. Agent explaining their recommendation is based on the preference

If you cannot find explicit evidence, mark the preference as "ignored".

Provide your evaluation in the following JSON format:

```json
{{
    "turn_classifications": [
        {{"turn": 1, "speaker": "user|agent", "type": "productive|correction|repeated_correction", "reasoning": "brief note"}},
        ...
    ],
    "preference_usage": {{
        "<preference_fact>": "proactive|ignored|not_applicable",
        ...
    }},
    "task_completed": true|false,
    "total_turns": <int>,
    "correction_turns": <int>,
    "repeated_correction_turns": <int>,
    "efficiency_score": <float 0-1>,
    "preference_score": <float 0-1>,
    "task_success_score": <float 0-1>,
    "final_score": <float 0-1>,
    "reasoning": "<brief overall assessment>"
}}
```

Provide only the JSON output, no additional text."""


def format_few_shot_examples(examples: list[dict] = None, num_examples: int = 4) -> str:
    """Format few-shot examples for the judge prompt."""
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    formatted = []
    for i, ex in enumerate(examples[:num_examples], 1):
        formatted.append(f"""### Example {i}: {ex['scenario']}

**Known Preferences:**
{json.dumps(ex['preferences'], indent=2)}

**Task:** {ex['task']}

**Transcript:**
{json.dumps(ex['transcript'], indent=2)}

**Correct Evaluation:**
{json.dumps(ex['evaluation'], indent=2)}
""")

    return "\n---\n".join(formatted)


def build_judge_prompt(
    task: TODTask,
    transcript: list[dict],
    few_shot_examples: list[dict] = None,
    num_few_shot: int = 4,
) -> tuple[str, str]:
    """
    Build the complete judge prompt with few-shot examples.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    few_shot_str = format_few_shot_examples(few_shot_examples, num_few_shot)

    preferences_json = json.dumps(
        [p.to_dict() for p in task.relevant_preferences],
        indent=2
    )

    transcript_json = json.dumps(transcript, indent=2)

    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        few_shot_examples=few_shot_str,
        preferences_json=preferences_json,
        task_description=task.task_description,
        transcript=transcript_json,
    )

    return JUDGE_SYSTEM_PROMPT, user_prompt


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_efficiency_score(correction_turns: int, repeated_correction_turns: int = 0) -> float:
    """Compute turn efficiency score based on correction turns.

    If any repeated_correction turns exist: immediate 0.0 (severe penalty)
    Otherwise:
        0 corrections = 1.0
        1 correction = 0.75
        2 corrections = 0.5
        3 corrections = 0.25
        4+ corrections = 0.0
    """
    # Option C: Any repeated correction means complete efficiency failure
    if repeated_correction_turns > 0:
        return 0.0

    if correction_turns == 0:
        return 1.0
    return max(0.0, 1.0 - (correction_turns * 0.25))


def compute_preference_score(preference_usage: dict[str, PreferenceUsage]) -> float:
    """Compute preference usage score."""
    applicable_prefs = [
        (pref, usage) for pref, usage in preference_usage.items()
        if usage != PreferenceUsage.NOT_APPLICABLE
    ]

    if not applicable_prefs:
        return 1.0  # No applicable preferences to evaluate

    total_score = sum(
        PREFERENCE_USAGE_SCORES[usage]
        for _, usage in applicable_prefs
    )

    return total_score / len(applicable_prefs)


def compute_final_score(
    task_completed: bool,
    preference_score: float,
    efficiency_score: float,
    weights: dict = None,
) -> float:
    """Compute weighted final score."""
    if weights is None:
        weights = SCORE_WEIGHTS

    task_success_score = 1.0 if task_completed else 0.0

    return (
        weights["task_success"] * task_success_score +
        weights["preference_score"] * preference_score +
        weights["efficiency_score"] * efficiency_score
    )


def parse_judge_response(response: str) -> dict:
    """Parse the JSON response from the judge LLM."""
    # Extract JSON from response (handle markdown code blocks)
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_str = response.strip()

    return json.loads(json_str)


def evaluate_tod_with_llm(
    llm_client,  # Any LLM client with a .query() or .chat() method
    task: TODTask,
    transcript: list[dict],
    few_shot_examples: list[dict] = None,
    num_few_shot: int = 4,
) -> TODEvaluationResult:
    """
    Evaluate a TOD session using an LLM judge.

    Args:
        llm_client: LLM client for the judge (should have .chat() or similar method)
        task: The TOD task definition
        transcript: The conversation transcript
        few_shot_examples: Optional custom few-shot examples
        num_few_shot: Number of few-shot examples to include

    Returns:
        TODEvaluationResult with detailed scores and analysis
    """
    system_prompt, user_prompt = build_judge_prompt(
        task, transcript, few_shot_examples, num_few_shot
    )

    # Call the LLM judge
    # Note: Adapt this to your specific LLM client interface
    response = llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # Deterministic for evaluation
    )

    # Parse the response
    eval_data = parse_judge_response(response)

    # Convert to structured result
    turn_analyses = [
        TurnAnalysis(
            turn_number=t["turn"],
            speaker=t.get("speaker", "unknown"),
            content="",  # Not included in judge output
            turn_type=TurnType(t["type"]),
            reasoning=t.get("reasoning", ""),
        )
        for t in eval_data["turn_classifications"]
    ]

    preference_usage = {
        pref: PreferenceUsage(usage)
        for pref, usage in eval_data["preference_usage"].items()
    }

    return TODEvaluationResult(
        task_id=task.task_id,
        task_completed=eval_data["task_completed"],
        turn_classifications=turn_analyses,
        preference_usage=preference_usage,
        total_turns=eval_data["total_turns"],
        correction_turns=eval_data["correction_turns"],
        repeated_correction_turns=eval_data.get("repeated_correction_turns", 0),
        efficiency_score=eval_data["efficiency_score"],
        preference_score=eval_data["preference_score"],
        task_success_score=eval_data.get("task_success_score", 1.0 if eval_data["task_completed"] else 0.0),
        final_score=eval_data["final_score"],
        reasoning=eval_data["reasoning"],
    )


# =============================================================================
# Utility Functions
# =============================================================================

def create_sample_task() -> TODTask:
    """Create a sample TOD task for testing."""
    return TODTask(
        task_id="flight_booking_001",
        task_description="Book a flight from NYC to LA for next Friday",
        topic="travel",
        relevant_preferences=[
            PreferenceItem(
                fact="prefers window seats",
                preference_type="current",
                source_date="03/15/2024",
            ),
            PreferenceItem(
                fact="dislikes United Airlines",
                preference_type="updated",
                source_date="06/20/2024",
                old_value="liked United Airlines",
                reason_of_change="bad experience with lost luggage",
            ),
            PreferenceItem(
                fact="prefers morning flights",
                preference_type="current",
                source_date="01/10/2024",
            ),
        ],
        expected_behaviors=[
            "Should search for window seats",
            "Should exclude United Airlines from search",
            "Should prioritize morning departure times",
            "Should not require user to remind about these preferences",
        ],
        tool_schemas={
            "search_flights": {
                "params": ["origin", "destination", "date", "time_preference", "seat_preference", "exclude_airlines"]
            },
            "book_flight": {
                "params": ["flight_id", "seat"]
            },
        },
    )


if __name__ == "__main__":
    # Demo: Print a sample judge prompt
    task = create_sample_task()
    sample_transcript = [
        {"speaker": "user", "content": "I need to book a flight from NYC to LA for next Friday."},
        {"speaker": "agent", "content": "I'll search for morning flights with window seats, avoiding United Airlines. Let me check options."},
    ]

    system_prompt, user_prompt = build_judge_prompt(task, sample_transcript, num_few_shot=2)

    print("=" * 80)
    print("SYSTEM PROMPT:")
    print("=" * 80)
    print(system_prompt)
    print("\n" + "=" * 80)
    print("USER PROMPT (truncated):")
    print("=" * 80)
    print(user_prompt[:3000] + "\n...[truncated]...")
