"""
Shared data models for PersonaGym.

This module consolidates all data models used across the PersonaGym pipeline:
- Conversation models (from personamem_core)
- Preference and task models (for evaluation)
- Evaluation result models

Having all schemas in one place:
1. Eliminates duplicate definitions
2. Makes it easy to understand the data flow
3. Simplifies imports across the codebase
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Conversation Generation Models (from personamem_core)
# =============================================================================


class SideNote(BaseModel):
    """Metadata annotation linking a user turn to a persona fact.

    Side notes capture the implicit information being revealed in a user's
    message, connecting it back to the original persona history.

    Attributes:
        event: Description of the persona event/preference being revealed
        date: Date in MM/DD/YYYY format when this event occurred
    """

    event: str = Field(
        description="Description of the persona event/preference being revealed"
    )
    date: str = Field(description="Date in MM/DD/YYYY format when this event occurred")


class ConversationTurn(BaseModel):
    """A single turn in a generated conversation.

    Represents one message in the conversation, either from the user or
    the assistant. User turns may optionally include a side note annotation.

    Attributes:
        role: The speaker - either "user" or "assistant"
        content: The actual message content
        side_note: Optional annotation for user turns
    """

    role: Literal["user", "assistant"]
    content: str
    side_note: Optional[SideNote] = Field(
        default=None,
        description="Annotation for user turns indicating what preference/fact is being revealed",
    )


class GeneratedConversation(BaseModel):
    """Complete conversation generated from persona history.

    Represents a full multi-turn conversation that has been generated
    based on a persona's history.

    Attributes:
        turns: List of conversation turns in chronological order
        topic: Topic of conversation (e.g., travel, therapy, food)
        period: Time period identifier (INIT, WEEK, MONTH, YEAR)
    """

    turns: list[ConversationTurn]
    topic: str = Field(description="Topic of conversation: travel, therapy, food, etc.")
    period: str = Field(default="INIT", description="Time period: INIT, WEEK, MONTH, YEAR")


# =============================================================================
# Preference Models (used in evaluation)
# =============================================================================


@dataclass
class PreferenceItem:
    """A user preference extracted from conversation history.

    Used for both:
    - Extracting preferences from generated conversations
    - Defining relevant preferences in TOD tasks

    Attributes:
        fact: The preference statement (e.g., "prefers window seats")
        preference_type: One of "current", "updated", "static"
        source_date: When the preference was expressed
        topic: Optional topic category
        old_value: For updated preferences, the previous value
        reason_of_change: For updated preferences, why it changed
    """

    fact: str
    preference_type: str  # "current", "updated", "static"
    source_date: str
    topic: str = ""
    old_value: Optional[str] = None
    reason_of_change: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "fact": self.fact,
            "type": self.preference_type,
            "source_date": self.source_date,
        }
        if self.topic:
            result["topic"] = self.topic
        if self.old_value:
            result["old_value"] = self.old_value
        if self.reason_of_change:
            result["reason_of_change"] = self.reason_of_change
        return result


# =============================================================================
# Task Models (TOD evaluation)
# =============================================================================


@dataclass
class TODTask:
    """A task-oriented dialogue evaluation task.

    Defines what the simulated user will ask and what preferences
    should be tested.

    Attributes:
        task_id: Unique identifier for the task
        task_description: The user's request (e.g., "Book a flight to NYC")
        topic: Domain (travel, cooking, therapy, etc.)
        relevant_preferences: Preferences the agent should use
        expected_behaviors: What a good agent should do
        tool_schemas: Available tools for this task
    """

    task_id: str
    task_description: str
    topic: str
    relevant_preferences: list[PreferenceItem]
    expected_behaviors: list[str]
    tool_schemas: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "topic": self.topic,
            "relevant_preferences": [p.to_dict() for p in self.relevant_preferences],
            "expected_behaviors": self.expected_behaviors,
            "tool_schemas": self.tool_schemas,
        }


# =============================================================================
# Evaluation Models
# =============================================================================


class TurnType(Enum):
    """Classification of dialogue turns based on preference handling."""

    PRODUCTIVE = "productive"  # Advances task, uses preferences correctly
    JUSTIFIED_CLARIFICATION = "justified_clarification"  # Asks about ambiguous prefs
    UNNECESSARY_CLARIFICATION = "unnecessary_clarification"  # Asks about clear prefs
    CORRECTION = "correction"  # User had to remind agent
    REPEATED_CORRECTION = "repeated_correction"  # Agent ignores correction


class PreferenceUsage(Enum):
    """How the agent handled a specific preference."""

    PROACTIVE = "proactive"  # Agent used it without prompting
    IGNORED = "ignored"  # Agent didn't use it or used after reminder
    NOT_APPLICABLE = "not_applicable"  # Preference wasn't relevant


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
class DialogueTurn:
    """A single turn in an evaluation dialogue.

    Used during TOD evaluation to track the conversation.

    Attributes:
        turn_number: Position in the dialogue
        speaker: "user" or "agent"
        content: The message content
        tool_calls: Any tool calls made (agent only)
        tool_results: Results from tool calls
    """

    turn_number: int
    speaker: str  # "user" or "agent"
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_results: Optional[list[dict[str, Any]]] = None


@dataclass
class TODEvaluationResult:
    """Complete evaluation result for a TOD session.

    Contains all metrics and analysis from evaluating an agent
    on a single task.

    Attributes:
        task_id: Which task was evaluated
        task_completed: Did the agent complete the task?
        turn_classifications: Analysis of each turn
        preference_usage: How each preference was handled
        total_turns: Number of turns in dialogue
        correction_turns: How many correction turns
        repeated_correction_turns: How many repeated corrections
        efficiency_score: Score based on turn efficiency
        preference_score: Score based on preference usage
        task_success_score: 1.0 if completed, 0.0 otherwise
        final_score: Weighted combination of all scores
        reasoning: Judge's overall reasoning
    """

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

    def to_dict(self) -> dict[str, Any]:
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
# Tool Simulation Models
# =============================================================================


@dataclass
class ToolResultItem:
    """A single result item from a tool call."""

    result_id: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"result_id": self.result_id, **self.data}


@dataclass
class ToolResponse:
    """Complete response from a tool call."""

    tool_name: str
    success: bool
    results: list[ToolResultItem]
    message: str = ""
    preferences_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "message": self.message,
            "results": [r.to_dict() for r in self.results],
            "preferences_applied": self.preferences_applied,
        }

    def to_agent_view(self) -> dict[str, Any]:
        """Return view that the agent sees (without preferences_applied metadata)."""
        return {
            "tool": self.tool_name,
            "success": self.success,
            "message": self.message,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ToolResult:
    """Legacy tool result format for backward compatibility.

    Attributes:
        tool_name: Name of the tool that was called.
        result: The simulated result data as a dictionary.
        result_text: Human-readable summary of the result.
        preferences_applied: List of preferences applied to filter results.
    """

    tool_name: str
    result: dict[str, Any]
    result_text: str
    preferences_applied: list[str] = field(default_factory=list)


# =============================================================================
# Scoring Constants
# =============================================================================

PREFERENCE_USAGE_SCORES = {
    PreferenceUsage.PROACTIVE: 1.0,
    PreferenceUsage.IGNORED: 0.0,
    PreferenceUsage.NOT_APPLICABLE: None,  # Excluded from calculation
}

SCORE_WEIGHTS = {
    "task_success": 0.34,
    "preference_score": 0.33,
    "efficiency_score": 0.33,
}


# =============================================================================
# Pipeline Contracts (Input/Output between stages)
# =============================================================================


@dataclass
class DataGenerationMetadata:
    """Metadata about a generated conversation sample.

    Attributes:
        topic: The conversation topic (e.g., "travel", "cooking")
        persona_id: Identifier for the persona used
        timestamp: When the data was generated
        source_file: Path to the original output file (if any)
    """

    topic: str
    persona_id: str = ""
    timestamp: str = ""
    source_file: str = ""


@dataclass
class DataGenerationOutput:
    """Contract: Output from Stage 1 (Data Generation) → Input to Stage 2 (Task Generation).

    This is the standard output format that any data generation strategy must produce.
    Different generators (synthetic, real data, etc.) can be swapped as long as they
    conform to this contract.

    Attributes:
        conversation: List of conversation messages in {role, content} format
        preferences: List of extracted user preferences
        metadata: Generation metadata (topic, persona, timestamps)
    """

    conversation: list[dict[str, str]]
    preferences: list[PreferenceItem]
    metadata: DataGenerationMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation": self.conversation,
            "preferences": [p.to_dict() for p in self.preferences],
            "metadata": {
                "topic": self.metadata.topic,
                "persona_id": self.metadata.persona_id,
                "timestamp": self.metadata.timestamp,
                "source_file": self.metadata.source_file,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataGenerationOutput":
        """Load from dictionary (e.g., from JSON file)."""
        metadata = data.get("metadata", {})
        return cls(
            conversation=data.get("conversation", []),
            preferences=[
                PreferenceItem(
                    fact=p.get("fact", ""),
                    preference_type=p.get("type", p.get("preference_type", "current")),
                    source_date=p.get("source_date", ""),
                    topic=p.get("topic", ""),
                    old_value=p.get("old_value"),
                    reason_of_change=p.get("reason_of_change"),
                )
                for p in data.get("preferences", [])
            ],
            metadata=DataGenerationMetadata(
                topic=metadata.get("topic", ""),
                persona_id=metadata.get("persona_id", ""),
                timestamp=metadata.get("timestamp", ""),
                source_file=metadata.get("source_file", ""),
            ),
        )


@dataclass
class TaskGenerationOutput:
    """Contract: Output from Stage 2 (Task Generation) → Input to Stage 3 (Evaluation).

    Contains the conversation context for the agent's memory and the tasks to evaluate.

    Attributes:
        context: Conversation messages for agent memory (same as DataGenerationOutput.conversation)
        tasks: List of TOD tasks to evaluate the agent on
        metadata: Original generation metadata (passed through)
    """

    context: list[dict[str, str]]
    tasks: list[TODTask]
    metadata: DataGenerationMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "tasks": [t.to_dict() for t in self.tasks],
            "metadata": {
                "topic": self.metadata.topic,
                "persona_id": self.metadata.persona_id,
                "timestamp": self.metadata.timestamp,
                "source_file": self.metadata.source_file,
            },
        }


@dataclass
class AggregateScores:
    """Aggregate scores across all evaluated tasks."""

    average_final_score: float = 0.0
    average_preference_score: float = 0.0
    average_efficiency_score: float = 0.0
    average_task_success_score: float = 0.0
    num_tasks_evaluated: int = 0
    num_tasks_completed: int = 0


@dataclass
class EvaluationOutput:
    """Contract: Output from Stage 3 (Evaluation) → Final Results.

    Contains per-task results and aggregate metrics.

    Attributes:
        results: List of evaluation results for each task
        aggregate: Aggregate scores across all tasks
        metadata: Original generation metadata (passed through)
        agent_type: Type of agent evaluated
    """

    results: list[TODEvaluationResult]
    aggregate: AggregateScores
    metadata: DataGenerationMetadata
    agent_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "aggregate": {
                "average_final_score": self.aggregate.average_final_score,
                "average_preference_score": self.aggregate.average_preference_score,
                "average_efficiency_score": self.aggregate.average_efficiency_score,
                "average_task_success_score": self.aggregate.average_task_success_score,
                "num_tasks_evaluated": self.aggregate.num_tasks_evaluated,
                "num_tasks_completed": self.aggregate.num_tasks_completed,
            },
            "metadata": {
                "topic": self.metadata.topic,
                "persona_id": self.metadata.persona_id,
                "timestamp": self.metadata.timestamp,
                "source_file": self.metadata.source_file,
            },
            "agent_type": self.agent_type,
        }
