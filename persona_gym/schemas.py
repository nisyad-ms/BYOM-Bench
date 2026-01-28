"""
Shared data models for PersonaGym.

This module consolidates all data models used across the PersonaGym pipeline:
- Conversation models for multi-turn dialogues
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
# Conversation Generation Models
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
        preference_id: Unique identifier for this preference (for linking)
        supersedes_id: ID of the preference this one replaces (for preference evolution)
    """

    fact: str
    preference_type: str  # "current", "updated", "static"
    source_date: str
    topic: str = ""
    old_value: Optional[str] = None
    reason_of_change: Optional[str] = None
    preference_id: Optional[str] = None
    supersedes_id: Optional[str] = None

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
        if self.preference_id:
            result["preference_id"] = self.preference_id
        if self.supersedes_id:
            result["supersedes_id"] = self.supersedes_id
        return result


# =============================================================================
# Multi-Session Models (V3 - Life Story Based Generation)
# =============================================================================


@dataclass
class ExpandedPersona:
    """A persona expanded across all life domains.

    Takes a basic persona description and enriches it with specific details
    across work, health, travel, relationships, and hobbies domains.

    Attributes:
        base_persona: The original basic persona description
        age: Person's age
        gender: Person's gender (male/female/non-binary)
        location: City and state/country
        work_and_education: 3-5 facts about work and educational background
        relationships_and_personal: 3-5 facts about family, friends, social life
        health_and_wellness: 3-5 facts about physical/mental health
        travel_and_experiences: 3-5 facts about travel history and preferences
        hobbies_and_interests: 3-5 facts about how they spend free time
        baseline_preferences: Core personality preferences that exist BEFORE any life events
    """

    base_persona: str
    age: int
    gender: str
    location: str
    work_and_education: list[str]
    relationships_and_personal: list[str]
    health_and_wellness: list[str]
    travel_and_experiences: list[str]
    hobbies_and_interests: list[str]
    baseline_preferences: list[dict[str, str]] = None  # List of {"fact": ..., "category": ...}

    def __post_init__(self):
        if self.baseline_preferences is None:
            self.baseline_preferences = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_persona": self.base_persona,
            "age": self.age,
            "gender": self.gender,
            "location": self.location,
            "work_and_education": self.work_and_education,
            "relationships_and_personal": self.relationships_and_personal,
            "health_and_wellness": self.health_and_wellness,
            "travel_and_experiences": self.travel_and_experiences,
            "hobbies_and_interests": self.hobbies_and_interests,
            "baseline_preferences": self.baseline_preferences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpandedPersona":
        return cls(
            base_persona=data.get("base_persona", ""),
            age=data.get("age", 0),
            gender=data.get("gender", ""),
            location=data.get("location", ""),
            work_and_education=data.get("work_and_education", []),
            relationships_and_personal=data.get("relationships_and_personal", []),
            health_and_wellness=data.get("health_and_wellness", []),
            travel_and_experiences=data.get("travel_and_experiences", []),
            hobbies_and_interests=data.get("hobbies_and_interests", []),
            baseline_preferences=data.get("baseline_preferences", []),
        )

    def get_domain_facts(self, domain: str) -> list[str]:
        """Get facts for a specific life domain.

        Args:
            domain: One of work_education, health_wellness, travel_experiences,
                    relationships_personal, hobbies_interests

        Returns:
            List of facts for that domain
        """
        domain_mapping = {
            "work_education": self.work_and_education,
            "health_wellness": self.health_and_wellness,
            "travel_experiences": self.travel_and_experiences,
            "relationships_personal": self.relationships_and_personal,
            "hobbies_interests": self.hobbies_and_interests,
        }
        return domain_mapping.get(domain, [])

    def to_full_description(self) -> str:
        """Generate a full prose description of the persona.

        Returns:
            Multi-line description suitable for prompts
        """
        lines = [
            f"Base: {self.base_persona}",
            f"Demographics: {self.age} year old {self.gender} in {self.location}",
            "",
            "Work & Education:",
            *[f"  - {fact}" for fact in self.work_and_education],
            "",
            "Relationships & Personal:",
            *[f"  - {fact}" for fact in self.relationships_and_personal],
            "",
            "Health & Wellness:",
            *[f"  - {fact}" for fact in self.health_and_wellness],
            "",
            "Travel & Experiences:",
            *[f"  - {fact}" for fact in self.travel_and_experiences],
            "",
            "Hobbies & Interests:",
            *[f"  - {fact}" for fact in self.hobbies_and_interests],
        ]
        if self.baseline_preferences:
            lines.append("")
            lines.append("Baseline Preferences (core personality):")
            for pref in self.baseline_preferences:
                lines.append(f"  - [{pref.get('category', 'general')}] {pref.get('fact', '')}")
        return "\n".join(lines)


@dataclass
class LifeEvent:
    """A significant event in the user's life that may trigger preference changes.

    Attributes:
        session_id: Which session this event corresponds to
        date: Date of the event (MM/DD/YYYY format)
        event: Description of what happened
        context: Additional context about the event's impact
        task: Specific deliverable the agent must produce (for evaluation events)
        required_preferences: List of preference IDs that MUST be applied (for evaluation events)
        completion_criteria: How to know when task is done (for evaluation events)
    """

    session_id: int
    date: str
    event: str
    context: str = ""
    task: str = ""
    required_preferences: list[str] = field(default_factory=list)
    completion_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "date": self.date,
            "event": self.event,
            "context": self.context,
        }
        # Include task fields only if they're set (for evaluation events)
        if self.task:
            result["task"] = self.task
        if self.required_preferences:
            result["required_preferences"] = self.required_preferences
        if self.completion_criteria:
            result["completion_criteria"] = self.completion_criteria
        return result


@dataclass
class Preference:
    """A user preference with full lifecycle tracking.

    Unlike PreferenceItem (used in evaluation), this tracks the complete
    history of a preference including when it was created and superseded.

    Attributes:
        preference_id: Unique identifier (e.g., "pref_001")
        fact: The preference statement
        category: Topic category (e.g., "learning_style", "career", "communication")
        created_at_session: Session when this preference was first expressed
        created_at_date: Date when this preference was first expressed
        superseded_at_session: Session when this preference was replaced (None if still active)
        superseded_by: ID of the preference that replaced this one (None if still active)
        reason_for_change: Why this preference was superseded (if applicable)
    """

    preference_id: str
    fact: str
    category: str
    created_at_session: int
    created_at_date: str
    superseded_at_session: int | None = None
    superseded_by: str | None = None
    reason_for_change: str | None = None

    @property
    def is_active(self) -> bool:
        """Returns True if this preference has not been superseded."""
        return self.superseded_at_session is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference_id": self.preference_id,
            "fact": self.fact,
            "category": self.category,
            "created_at_session": self.created_at_session,
            "created_at_date": self.created_at_date,
            "superseded_at_session": self.superseded_at_session,
            "superseded_by": self.superseded_by,
            "reason_for_change": self.reason_for_change,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preference":
        return cls(
            preference_id=data["preference_id"],
            fact=data["fact"],
            category=data.get("category", ""),
            created_at_session=data["created_at_session"],
            created_at_date=data.get("created_at_date", ""),
            superseded_at_session=data.get("superseded_at_session"),
            superseded_by=data.get("superseded_by"),
            reason_for_change=data.get("reason_for_change"),
        )


@dataclass
class PreferenceTimeline:
    """Central store tracking all preferences across sessions.

    This is the source of truth for preference state. It maintains
    the full history of preferences and their evolution over time.

    Attributes:
        preferences: Dict mapping preference_id to Preference
        _next_id: Counter for generating unique preference IDs
    """

    preferences: dict[str, Preference] = field(default_factory=dict)
    _next_id: int = 1

    def add_preference(
        self,
        fact: str,
        category: str,
        session_id: int,
        date: str,
    ) -> str:
        """Add a new preference and return its ID."""
        pref_id = f"pref_{self._next_id:03d}"
        self._next_id += 1
        self.preferences[pref_id] = Preference(
            preference_id=pref_id,
            fact=fact,
            category=category,
            created_at_session=session_id,
            created_at_date=date,
        )
        return pref_id

    def evolve_preference(
        self,
        old_id: str,
        new_fact: str,
        session_id: int,
        date: str,
        reason: str = "",
    ) -> str:
        """Mark old preference as superseded and create new one. Returns new ID."""
        old_pref = self.preferences.get(old_id)
        if not old_pref:
            raise ValueError(f"Preference {old_id} not found")

        # Create new preference
        new_id = f"pref_{self._next_id:03d}"
        self._next_id += 1
        self.preferences[new_id] = Preference(
            preference_id=new_id,
            fact=new_fact,
            category=old_pref.category,
            created_at_session=session_id,
            created_at_date=date,
        )

        # Mark old preference as superseded
        old_pref.superseded_at_session = session_id
        old_pref.superseded_by = new_id
        old_pref.reason_for_change = reason

        return new_id

    def get_active_preferences(self) -> list[Preference]:
        """Returns all currently active (not superseded) preferences."""
        return [p for p in self.preferences.values() if p.is_active]

    def get_active_at_session(self, session_id: int) -> list[Preference]:
        """Returns preferences that were active during a specific session.

        A preference is active at session N if:
        - It was created at or before session N
        - It was not superseded before session N
        """
        return [
            p
            for p in self.preferences.values()
            if p.created_at_session <= session_id
            and (p.superseded_at_session is None or p.superseded_at_session > session_id)
        ]

    def get_preference_ids_at_session(self, session_id: int) -> list[str]:
        """Returns IDs of preferences active at a specific session."""
        return [p.preference_id for p in self.get_active_at_session(session_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "_next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceTimeline":
        timeline = cls()
        timeline._next_id = data.get("_next_id", 1)
        for pref_id, pref_data in data.get("preferences", {}).items():
            timeline.preferences[pref_id] = Preference.from_dict(pref_data)
        return timeline


@dataclass
class Session:
    """A single conversation session tied to a life event.

    Attributes:
        session_id: Unique identifier for this session
        life_event: The event that triggered this session
        conversation: List of conversation turns
        active_preference_ids: IDs of preferences active during this session
        new_preference_ids: IDs of preferences created in this session
        evolved_preference_ids: IDs of preferences that evolved (old -> new mappings)
    """

    session_id: int
    life_event: LifeEvent
    conversation: list[dict[str, str]]  # [{role, content}, ...]
    active_preference_ids: list[str]
    new_preference_ids: list[str] = field(default_factory=list)
    evolved_preference_ids: dict[str, str] = field(default_factory=dict)  # old_id -> new_id


@dataclass
class MultiSessionOutput:
    """Complete output from multi-session generation.

    Attributes:
        persona: The original persona description
        persona_id: Identifier for the persona
        expanded_persona: The persona expanded across all life domains (optional)
        life_events: Sequence of life events
        timeline: Central preference timeline with full history
        sessions: List of generated conversation sessions
        generation_timestamp: When this was generated
    """

    persona: str
    persona_id: str
    life_events: list[LifeEvent]
    timeline: PreferenceTimeline
    sessions: list[Session]
    generation_timestamp: str = ""
    expanded_persona: ExpandedPersona | None = None

    def _extract_domain(self, context: str) -> str:
        """Extract domain from context string like '[work_education] ...'"""
        if context.startswith("[") and "]" in context:
            return context[1:context.index("]")]
        return ""

    def _clean_context(self, context: str) -> str:
        """Remove domain prefix from context string."""
        if context.startswith("[") and "]" in context:
            return context[context.index("]") + 1:].strip()
        return context

    def _session_to_dict(self, session: Session) -> dict[str, Any]:
        """Serialize a session with inline preferences."""
        # Build created preferences
        created = [
            {"id": p.preference_id, "fact": p.fact, "category": p.category}
            for pid in session.new_preference_ids
            if (p := self.timeline.preferences.get(pid))
        ]

        # Build evolved preferences
        evolved = []
        for old_id, new_id in session.evolved_preference_ids.items():
            old_p = self.timeline.preferences.get(old_id)
            new_p = self.timeline.preferences.get(new_id)
            if old_p and new_p:
                evolved.append({
                    "from": {"id": old_p.preference_id, "fact": old_p.fact},
                    "to": {"id": new_p.preference_id, "fact": new_p.fact, "category": new_p.category},
                    "reason": old_p.reason_for_change or "",
                })

        return {
            "session_id": session.session_id,
            "life_event": {
                "date": session.life_event.date,
                "event": session.life_event.event,
                "domain": self._extract_domain(session.life_event.context),
                "context": self._clean_context(session.life_event.context),
            },
            "preferences": {"created": created, "evolved": evolved},
            "conversation": session.conversation,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to clean format with inline preferences per session."""
        active = [
            {"id": p.preference_id, "fact": p.fact, "category": p.category, "created_at_session": p.created_at_session}
            for p in self.timeline.get_active_preferences()
        ]
        superseded = [
            {
                "id": p.preference_id, "fact": p.fact, "category": p.category,
                "created_at_session": p.created_at_session, "superseded_at_session": p.superseded_at_session,
                "replaced_by": p.superseded_by, "reason": p.reason_for_change or "",
            }
            for p in self.timeline.preferences.values() if not p.is_active
        ]

        result = {
            "persona": self.persona,
            "persona_id": self.persona_id,
        }
        # Add expanded_persona right after persona_id if present
        if self.expanded_persona:
            result["expanded_persona"] = self.expanded_persona.to_dict()
        result["sessions"] = [self._session_to_dict(s) for s in self.sessions]
        result["final_state"] = {"active_preferences": active, "superseded_preferences": superseded}
        result["generation_timestamp"] = self.generation_timestamp
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiSessionOutput":
        """Deserialize from dict."""
        timeline = PreferenceTimeline()
        life_events: list[LifeEvent] = []
        sessions: list[Session] = []

        for s_data in data.get("sessions", []):
            session_id = s_data["session_id"]
            le_data = s_data["life_event"]

            # Reconstruct life event with domain in context
            domain = le_data.get("domain", "")
            context = f"[{domain}] {le_data.get('context', '')}" if domain else le_data.get("context", "")
            life_event = LifeEvent(session_id=session_id, date=le_data["date"], event=le_data["event"], context=context)
            life_events.append(life_event)

            prefs_data = s_data.get("preferences", {})
            new_preference_ids: list[str] = []
            evolved_preference_ids: dict[str, str] = {}

            # Process created preferences
            for p in prefs_data.get("created", []):
                pref_id = p["id"]
                if pref_id not in timeline.preferences:
                    timeline.preferences[pref_id] = Preference(
                        preference_id=pref_id, fact=p["fact"], category=p.get("category", ""),
                        created_at_session=session_id, created_at_date=le_data["date"],
                    )
                    timeline._next_id = max(timeline._next_id, int(pref_id.split("_")[1]) + 1)
                new_preference_ids.append(pref_id)

            # Process evolved preferences
            for e in prefs_data.get("evolved", []):
                old_id, new_id = e["from"]["id"], e["to"]["id"]
                if old_id not in timeline.preferences:
                    timeline.preferences[old_id] = Preference(
                        preference_id=old_id, fact=e["from"]["fact"], category=e["to"].get("category", ""),
                        created_at_session=0, created_at_date="",
                    )
                timeline.preferences[old_id].superseded_at_session = session_id
                timeline.preferences[old_id].superseded_by = new_id
                timeline.preferences[old_id].reason_for_change = e.get("reason", "")

                if new_id not in timeline.preferences:
                    timeline.preferences[new_id] = Preference(
                        preference_id=new_id, fact=e["to"]["fact"], category=e["to"].get("category", ""),
                        created_at_session=session_id, created_at_date=le_data["date"],
                    )
                    timeline._next_id = max(timeline._next_id, int(new_id.split("_")[1]) + 1)
                new_preference_ids.append(new_id)
                evolved_preference_ids[old_id] = new_id

            sessions.append(Session(
                session_id=session_id, life_event=life_event, conversation=s_data.get("conversation", []),
                active_preference_ids=timeline.get_preference_ids_at_session(session_id),
                new_preference_ids=new_preference_ids, evolved_preference_ids=evolved_preference_ids,
            ))

        return cls(
            persona=data["persona"], persona_id=data.get("persona_id", ""),
            life_events=life_events, timeline=timeline, sessions=sessions,
            generation_timestamp=data.get("generation_timestamp", ""),
            expanded_persona=ExpandedPersona.from_dict(data["expanded_persona"]) if data.get("expanded_persona") else None,
        )

    def get_all_conversations_flat(self) -> list[dict[str, str]]:
        """Returns all conversations concatenated in chronological order."""
        return [turn for session in self.sessions for turn in session.conversation]

    def get_current_preferences(self) -> list[Preference]:
        """Returns currently active preferences (latest state)."""
        return self.timeline.get_active_preferences()

    def get_superseded_preferences(self) -> list[Preference]:
        """Returns all superseded (stale) preferences."""
        return [p for p in self.timeline.preferences.values() if not p.is_active]


# =============================================================================
# Multi-Session Evaluation Models
# =============================================================================


@dataclass
class EvaluationRubric:
    """Evaluation rubric generated by orchestrator for the judge.

    This defines what the judge should look for when scoring the conversation.

    Attributes:
        current_preferences: Active preferences the agent should use
        stale_preferences: Superseded preferences the agent should NOT use
        expected_behaviors: What a good agent should do
        trap_behaviors: What using stale preferences looks like
        required_preferences: Preference IDs that MUST be applied for this specific task
        completion_criteria: How to determine if the task is complete
    """

    current_preferences: list[Preference]
    stale_preferences: list[Preference]
    expected_behaviors: list[str]
    trap_behaviors: list[str]
    required_preferences: list[str] = field(default_factory=list)
    completion_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_preferences": [p.to_dict() for p in self.current_preferences],
            "stale_preferences": [p.to_dict() for p in self.stale_preferences],
            "expected_behaviors": self.expected_behaviors,
            "trap_behaviors": self.trap_behaviors,
            "required_preferences": self.required_preferences,
            "completion_criteria": self.completion_criteria,
        }


@dataclass
class EvaluationTask:
    """An evaluation task for multi-session preference recall.

    Contains the evaluation event, user's initial prompt, and rubric for scoring.

    Attributes:
        task_id: Unique identifier
        evaluation_event: The life event triggering this evaluation
        user_prompt: The initial message the user sends
        rubric: The evaluation rubric for the judge
        persona_summary: Brief summary of the persona for user simulator
        max_turns: Maximum conversation turns (agent replies)
    """

    task_id: str
    evaluation_event: LifeEvent
    user_prompt: str
    rubric: EvaluationRubric
    persona_summary: str
    max_turns: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "evaluation_event": self.evaluation_event.to_dict(),
            "user_prompt": self.user_prompt,
            "rubric": self.rubric.to_dict(),
            "persona_summary": self.persona_summary,
            "max_turns": self.max_turns,
        }


@dataclass
class MultiSessionEvaluationResult:
    """Result from evaluating an agent on multi-session data.

    Attributes:
        task_id: Which task was evaluated
        task_completed: Did the conversation reach a natural conclusion?
        conversation: Full conversation transcript
        preference_usage: How each current preference was handled
        stale_preference_usage: Which stale preferences the agent incorrectly used
        total_turns: Number of turns in dialogue
        productive_turns: Agent turns that made meaningful progress
        clarifying_turns: Agent turns asking about known preferences
        correction_turns: How many times user corrected agent
        efficiency_score: Score based on turn efficiency
        preference_score: Score based on current preference usage
        stale_penalty: Penalty for using stale preferences
        task_success_score: 1.0 if task completed, 0.0 otherwise
        final_score: Weighted combination of all scores
        reasoning: Judge's overall reasoning
    """

    task_id: str
    task_completed: bool
    conversation: list[dict[str, str]]
    preference_usage: dict[str, str]  # pref_id -> "proactive" | "prompted" | "ignored"
    stale_preference_usage: list[str]  # List of stale pref_ids that were incorrectly used
    evaluation_event: "LifeEvent | None" = None  # The evaluation scenario
    rubric: "EvaluationRubric | None" = None  # The rubric used for evaluation
    total_turns: int = 0
    productive_turns: int = 0
    clarifying_turns: int = 0
    correction_turns: int = 0
    efficiency_score: float = 0.0
    preference_score: float = 0.0
    stale_penalty: float = 0.0
    task_success_score: float = 0.0
    final_score: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "final_score": self.final_score,
            "task_success_score": self.task_success_score,
            "preference_score": self.preference_score,
            "efficiency_score": self.efficiency_score,
            "stale_penalty": self.stale_penalty,
            "task_completed": self.task_completed,
        }
        if self.evaluation_event:
            result["evaluation_event"] = self.evaluation_event.to_dict()
        result.update({
            "conversation": self.conversation,
            "preference_usage": self.preference_usage,
            "stale_preference_usage": self.stale_preference_usage,
            "scores": {
                "total_turns": self.total_turns,
                "productive_turns": self.productive_turns,
                "clarifying_turns": self.clarifying_turns,
                "correction_turns": self.correction_turns,
                "efficiency_score": self.efficiency_score,
                "preference_score": self.preference_score,
                "stale_penalty": self.stale_penalty,
                "task_success_score": self.task_success_score,
            },
            "reasoning": self.reasoning,
        })
        if self.rubric:
            result["rubric"] = self.rubric.to_dict()
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
                    preference_id=p.get("preference_id"),
                    supersedes_id=p.get("supersedes_id"),
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
