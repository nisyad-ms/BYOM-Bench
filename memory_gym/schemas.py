"""
Shared data models for PersonaGym.

This module consolidates all data models used across the PersonaGym pipeline:
- Preference and task models (for evaluation)
- Multi-session conversation and timeline models
- Evaluation result models

Having all schemas in one place:
1. Eliminates duplicate definitions
2. Makes it easy to understand the data flow
3. Simplifies imports across the codebase
"""

from dataclasses import dataclass, field
from typing import Any, Optional

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
        baseline_preferences: Core personality preferences grouped by domain (5 per domain)
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
    baseline_preferences: dict[str, list[str]] = None  # {domain: [pref1, pref2, ...]}

    def __post_init__(self):
        if self.baseline_preferences is None:
            self.baseline_preferences = {}

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
            for domain, prefs in self.baseline_preferences.items():
                lines.append(f"  [{domain}]")
                for pref in prefs:
                    lines.append(f"    - {pref}")
        return "\n".join(lines)


@dataclass
class LifeEvent:
    """A significant event in the user's life that may trigger preference changes.

    Attributes:
        session_id: Which session this event corresponds to
        date: Date of the event (MM/DD/YYYY format)
        event: Description of what happened
        domain: Life domain for this event
        user_prompt: Natural user message (preference-neutral) to start the conversation
    """

    session_id: int
    date: str
    event: str
    domain: str = ""
    user_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "date": self.date,
            "event": self.event,
            "domain": self.domain,
        }
        # Include task fields only if they're set (for evaluation events)
        if self.user_prompt:
            result["user_prompt"] = self.user_prompt
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LifeEvent":
        return cls(
            session_id=data["session_id"],
            date=data["date"],
            event=data["event"],
            domain=data.get("domain", ""),
            user_prompt=data.get("user_prompt", ""),
        )


@dataclass
class Preference:
    """A user preference with full lifecycle tracking.

    Unlike PreferenceItem (used in evaluation), this tracks the complete
    history of a preference including when it was created and superseded.

    Attributes:
        preference_id: Unique identifier (e.g., "pref_001")
        fact: The preference statement
        domain: Life domain (work_education, health_wellness, travel_experiences, relationships_personal, hobbies_interests)
        created_at_session: Session when this preference was first expressed
        created_at_date: Date when this preference was first expressed
        superseded_at_session: Session when this preference was replaced (None if still active)
        superseded_by: ID of the preference that replaced this one (None if still active)
        reason_for_change: Why this preference was superseded (if applicable)
    """

    preference_id: str
    fact: str
    domain: str
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
            "domain": self.domain,
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
            domain=data.get("domain", ""),
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
        domain: str,
        session_id: int,
        date: str,
    ) -> str:
        """Add a new preference and return its ID."""
        pref_id = f"pref_{self._next_id:03d}"
        self._next_id += 1
        self.preferences[pref_id] = Preference(
            preference_id=pref_id,
            fact=fact,
            domain=domain,
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
        new_domain: str | None = None,
    ) -> str:
        """Mark old preference as superseded and create new one. Returns new ID.

        Args:
            old_id: ID of preference to evolve
            new_fact: The new preference text
            session_id: Session when evolution occurs
            date: Date of evolution
            reason: Why this preference changed
            new_domain: Optional new domain (defaults to inheriting from old preference)
        """
        old_pref = self.preferences.get(old_id)
        if not old_pref:
            raise ValueError(f"Preference {old_id} not found")

        # Create new preference (use new_domain if provided, else inherit)
        new_id = f"pref_{self._next_id:03d}"
        self._next_id += 1
        self.preferences[new_id] = Preference(
            preference_id=new_id,
            fact=new_fact,
            domain=new_domain or old_pref.domain,
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

    def _session_to_dict(self, session: Session) -> dict[str, Any]:
        """Serialize a session with inline preferences."""
        # Build created preferences
        created = [
            {"id": p.preference_id, "fact": p.fact, "domain": p.domain}
            for pid in session.new_preference_ids
            if (p := self.timeline.preferences.get(pid))
        ]

        # Build evolved preferences
        evolved = []
        for old_id, new_id in session.evolved_preference_ids.items():
            old_p = self.timeline.preferences.get(old_id)
            new_p = self.timeline.preferences.get(new_id)
            if old_p and new_p:
                evolved.append(
                    {
                        "from": {"id": old_p.preference_id, "fact": old_p.fact},
                        "to": {"id": new_p.preference_id, "fact": new_p.fact, "domain": new_p.domain},
                        "reason": old_p.reason_for_change or "",
                    }
                )

        return {
            "session_id": session.session_id,
            "life_event": {
                "date": session.life_event.date,
                "event": session.life_event.event,
                "domain": session.life_event.domain,
            },
            "preferences": {"created": created, "evolved": evolved},
            "conversation": session.conversation,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to clean format with inline preferences per session."""
        active = [
            {"id": p.preference_id, "fact": p.fact, "domain": p.domain, "created_at_session": p.created_at_session}
            for p in self.timeline.get_active_preferences()
        ]
        superseded = [
            {
                "id": p.preference_id,
                "fact": p.fact,
                "domain": p.domain,
                "created_at_session": p.created_at_session,
                "superseded_at_session": p.superseded_at_session,
                "replaced_by": p.superseded_by,
                "reason": p.reason_for_change or "",
            }
            for p in self.timeline.preferences.values()
            if not p.is_active
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
        """Deserialize from dict.

        Reads the complete preference ledger from final_state, which contains:
        - All 25 baseline preferences (created_at_session=-1)
        - All preferences created during sessions
        - All superseded preferences with full evolution metadata
        """
        timeline = PreferenceTimeline()
        life_events: list[LifeEvent] = []
        sessions: list[Session] = []

        final_state = data.get("final_state", {})

        for p in final_state.get("active_preferences", []):
            pref_id = p["id"]
            timeline.preferences[pref_id] = Preference(
                preference_id=pref_id,
                fact=p["fact"],
                domain=p.get("domain", ""),
                created_at_session=p.get("created_at_session", -1),
                created_at_date="",
            )
            timeline._next_id = max(timeline._next_id, int(pref_id.split("_")[1]) + 1)

        for p in final_state.get("superseded_preferences", []):
            pref_id = p["id"]
            timeline.preferences[pref_id] = Preference(
                preference_id=pref_id,
                fact=p["fact"],
                domain=p.get("domain", ""),
                created_at_session=p.get("created_at_session", -1),
                created_at_date="",
                superseded_at_session=p.get("superseded_at_session"),
                superseded_by=p.get("replaced_by"),
                reason_for_change=p.get("reason", ""),
            )
            timeline._next_id = max(timeline._next_id, int(pref_id.split("_")[1]) + 1)

        for s_data in data.get("sessions", []):
            session_id = s_data["session_id"]
            le_data = s_data["life_event"]

            life_event = LifeEvent(
                session_id=session_id, date=le_data["date"], event=le_data["event"], domain=le_data.get("domain", "")
            )
            life_events.append(life_event)

            prefs_data = s_data.get("preferences", {})
            new_preference_ids = [p["id"] for p in prefs_data.get("created", [])]
            for e in prefs_data.get("evolved", []):
                new_preference_ids.append(e["to"]["id"])
            evolved_preference_ids = {e["from"]["id"]: e["to"]["id"] for e in prefs_data.get("evolved", [])}

            sessions.append(
                Session(
                    session_id=session_id,
                    life_event=life_event,
                    conversation=s_data.get("conversation", []),
                    active_preference_ids=timeline.get_preference_ids_at_session(session_id),
                    new_preference_ids=new_preference_ids,
                    evolved_preference_ids=evolved_preference_ids,
                )
            )

        return cls(
            persona=data["persona"],
            persona_id=data.get("persona_id", ""),
            life_events=life_events,
            timeline=timeline,
            sessions=sessions,
            generation_timestamp=data.get("generation_timestamp", ""),
            expanded_persona=ExpandedPersona.from_dict(data["expanded_persona"])
            if data.get("expanded_persona")
            else None,
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

    def get_evolved_preferences(self) -> list[tuple[Preference, Preference]]:
        """Returns pairs of (stale_preference, evolved_preference) for all evolutions.

        An evolved preference is a current preference that replaced a stale one.
        This is the crown jewel of our evaluation - testing whether agents
        use the new preference vs the old one.

        Returns:
            List of (old_pref, new_pref) tuples showing the evolution chain
        """
        evolutions = []
        for stale in self.get_superseded_preferences():
            if stale.superseded_by:
                new_pref = self.timeline.preferences.get(stale.superseded_by)
                if new_pref and new_pref.is_active:
                    evolutions.append((stale, new_pref))
        return evolutions

    def get_evolved_preference_ids(self) -> set[str]:
        """Returns IDs of current preferences that evolved from stale ones.

        These are the preferences that replaced older versions and are
        critical for testing preference recall.
        """
        return {new_pref.preference_id for _, new_pref in self.get_evolved_preferences()}

    def get_baseline_preferences(self) -> list[Preference]:
        """Returns current preferences that are NOT evolved (original baseline).

        These are preferences that have never changed - either baseline prefs
        (created_at_session=-1) or new prefs that were never superseded.
        """
        evolved_ids = self.get_evolved_preference_ids()
        return [p for p in self.get_current_preferences() if p.preference_id not in evolved_ids]


# =============================================================================
# Multi-Session Evaluation Models
# =============================================================================


@dataclass
class EvaluationRubric:
    """Evaluation rubric generated by orchestrator for the judge.

    This defines what the judge should look for when scoring the conversation.

    Attributes:
        required_preferences: Preferences that MUST be applied for this task.
            Each entry is a dict with {id, fact, supersedes?: {id, fact}}.
            - id: Preference identifier
            - fact: The preference statement
            - supersedes: (optional) The stale preference this replaced
    """

    required_preferences: list[dict]  # [{id, fact, supersedes?: {id, fact}}, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_preferences": self.required_preferences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRubric":
        return cls(
            required_preferences=data["required_preferences"],
        )


@dataclass
class EvaluationTask:
    """An evaluation task for multi-session preference recall.

    Contains the evaluation event and rubric for scoring.
    The user's initial prompt is accessed via evaluation_event.user_prompt.

    Attributes:
        task_id: Unique identifier
        evaluation_event: The life event triggering this evaluation (contains user_prompt)
        scenario_type: Broad scenario category chosen by the task generator
        reasoning: Task generator's reasoning about how preferences interact in this scenario
        rubric: The evaluation rubric for the judge (contains required_preferences)
        persona_summary: Brief summary of the persona for user simulator
    """

    task_id: str
    evaluation_event: LifeEvent
    rubric: EvaluationRubric
    persona_summary: str
    scenario_type: str = ""
    reasoning: str = ""

    @property
    def user_prompt(self) -> str:
        """Convenience accessor for the starting user message."""
        return self.evaluation_event.user_prompt

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "evaluation_event": self.evaluation_event.to_dict(),
        }
        if self.scenario_type:
            result["event_type"] = self.scenario_type
        if self.reasoning:
            result["reasoning"] = self.reasoning
        result["rubric"] = self.rubric.to_dict()
        result["persona_summary"] = self.persona_summary
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationTask":
        return cls(
            task_id=data["task_id"],
            evaluation_event=LifeEvent.from_dict(data["evaluation_event"]),
            rubric=EvaluationRubric.from_dict(data["rubric"]),
            persona_summary=data["persona_summary"],
            scenario_type=data.get("event_type", data.get("scenario_type", "")),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class MultiSessionEvaluationResult:
    """Result from evaluating an agent on multi-session data.

    Attributes:
        task_id: Which task was evaluated
        conversation: Full conversation transcript
        preference_usage: How each current preference was handled
        stale_preference_usage: Which stale preferences the agent incorrectly used
        turn_classifications: Per-turn scoring details from judge (for debugging)
        total_turns: Number of turns in dialogue
        productive_turns: Agent turns that demonstrated specific preference knowledge
        generic_turns: Agent turns with helpful but unpersonalized advice
        correction_turns: How many times user corrected agent
        ignored_turns: Agent omitted preference, user had to reveal it
        repeated_correction_turns: Same preference violated after being corrected
        stale_count: Number of stale (outdated) preferences used
        proactive_count: Number of preferences proactively applied
        efficiency_score: Score based on turn efficiency
        preference_score: Score based on preference usage (stale penalty integrated)
        reasoning: Judge's overall reasoning
    """

    task_id: str
    conversation: list[dict[str, str]]
    preference_usage: dict[str, str]  # pref_id -> "proactive" | "ignored"
    stale_preference_usage: list[str]  # List of stale pref_ids that were incorrectly used
    evaluation_event: "LifeEvent | None" = None  # The evaluation scenario
    rubric: "EvaluationRubric | None" = None  # The rubric used for evaluation
    first_mention_trace: list[dict[str, Any]] | None = None  # v2: Chronological first-mention analysis
    turn_classifications: list[dict[str, Any]] | None = None  # Per-turn scoring details
    total_turns: int = 0
    productive_turns: int = 0
    generic_turns: int = 0
    correction_turns: int = 0
    ignored_turns: int = 0
    repeated_correction_turns: int = 0
    stale_count: int = 0
    proactive_count: int = 0
    efficiency_score: float = 0.0
    preference_score: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "scores": {
                "preference_score": self.preference_score,
                "efficiency_score": self.efficiency_score,
            },
        }
        if self.evaluation_event:
            result["evaluation_event"] = self.evaluation_event.to_dict()
        result.update(
            {
                "conversation": self.conversation,
                "preference_scoring": {
                    "proactive_count": self.proactive_count,
                    "stale_count": self.stale_count,
                    "stale_preference_usage": self.stale_preference_usage,
                    "first_mention_trace": self.first_mention_trace,
                },
                "efficiency_scoring": {
                    "total_turns": self.total_turns,
                    "productive_turns": self.productive_turns,
                    "generic_turns": self.generic_turns,
                    "correction_turns": self.correction_turns,
                    "ignored_turns": self.ignored_turns,
                    "repeated_correction_turns": self.repeated_correction_turns,
                    "turn_classifications": self.turn_classifications,
                },
            }
        )
        if self.rubric:
            result["rubric"] = self.rubric.to_dict()
        return result


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
