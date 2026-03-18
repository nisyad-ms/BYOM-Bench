"""
Shared data models for MemoryGym.

This module consolidates all data models used across the MemoryGym pipeline:
- Preference and task models (for evaluation)
- Multi-session conversation and timeline models
- Evaluation result models

Having all schemas in one place:
1. Eliminates duplicate definitions
2. Makes it easy to understand the data flow
3. Simplifies imports across the codebase
"""

from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Multi-Session Models (V3 - Life Story Based Generation)
# =============================================================================


@dataclass
class ExpandedPersona:
    """A persona expanded across 3 life circumstance domains.

    Life circumstances describe who the person is and what their situation is —
    not their behaviors or habits (those are captured as preferences).

    Attributes:
        base_persona: The original basic persona description
        age: Person's age
        gender: Person's gender (male/female/non-binary)
        location: City and state/country
        work_education: 3-5 circumstances about job, career, education, commute
        health_wellness: 3-5 circumstances about medical history, fitness level, conditions
        family_relationships: 3-5 circumstances about family structure, partner, living situation
        baseline_preferences: Behavioral patterns grouped by domain (5 per domain, 25 total)
            Domains: work_habits, health_body, social_relationships, leisure_hobbies, daily_routines
    """

    base_persona: str
    age: int
    gender: str
    location: str
    work_education: list[str]
    health_wellness: list[str]
    family_relationships: list[str]
    baseline_preferences: dict[str, list[str]] | None = None  # {domain: [pref1, pref2, ...]}

    def __post_init__(self):
        if self.baseline_preferences is None:
            self.baseline_preferences = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_persona": self.base_persona,
            "age": self.age,
            "gender": self.gender,
            "location": self.location,
            "work_education": self.work_education,
            "health_wellness": self.health_wellness,
            "family_relationships": self.family_relationships,
            "baseline_preferences": self.baseline_preferences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpandedPersona":
        return cls(
            base_persona=data.get("base_persona", ""),
            age=data.get("age", 0),
            gender=data.get("gender", ""),
            location=data.get("location", ""),
            work_education=data.get("work_education", []),
            health_wellness=data.get("health_wellness", []),
            family_relationships=data.get("family_relationships", []),
            baseline_preferences=data.get("baseline_preferences", {}),
        )

    def to_full_description(self, include_preferences: bool = True) -> str:
        """Generate a full prose description of the persona for use in prompts.

        Args:
            include_preferences: Whether to include baseline preferences (default True).

        Returns:
            Multi-line description with life circumstances and optionally preferences
        """
        lines = [
            f"Base: {self.base_persona}",
            f"Demographics: {self.age} year old {self.gender} in {self.location}",
            "",
            "Work & Education:",
            *[f"  - {fact}" for fact in self.work_education],
            "",
            "Health & Wellness:",
            *[f"  - {fact}" for fact in self.health_wellness],
            "",
            "Family & Relationships:",
            *[f"  - {fact}" for fact in self.family_relationships],
        ]
        if include_preferences and self.baseline_preferences:
            lines.append("")
            lines.append("Baseline Preferences:")
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
        domain: Life domain (work_habits, health_body, social_relationships, leisure_hobbies, daily_routines)
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


@dataclass
class PreferenceTimeline:
    """Central store tracking all preferences across sessions.

    This is the source of truth for preference state. It maintains
    the full history of preferences and their evolution over time.

    Attributes:
        preferences: Dict mapping preference_id to Preference
        next_id: Counter for generating unique preference IDs
    """

    preferences: dict[str, Preference] = field(default_factory=dict)
    next_id: int = 1

    def add_preference(
        self,
        fact: str,
        domain: str,
        session_id: int,
        date: str,
    ) -> str:
        """Add a new preference and return its ID."""
        pref_id = f"pref_{self.next_id:03d}"
        self.next_id += 1
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
        new_id = f"pref_{self.next_id:03d}"
        self.next_id += 1
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

    def drop_preference(self, pref_id: str, session_id: int, reason: str = "") -> None:
        """Mark a preference as dropped (no replacement).

        Used when the underlying life circumstance no longer exists,
        making the preference invalid rather than evolved.
        """
        pref = self.preferences.get(pref_id)
        if not pref:
            raise ValueError(f"Preference {pref_id} not found")
        pref.superseded_at_session = session_id
        pref.superseded_by = None
        pref.reason_for_change = reason

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
        dropped_preference_ids: IDs of preferences dropped (circumstance no longer exists)
    """

    session_id: int
    life_event: LifeEvent
    conversation: list[dict[str, str]]  # [{role, content}, ...]
    active_preference_ids: list[str]
    new_preference_ids: list[str] = field(default_factory=list)
    evolved_preference_ids: dict[str, str] = field(default_factory=dict)  # old_id -> new_id
    dropped_preference_ids: list[str] = field(default_factory=list)


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

        # Build dropped preferences
        dropped = [
            {"id": p.preference_id, "fact": p.fact, "domain": p.domain, "reason": p.reason_for_change or ""}
            for pid in session.dropped_preference_ids
            if (p := self.timeline.preferences.get(pid))
        ]

        return {
            "session_id": session.session_id,
            "life_event": {
                "date": session.life_event.date,
                "event": session.life_event.event,
                "domain": session.life_event.domain,
            },
            "preferences": {"created": created, "evolved": evolved, "dropped": dropped},
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

        result: dict[str, Any] = {
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
            timeline.next_id = max(timeline.next_id, int(pref_id.split("_")[1]) + 1)

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
            timeline.next_id = max(timeline.next_id, int(pref_id.split("_")[1]) + 1)

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

    required_preferences: list[dict[str, Any]]  # [{id, fact, supersedes?: {id, fact}}, ...]

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
class EvaluationTaskSpec:
    """Minimal evaluation task specification for multi-session preference recall.

    Contains only the task ID, rubric (which preferences to test), and the
    base persona string. Preference selection is random; the user simulator
    drives the conversation from its own opening message.

    Attributes:
        task_id: Unique identifier (e.g., "eval_a1b2c3d4")
        rubric: The evaluation rubric (contains required_preferences with supersedes info)
        persona: The base persona string from MultiSessionOutput.persona
    """

    task_id: str
    rubric: EvaluationRubric
    persona: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rubric": self.rubric.to_dict(),
            "persona": self.persona,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationTaskSpec":
        return cls(
            task_id=data["task_id"],
            rubric=EvaluationRubric.from_dict(data["rubric"]),
            persona=data["persona"],
        )


@dataclass
class MultiSessionEvaluationResult:
    """Result from evaluating an agent on multi-session data.

    Attributes:
        task_id: Which task was evaluated
        conversation: Full conversation transcript
        preference_usage: How each current preference was handled
        stale_preference_usage: Which stale preferences the agent incorrectly used
        stale_count: Number of stale (outdated) preferences used
        recalled_count: Number of preferences recalled by the agent
        preference_score: Score based on preference usage (stale penalty integrated)
        reasoning: Judge's overall reasoning
        error: Error message if evaluation failed (None if successful)
    """

    task_id: str
    conversation: list[dict[str, str]]
    preference_usage: dict[str, str]  # pref_id -> "recalled" | "missed"
    stale_preference_usage: list[str]  # List of stale pref_ids that were incorrectly used
    preference_verdicts: list[dict[str, Any]] | None = None  # Per-preference recall verdicts from judge
    stale_count: int = 0
    recalled_count: int = 0
    preference_score: float = 0.0
    eval_seconds: float | None = None
    reasoning: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "scores": {
                "preference_score": self.preference_score,
                "eval_seconds": self.eval_seconds,
            },
        }
        result.update(
            {
                "conversation": self.conversation,
                "preference_scoring": {
                    "recalled_count": self.recalled_count,
                    "stale_count": self.stale_count,
                    "stale_preference_usage": self.stale_preference_usage,
                    "preference_verdicts": self.preference_verdicts,
                },
            }
        )
        if self.error:
            result["error"] = self.error
        return result
