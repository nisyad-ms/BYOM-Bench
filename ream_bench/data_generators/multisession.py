"""
Multi-session data generator.

This generator creates realistic multi-session conversations where user
preferences evolve organically over time due to life events.

Key capabilities:
- Generates a life story (sequence of events) for the persona
- Creates multiple conversation sessions, one per life event
- Tracks preference evolution across sessions with a central timeline
- Preferences change naturally due to life events

The flow:
1. Persona → Life Story (sequence of events)
2. Event 0 → Initial Preferences → Session 0 Conversation
3. Event N → Evolve Preferences → Session N Conversation
4. Output: Complete timeline with all sessions and preference history
"""

import json
import random
from datetime import datetime

from byom_bench.client import CONFIG, PIPELINE_CONFIG, LLMClient, PooledLLMClient
from byom_bench.prompts import render_prompt
from byom_bench.schemas import (
    ExpandedPersona,
    LifeEvent,
    MultiSessionOutput,
    Preference,
    PreferenceTimeline,
    Session,
)

# Life event domains — fundamental life areas where lasting change originates.
# These match the life facts domains, not the preference domains.
LIFE_DOMAINS = [
    "work_education",
    "health_wellness",
    "family_relationships",
]

# Default configuration
DEFAULT_NUM_SESSIONS: int = PIPELINE_CONFIG["data_generation"]["default_num_sessions"]


def _active_prefs_to_json(prefs: list[Preference]) -> str:
    """Serialize active preferences to JSON for prompt inclusion."""
    return json.dumps(
        [{"preference_id": p.preference_id, "fact": p.fact, "domain": p.domain} for p in prefs],
        indent=2,
        ensure_ascii=False,
    )


class GenerationError(Exception):
    """Raised when data generation fails."""


class MultiSessionGenerator:
    """Generate multi-session conversation data with preference evolution.

    This generator creates realistic user journeys where preferences
    evolve over time due to life events. Each session represents a
    conversation that happens after a significant life event.

    Attributes:
        persona: User persona description
        num_sessions: Number of sessions to generate (default: 2)
        start_date: Starting date for the timeline
    """

    def __init__(
        self,
        persona: str,
        llm: LLMClient | PooledLLMClient | None = None,
        num_sessions: int = DEFAULT_NUM_SESSIONS,
        start_date: str | None = None,
    ):
        """Initialize MultiSession generator.

        Args:
            persona: User persona description (required)
            llm: LLMClient instance (optional, creates one if not provided)
            num_sessions: Number of sessions to generate
            start_date: Start date (MM/DD/YYYY), defaults to today
        """
        self.persona = persona
        self.num_sessions = num_sessions
        self.start_date = start_date or datetime.now().strftime("%m/%d/%Y")
        self._llm = llm

    @property
    def llm(self) -> LLMClient | PooledLLMClient:
        """Lazy-load LLM client."""
        if self._llm is None:
            self._llm = PooledLLMClient()
        return self._llm

    def _expand_persona(self) -> ExpandedPersona:
        """Expand the basic persona into a rich character with life facts and preferences.

        Makes two LLM calls:
        1. Generate life facts (biography across 5 domains)
        2. Generate 25 baseline preferences from those life facts

        Returns:
            ExpandedPersona with facts across all domains and baseline_preferences
        """
        # --- Call 1: Life facts ---
        gender = random.choice(["male", "female"])
        life_facts_system = render_prompt("data_generation/multisession/expand_life_facts_system")
        life_facts_user = render_prompt(
            "data_generation/multisession/expand_life_facts_user", persona=self.persona, gender=gender
        )

        try:
            facts_result = self.llm.complete_json(
                life_facts_user,
                system_prompt=life_facts_system,
                max_tokens=CONFIG["max_tokens"]["expand_persona"],
            )
            if not isinstance(facts_result, dict):
                raise GenerationError(f"Life facts: unexpected response type: {type(facts_result)}")
        except Exception as e:
            raise GenerationError(f"Life facts generation failed: {e}") from e

        # Build partial ExpandedPersona (no preferences yet) for to_full_description()
        facts_result["baseline_preferences"] = {}
        expanded = ExpandedPersona.from_dict(facts_result)

        # --- Call 2: Baseline preferences ---
        prefs_system = render_prompt("data_generation/multisession/generate_baseline_preferences_system")
        prefs_user = render_prompt(
            "data_generation/multisession/generate_baseline_preferences_user",
            persona=expanded.to_full_description(),
        )

        try:
            prefs_result = self.llm.complete_json(
                prefs_user,
                system_prompt=prefs_system,
                max_tokens=CONFIG["max_tokens"]["expand_persona"],
            )
            if not isinstance(prefs_result, dict):
                raise GenerationError(f"Baseline preferences: unexpected response type: {type(prefs_result)}")
        except Exception as e:
            raise GenerationError(f"Baseline preferences generation failed: {e}") from e

        expanded.baseline_preferences = prefs_result.get("baseline_preferences", {})
        return expanded

    def _generate_life_events(
        self,
        expanded_persona: ExpandedPersona,
        domains: list[str] | None = None,
    ) -> list[LifeEvent]:
        """Generate a sequence of life events for the persona.

        Args:
            expanded_persona: The expanded persona with life facts
            domains: List of domains to use, or None to randomly sample from LIFE_DOMAINS

        Returns:
            List of LifeEvent objects in chronological order
        """
        if domains is None:
            # Equal sampling: each domain gets floor(n/k) slots, remainder filled randomly.
            # Then shuffle, rejecting sequences with 3+ consecutive same domain.
            n = self.num_sessions
            k = len(LIFE_DOMAINS)
            base_count = n // k
            remainder = n % k
            pool = LIFE_DOMAINS * base_count + random.sample(LIFE_DOMAINS, remainder)
            while True:
                random.shuffle(pool)
                # No 3+ consecutive same domain
                if len(pool) >= 3 and any(
                    pool[i] == pool[i + 1] == pool[i + 2] for i in range(len(pool) - 2)
                ):
                    continue
                break
            domains = pool

        events: list[LifeEvent] = []
        for idx, domain in enumerate(domains[: self.num_sessions]):
            print(f"    Event {idx + 1}/{self.num_sessions}: {domain}...", flush=True)
            # Pass previous events to ensure coherent narrative
            event = self._generate_single_life_event(idx, domain, expanded_persona, events)
            events.append(event)

        return events

    def _generate_single_life_event(
        self,
        session_id: int,
        domain: str,
        expanded_persona: ExpandedPersona,
        previous_events: list[LifeEvent],
    ) -> LifeEvent:
        """Generate a single life event for a specific domain.

        Args:
            session_id: The session this event corresponds to
            domain: The life domain for this event
            expanded_persona: The expanded persona with life facts
            previous_events: List of previously generated events for context

        Returns:
            LifeEvent with domain stored in context
        """
        if previous_events:
            previous_events_str = "\n".join(f"- Event {e.session_id + 1}: {e.event}" for e in previous_events)
        else:
            previous_events_str = "None (this is the first event)"

        system_prompt = render_prompt("data_generation/multisession/generate_life_event_system")
        prompt = render_prompt(
            "data_generation/multisession/generate_life_event_user",
            persona=expanded_persona.to_full_description(),
            domain=domain,
            previous_events=previous_events_str,
        )

        try:
            result = self.llm.complete_json(
                prompt,
                system_prompt=system_prompt,
                max_tokens=CONFIG["max_tokens"]["life_event"],
            )

            if not isinstance(result, dict):
                raise GenerationError(f"Unexpected response type: {type(result)}")

            return LifeEvent(
                session_id=session_id,
                date=self.start_date,
                event=result.get("event", ""),
                domain=domain,
            )

        except Exception as e:
            raise GenerationError(f"Life event generation failed: {e}") from e

    def _load_baseline_preferences(
        self,
        timeline: PreferenceTimeline,
        expanded_persona: ExpandedPersona,
    ) -> list[str]:
        """Load baseline preferences from expanded persona into the timeline.

        Baseline preferences represent the person's core personality traits
        that exist BEFORE any life events. They are added with session_id=-1
        to indicate they predate all sessions.

        Args:
            timeline: PreferenceTimeline to add preferences to
            expanded_persona: The expanded persona with baseline_preferences

        Returns:
            List of baseline preference IDs that were added
        """
        if not expanded_persona.baseline_preferences:
            return []

        baseline_ids = []
        # Use a date before the start_date to indicate these are pre-existing
        baseline_date = "01/01/2000"  # Marker date for baseline preferences

        for domain, prefs in expanded_persona.baseline_preferences.items():
            if not isinstance(prefs, list):
                continue
            for pref in prefs:
                # Handle LLM returning prefs as dicts instead of strings
                if isinstance(pref, dict):
                    pref = pref.get("fact") or pref.get("text") or pref.get("preference") or str(pref)
                elif not isinstance(pref, str):
                    pref = str(pref)
                pref_id = timeline.add_preference(
                    fact=pref,
                    domain=domain,
                    session_id=-1,  # -1 indicates baseline (pre-event)
                    date=baseline_date,
                )
                baseline_ids.append(pref_id)

        return baseline_ids

    def _format_evolution_history(self, timeline: PreferenceTimeline) -> str:
        """Format the complete preference evolution history for prompt inclusion."""
        superseded = [p for p in timeline.preferences.values() if not p.is_active]
        if not superseded:
            return "No evolutions yet (all preferences are original)"

        history = []
        for old_pref in superseded:
            if old_pref.superseded_by:
                new_pref = timeline.preferences.get(old_pref.superseded_by)
                if new_pref:
                    history.append(
                        {
                            "session": old_pref.superseded_at_session,
                            "type": "evolved",
                            "from_id": old_pref.preference_id,
                            "from": old_pref.fact,
                            "to_id": new_pref.preference_id,
                            "to": new_pref.fact,
                            "reason": old_pref.reason_for_change or "Not specified",
                        }
                    )
            else:
                history.append(
                    {
                        "session": old_pref.superseded_at_session,
                        "type": "dropped",
                        "id": old_pref.preference_id,
                        "preference": old_pref.fact,
                        "reason": old_pref.reason_for_change or "Not specified",
                    }
                )

        return json.dumps(history, indent=2, ensure_ascii=False)

    def _update_preferences(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        session_id: int,
        expanded_persona: ExpandedPersona,
    ) -> tuple[dict[str, str], list[str], list[str]]:
        """Unified method to evolve existing preferences, drop invalid ones, and create new ones.

        This method provides the LLM with full context including:
        - Complete preference evolution history
        - All life facts (without baseline preferences)
        - All currently active preferences

        Args:
            life_event: The current life event being processed
            all_events: All life events (for history context)
            timeline: PreferenceTimeline with current state
            session_id: Current session number
            expanded_persona: The expanded persona with life facts

        Returns:
            Tuple of (evolved_mapping, new_pref_ids, dropped_pref_ids):
            - evolved_mapping: {old_preference_id: new_preference_id} for evolved preferences
            - new_pref_ids: List of newly created preference IDs
            - dropped_pref_ids: List of preference IDs that were dropped
        """
        # Format all context for the prompt
        active_prefs = timeline.get_active_preferences()
        active_prefs_json = _active_prefs_to_json(active_prefs)

        evolution_history_str = self._format_evolution_history(timeline)

        system_prompt = render_prompt("data_generation/multisession/update_preferences_system")
        prompt = render_prompt(
            "data_generation/multisession/update_preferences_user",
            persona=expanded_persona.to_full_description(include_preferences=False),
            current_event=life_event.event,
            event_date=life_event.date,
            active_preferences=active_prefs_json,
            evolution_history=evolution_history_str,
        )

        try:
            result = self.llm.complete_json(
                prompt,
                system_prompt=system_prompt,
                max_tokens=CONFIG["max_tokens"]["update_preferences"],
            )

            if not isinstance(result, dict):
                result = {"evolutions": [], "drops": [], "new_preferences": []}

            # Process evolutions
            evolutions = result.get("evolutions", [])
            if not isinstance(evolutions, list):
                evolutions = []
            evolved_mapping: dict[str, str] = {}

            # Build lookup from preference text to ID (fallback for matching)
            pref_text_to_id = {p.fact: p.preference_id for p in active_prefs}

            for evo in evolutions:
                if not isinstance(evo, dict):
                    continue
                # Try to get preference_id directly, fall back to text matching
                old_id = evo.get("preference_id", "")
                if not old_id:
                    original_text = evo.get("original_preference", "")
                    old_id = pref_text_to_id.get(original_text, "")

                evolved_text = evo.get("evolved_preference", "")
                reason = evo.get("reason", "")
                # Use 'domain' field if provided (allows domain change on evolution)
                new_domain = evo.get("domain")

                if old_id and evolved_text and old_id in timeline.preferences:
                    # Check if this preference is still active (not already evolved)
                    if timeline.preferences[old_id].is_active:
                        try:
                            new_id = timeline.evolve_preference(
                                old_id=old_id,
                                new_fact=evolved_text,
                                session_id=session_id,
                                date=life_event.date,
                                reason=reason,
                                new_domain=new_domain,
                            )
                            evolved_mapping[old_id] = new_id
                        except ValueError:
                            print(f"  Skipped evolution of {old_id}: preference not found or inactive")

            # Process new preferences
            new_prefs_data = result.get("new_preferences", [])
            if not isinstance(new_prefs_data, list):
                new_prefs_data = []
            new_pref_ids: list[str] = []

            for p in new_prefs_data:
                if not isinstance(p, dict):
                    continue
                # Use 'domain' field (new format) or fall back to 'category' (old format)
                pref_domain = p.get("domain") or p.get("category", "general")
                pref_id = timeline.add_preference(
                    fact=p.get("fact", ""),
                    domain=pref_domain,
                    session_id=session_id,
                    date=life_event.date,
                )
                new_pref_ids.append(pref_id)

            # Process drops
            drops_data = result.get("drops", [])
            if not isinstance(drops_data, list):
                drops_data = []
            dropped_pref_ids: list[str] = []

            for d in drops_data:
                if not isinstance(d, dict):
                    continue
                drop_id = d.get("preference_id", "")
                if not drop_id:
                    original_text = d.get("original_preference", "")
                    drop_id = pref_text_to_id.get(original_text, "")

                reason = d.get("reason", "")
                if drop_id and drop_id in timeline.preferences:
                    if timeline.preferences[drop_id].is_active:
                        try:
                            timeline.drop_preference(drop_id, session_id, reason)
                            dropped_pref_ids.append(drop_id)
                        except ValueError:
                            print(f"  Skipped drop of {drop_id}: preference not found or inactive")

            return evolved_mapping, new_pref_ids, dropped_pref_ids

        except Exception as e:
            raise GenerationError(f"Failed to update preferences for session {session_id}: {e}") from e

    def _generate_session_conversation(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        session_id: int,
        evolved_mapping: dict[str, str],
        new_pref_ids: list[str],
        dropped_pref_ids: list[str],
        expanded_persona: ExpandedPersona,
    ) -> list[dict[str, str]]:
        """Generate a conversation for a session.

        Args:
            life_event: The life event for this session
            timeline: PreferenceTimeline with current state
            session_id: Current session number
            evolved_mapping: Mapping of old_id -> new_id for recently evolved prefs
            new_pref_ids: IDs of newly created preferences this session
            dropped_pref_ids: IDs of preferences dropped this session
            expanded_persona: The expanded persona with life facts

        Returns:
            List of conversation turns [{role, content}, ...]
        """
        # Build session delta: evolved preferences (old -> new), newly created, and dropped
        session_delta = []
        for old_id, new_id in evolved_mapping.items():
            old_pref = timeline.preferences.get(old_id)
            new_pref = timeline.preferences.get(new_id)
            if old_pref and new_pref:
                session_delta.append(
                    {
                        "type": "evolved",
                        "old": old_pref.fact,
                        "new": new_pref.fact,
                        "reason": old_pref.reason_for_change or "",
                    }
                )
        for pref_id in new_pref_ids:
            pref = timeline.preferences.get(pref_id)
            if pref:
                session_delta.append(
                    {
                        "type": "new",
                        "fact": pref.fact,
                    }
                )
        for pref_id in dropped_pref_ids:
            pref = timeline.preferences.get(pref_id)
            if pref:
                session_delta.append(
                    {
                        "type": "drop",
                        "fact": pref.fact,
                        "reason": pref.reason_for_change or "",
                    }
                )
        session_delta_json = json.dumps(session_delta, indent=2) if session_delta else "None"

        prompt = render_prompt(
            "data_generation/multisession/generate_session_conversation_user",
            persona=expanded_persona.to_full_description(),
            life_event=life_event.event,
            event_date=life_event.date,
            session_delta=session_delta_json,
            session_id=session_id,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=CONFIG["max_tokens"]["session_conversation"])

            if isinstance(result, dict):
                conversation = result.get("conversation", result.get("turns", []))
            elif isinstance(result, list):
                conversation = result
            else:
                raise GenerationError(f"Unexpected response type: {type(result)}")

            # Validate and normalize turns
            normalized = []
            for turn in conversation:
                role = turn.get("role", "").lower()
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    normalized.append({"role": role, "content": content})

            return normalized

        except Exception as e:
            raise GenerationError(f"Conversation generation failed: {e}") from e

    def generate_multi_session(
        self,
        persona_id: str = "",
        domains: list[str] | None = None,
    ) -> MultiSessionOutput:
        """Generate multi-session data with preference evolution.

        This is the primary method for generating multi-session data.
        For pipeline compatibility, use generate() instead.

        Args:
            persona_id: Optional identifier for the persona
            domains: Optional list of domains to use for life events.
                     If None, cycles through LIFE_DOMAINS.

        Returns:
            MultiSessionOutput with full timeline and sessions
        """
        print("  [1/3] Expanding persona (life facts + baseline preferences)...", flush=True)
        expanded_persona = self._expand_persona()
        print("  [2/3] Generating life events...", flush=True)
        life_events = self._generate_life_events(expanded_persona, domains)

        timeline = PreferenceTimeline()
        self._load_baseline_preferences(timeline, expanded_persona)

        sessions: list[Session] = []

        print(f"  [3/3] Generating {self.num_sessions} sessions...", flush=True)
        for idx, event in enumerate(life_events):
            print(f"    Session {idx + 1}/{self.num_sessions}: updating preferences ({event.domain})...", flush=True)
            evolved_mapping, new_pref_ids, dropped_pref_ids = self._update_preferences(
                event, timeline, idx, expanded_persona
            )

            print(f"    Session {idx + 1}/{self.num_sessions}: generating conversation...", flush=True)
            # Generate conversation for this session
            conversation = self._generate_session_conversation(event, timeline, idx, evolved_mapping, new_pref_ids, dropped_pref_ids, expanded_persona)

            # Create session record
            active_pref_ids = timeline.get_preference_ids_at_session(idx)
            session = Session(
                session_id=idx,
                life_event=event,
                conversation=conversation,
                active_preference_ids=active_pref_ids,
                new_preference_ids=new_pref_ids,
                evolved_preference_ids=evolved_mapping,
                dropped_preference_ids=dropped_pref_ids,
            )
            sessions.append(session)

        return MultiSessionOutput(
            persona=self.persona,
            persona_id=persona_id or f"persona_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            life_events=life_events,
            timeline=timeline,
            sessions=sessions,
            generation_timestamp=datetime.now().isoformat(),
            expanded_persona=expanded_persona,
        )
