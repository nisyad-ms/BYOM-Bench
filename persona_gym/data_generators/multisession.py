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
import logging
import random
from datetime import datetime

from persona_gym.client import LLMClient, get_client
from persona_gym.data_generators.base import BaseDataGenerator, GenerationError
from persona_gym.prompts import render_prompt
from persona_gym.schemas import (
    DataGenerationMetadata,
    DataGenerationOutput,
    ExpandedPersona,
    LifeEvent,
    MultiSessionOutput,
    PreferenceTimeline,
    Session,
)

logger = logging.getLogger(__name__)

# Life domains for event generation
LIFE_DOMAINS = [
    "work_education",
    "health_wellness",
    "travel_experiences",
    "relationships_personal",
    "hobbies_interests",
]

# Default configuration
DEFAULT_NUM_SESSIONS = 2


class MultiSessionGenerator(BaseDataGenerator):
    """Generate multi-session conversation data with preference evolution.

    This generator creates realistic user journeys where preferences
    evolve over time due to life events. Each session represents a
    conversation that happens after a significant life event.

    Attributes:
        persona: User persona description
        num_sessions: Number of sessions to generate (default: 2)
        start_date: Starting date for the timeline
        output_dir: Directory for output files
    """

    def __init__(
        self,
        persona: str,
        num_sessions: int = DEFAULT_NUM_SESSIONS,
        start_date: str | None = None,
        output_dir: str | None = None,
    ):
        """Initialize MultiSession generator.

        Args:
            persona: User persona description (required)
            num_sessions: Number of sessions to generate
            start_date: Start date (MM/DD/YYYY), defaults to today
            output_dir: Output directory (optional)
        """
        # We don't need a topic - it's inferred from persona
        super().__init__(topic="", output_dir=output_dir)
        self.persona = persona
        self.num_sessions = num_sessions
        self.start_date = start_date or datetime.now().strftime("%m/%d/%Y")
        self._llm: LLMClient | None = None

    @property
    def llm(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm is None:
            self._llm = get_client()
        return self._llm

    def _expand_persona(self) -> ExpandedPersona:
        """Expand the basic persona into a rich character with details across all life domains.

        Also generates 25 baseline preferences (5 per domain) that represent
        the person's core personality before any life events.

        Returns:
            ExpandedPersona with facts across all domains and baseline_preferences
        """
        prompt = render_prompt(
            "data_generation/multisession/expand_persona_instruction",
            persona=self.persona,
        )

        try:
            # Increased token limit to accommodate 25 baseline preferences
            result = self.llm.complete_json(prompt, max_tokens=8000)

            if not isinstance(result, dict):
                raise GenerationError(f"Unexpected response type: {type(result)}")

            expanded = ExpandedPersona.from_dict(result)
            baseline_count = len(expanded.baseline_preferences) if expanded.baseline_preferences else 0
            logger.info(
                f"Expanded persona: {expanded.age}yo {expanded.gender} in {expanded.location}, "
                f"{baseline_count} baseline preferences"
            )
            return expanded

        except Exception as e:
            logger.error(f"Failed to expand persona: {e}")
            raise GenerationError(f"Persona expansion failed: {e}") from e

    def _generate_life_events(
        self,
        expanded_persona: ExpandedPersona,
        domains: list[str] | None = None,
    ) -> list[LifeEvent]:
        """Generate a sequence of life events for the persona.

        Args:
            expanded_persona: The expanded persona with domain facts
            domains: List of domains to use, or None to randomly sample from LIFE_DOMAINS

        Returns:
            List of LifeEvent objects in chronological order
        """
        if domains is None:
            # Randomly sample domains (with replacement) for variety
            domains = random.choices(LIFE_DOMAINS, k=self.num_sessions)

        events: list[LifeEvent] = []
        for idx, domain in enumerate(domains[:self.num_sessions]):
            # Pass previous events to ensure coherent narrative
            event = self._generate_single_life_event(idx, domain, expanded_persona, events)
            events.append(event)

        logger.info(f"Generated {len(events)} life events")
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
            expanded_persona: The expanded persona with domain facts
            previous_events: List of previously generated events for context

        Returns:
            LifeEvent with domain stored in context
        """
        if previous_events:
            previous_events_str = "\n".join(
                f"- Event {e.session_id + 1}: {e.event}" for e in previous_events
            )
        else:
            previous_events_str = "None (this is the first event)"

        prompt = render_prompt(
            "data_generation/multisession/generate_life_story_instruction",
            persona=expanded_persona.to_full_description(),
            domain=domain,
            previous_events=previous_events_str,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=500)

            if not isinstance(result, dict):
                raise GenerationError(f"Unexpected response type: {type(result)}")

            return LifeEvent(
                session_id=session_id,
                date=self.start_date,
                event=result.get("event", ""),
                domain=domain,
            )

        except Exception as e:
            logger.error(f"Failed to generate life event for domain {domain}: {e}")
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
            logger.warning("No baseline preferences found in expanded persona")
            return []

        baseline_ids = []
        # Use a date before the start_date to indicate these are pre-existing
        baseline_date = "01/01/2000"  # Marker date for baseline preferences

        for domain, prefs in expanded_persona.baseline_preferences.items():
            for pref in prefs:
                pref_id = timeline.add_preference(
                    fact=pref,
                    domain=domain,
                    session_id=-1,  # -1 indicates baseline (pre-event)
                    date=baseline_date,
                )
                baseline_ids.append(pref_id)

        logger.info(f"Loaded {len(baseline_ids)} baseline preferences into timeline")
        return baseline_ids

    def _format_domain_facts(self, expanded_persona: ExpandedPersona) -> str:
        """Format all domain facts for prompt inclusion."""
        sections = []
        domains = [
            ("Work & Education", expanded_persona.work_and_education),
            ("Health & Wellness", expanded_persona.health_and_wellness),
            ("Travel & Experiences", expanded_persona.travel_and_experiences),
            ("Relationships & Personal", expanded_persona.relationships_and_personal),
            ("Hobbies & Interests", expanded_persona.hobbies_and_interests),
        ]
        for name, facts in domains:
            if facts:
                facts_str = "\n".join(f"  - {f}" for f in facts)
                sections.append(f"{name}:\n{facts_str}")
        return "\n\n".join(sections) if sections else "No domain facts available"

    def _format_evolution_history(self, timeline: PreferenceTimeline) -> str:
        """Format the complete preference evolution history for prompt inclusion."""
        superseded = [p for p in timeline.preferences.values() if not p.is_active]
        if not superseded:
            return "No evolutions yet (all preferences are original)"

        history = []
        for old_pref in superseded:
            new_pref = timeline.preferences.get(old_pref.superseded_by)
            if new_pref:
                history.append({
                    "session": old_pref.superseded_at_session,
                    "from_id": old_pref.preference_id,
                    "from": old_pref.fact,
                    "to_id": new_pref.preference_id,
                    "to": new_pref.fact,
                    "reason": old_pref.reason_for_change or "Not specified",
                })

        return json.dumps(history, indent=2, ensure_ascii=False)

    def _format_previous_events(self, life_events: list[LifeEvent], current_session: int) -> str:
        """Format all previous life events for prompt inclusion."""
        previous = [e for e in life_events if e.session_id < current_session]
        if not previous:
            return "None (this is the first event)"

        lines = []
        for e in previous:
            lines.append(f"- Session {e.session_id} ({e.date}): {e.event}")
        return "\n".join(lines)

    def _update_preferences(
        self,
        life_event: LifeEvent,
        all_events: list[LifeEvent],
        timeline: PreferenceTimeline,
        session_id: int,
        expanded_persona: ExpandedPersona,
    ) -> tuple[dict[str, str], list[str]]:
        """Unified method to evolve existing preferences and create new ones.

        This method replaces the separate _evolve_preferences and _generate_event_preferences
        methods. It provides the LLM with full context including:
        - All previous life events
        - Complete preference evolution history
        - All domain facts
        - All currently active preferences

        Args:
            life_event: The current life event being processed
            all_events: All life events (for history context)
            timeline: PreferenceTimeline with current state
            session_id: Current session number
            expanded_persona: The expanded persona with domain facts

        Returns:
            Tuple of (evolved_mapping, new_pref_ids):
            - evolved_mapping: {old_preference_id: new_preference_id} for evolved preferences
            - new_pref_ids: List of newly created preference IDs
        """
        # Format all context for the prompt
        active_prefs = timeline.get_active_preferences()
        active_prefs_json = json.dumps(
            [{"preference_id": p.preference_id, "fact": p.fact, "domain": p.domain}
             for p in active_prefs],
            indent=2,
            ensure_ascii=False,
        )

        evolution_history_str = self._format_evolution_history(timeline)
        previous_events_str = self._format_previous_events(all_events, session_id)
        current_event_str = life_event.event

        prompt = render_prompt(
            "data_generation/multisession/update_preferences_instruction",
            persona=expanded_persona.to_full_description(),
            current_event=current_event_str,
            event_date=life_event.date,
            previous_events=previous_events_str,
            active_preferences=active_prefs_json,
            evolution_history=evolution_history_str,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=2000)

            if not isinstance(result, dict):
                result = {"evolutions": [], "new_preferences": []}

            # Process evolutions
            evolutions = result.get("evolutions", [])
            evolved_mapping: dict[str, str] = {}

            # Build lookup from preference text to ID (fallback for matching)
            pref_text_to_id = {p.fact: p.preference_id for p in active_prefs}

            for evo in evolutions:
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
                            logger.info(f"Evolved {old_id} -> {new_id}: {reason[:50]}...")
                        except ValueError as e:
                            logger.warning(f"Failed to evolve preference {old_id}: {e}")
                    else:
                        logger.warning(f"Preference {old_id} already superseded, skipping")

            # Process new preferences
            new_prefs_data = result.get("new_preferences", [])
            new_pref_ids: list[str] = []

            for p in new_prefs_data:
                # Use 'domain' field (new format) or fall back to 'category' (old format)
                pref_domain = p.get("domain") or p.get("category", "general")
                pref_id = timeline.add_preference(
                    fact=p.get("fact", ""),
                    domain=pref_domain,
                    session_id=session_id,
                    date=life_event.date,
                )
                new_pref_ids.append(pref_id)

            # Log analysis if provided
            analysis = result.get("analysis", {})
            rationale = analysis.get("rationale", "")
            if rationale:
                logger.info(f"Update rationale: {rationale[:100]}...")

            logger.info(
                f"Session {session_id}: {len(evolved_mapping)} evolved, "
                f"{len(new_pref_ids)} new preferences created"
            )

            return evolved_mapping, new_pref_ids

        except Exception as e:
            logger.error(f"Failed to update preferences for session {session_id}: {e}")
            # Return empty results on failure
            return {}, []

    def _generate_session_conversation(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        session_id: int,
        evolved_mapping: dict[str, str],
        expanded_persona: ExpandedPersona,
    ) -> list[dict[str, str]]:
        """Generate a conversation for a session.

        Args:
            life_event: The life event for this session
            timeline: PreferenceTimeline with current state
            session_id: Current session number
            evolved_mapping: Mapping of old_id -> new_id for recently evolved prefs
            expanded_persona: The expanded persona with domain facts

        Returns:
            List of conversation turns [{role, content}, ...]
        """
        # Get active preferences
        active_prefs = timeline.get_active_at_session(session_id)
        active_prefs_json = json.dumps(
            [{"preference_id": p.preference_id, "fact": p.fact, "domain": p.domain}
             for p in active_prefs],
            indent=2,
        )

        # Get evolved preferences (the new ones that replaced old ones)
        evolved_prefs = []
        for old_id, new_id in evolved_mapping.items():
            old_pref = timeline.preferences.get(old_id)
            new_pref = timeline.preferences.get(new_id)
            if old_pref and new_pref:
                evolved_prefs.append({
                    "old_fact": old_pref.fact,
                    "new_fact": new_pref.fact,
                    "reason": new_pref.reason_for_change or old_pref.reason_for_change or "",
                })
        evolved_prefs_json = json.dumps(evolved_prefs, indent=2) if evolved_prefs else "None"

        prompt = render_prompt(
            "data_generation/multisession/generate_session_conversation_instruction",
            persona=expanded_persona.to_full_description(),
            life_event=life_event.event,
            event_date=life_event.date,
            active_preferences=active_prefs_json,
            evolved_preferences=evolved_prefs_json,
            session_id=session_id,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=2000)

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

            logger.info(f"Generated conversation with {len(normalized)} turns")
            return normalized

        except Exception as e:
            logger.error(f"Failed to generate conversation: {e}")
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
        logger.info(f"Starting multi-session generation for persona: {self.persona[:50]}...")

        # Step 1: Expand the basic persona into a rich character
        logger.info("Step 1: Expanding persona across life domains...")
        expanded_persona = self._expand_persona()

        # Step 2: Generate life events (distributed across domains)
        logger.info("Step 2: Generating life events...")
        life_events = self._generate_life_events(expanded_persona, domains)
        logger.info(f"Generated {len(life_events)} life events")

        # Initialize timeline and load baseline preferences
        timeline = PreferenceTimeline()

        # Step 2.5: Load baseline preferences (core personality traits)
        logger.info("Step 2.5: Loading baseline preferences...")
        baseline_ids = self._load_baseline_preferences(timeline, expanded_persona)
        logger.info(f"Loaded {len(baseline_ids)} baseline preferences")

        sessions: list[Session] = []

        # Step 3: Process each session
        # UNIFIED FLOW: Single call to update preferences (evolve + create new)
        # with full context (all events, evolution history, domain facts)
        for idx, event in enumerate(life_events):
            logger.info(f"Step 3.{idx}: Processing session {idx}...")

            # Unified preference update: evolve existing AND create new
            # This single call sees full context including:
            # - All previous life events
            # - Complete preference evolution history
            # - All domain facts
            # - All currently active preferences
            evolved_mapping, new_pref_ids = self._update_preferences(
                event, life_events, timeline, idx, expanded_persona
            )

            # Generate conversation for this session
            conversation = self._generate_session_conversation(
                event, timeline, idx, evolved_mapping, expanded_persona
            )

            # Create session record
            active_pref_ids = timeline.get_preference_ids_at_session(idx)
            session = Session(
                session_id=idx,
                life_event=event,
                conversation=conversation,
                active_preference_ids=active_pref_ids,
                new_preference_ids=new_pref_ids,
                evolved_preference_ids=evolved_mapping,
            )
            sessions.append(session)

            logger.info(
                f"Session {idx}: {len(conversation)} turns, "
                f"{len(active_pref_ids)} active prefs, "
                f"{len(evolved_mapping)} evolved"
            )

        # Build final output
        output = MultiSessionOutput(
            persona=self.persona,
            persona_id=persona_id or f"persona_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            life_events=life_events,
            timeline=timeline,
            sessions=sessions,
            generation_timestamp=datetime.now().isoformat(),
            expanded_persona=expanded_persona,
        )

        logger.info(
            f"Generation complete: {len(sessions)} sessions, "
            f"{len(timeline.preferences)} total preferences "
            f"({len(baseline_ids)} baseline)"
        )

        return output

    def generate(self) -> DataGenerationOutput:
        """Generate data in the standard pipeline format.

        This method implements the BaseDataGenerator interface by
        flattening all sessions into a single conversation.

        Note: For full multi-session data, use generate_multi_session() instead.
        """
        result = self.generate_multi_session()

        # Flatten all conversations
        all_turns = result.get_all_conversations_flat()

        # Convert current preferences to PreferenceItem format
        from persona_gym.schemas import PreferenceItem

        preferences = [
            PreferenceItem(
                fact=p.fact,
                preference_type="current" if p.is_active else "superseded",
                source_date=p.created_at_date,
                topic=p.domain,
                preference_id=p.preference_id,
                supersedes_id=None,  # Reverse lookup would be expensive
            )
            for p in result.timeline.preferences.values()
        ]

        return DataGenerationOutput(
            conversation=all_turns,
            preferences=preferences,
            metadata=DataGenerationMetadata(
                topic="multi-session",
                persona_id=result.persona_id,
                timestamp=result.generation_timestamp,
            ),
        )


# Convenience function for CLI usage
def generate_multi_session(
    persona: str,
    num_sessions: int = DEFAULT_NUM_SESSIONS,
    output_dir: str | None = None,
    domains: list[str] | None = None,
) -> MultiSessionOutput:
    """Generate multi-session data with preference evolution.

    Args:
        persona: User persona description
        num_sessions: Number of sessions to generate
        output_dir: Output directory
        domains: Optional list of domains to use for life events

    Returns:
        MultiSessionOutput with full timeline and sessions
    """
    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=num_sessions,
        output_dir=output_dir,
    )
    return generator.generate_multi_session(domains=domains)
