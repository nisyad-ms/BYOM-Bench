"""
Multi-session data generator (V3).

This generator creates realistic multi-session conversations where user
preferences evolve organically over time due to life events.

Key differences from V2:
- Generates a life story (sequence of events) for the persona
- Creates multiple conversation sessions, one per life event
- Tracks preference evolution across sessions with a central timeline
- Preferences change naturally due to life events, not within a single session

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
DEFAULT_NUM_PREFERENCES = 3


class MultiSessionGenerator(BaseDataGenerator):
    """Generate multi-session conversation data with preference evolution.

    This generator creates realistic user journeys where preferences
    evolve over time due to life events. Each session represents a
    conversation that happens after a significant life event.

    Attributes:
        persona: User persona description
        num_sessions: Number of sessions to generate (default: 2)
        num_preferences: Initial number of preferences (default: 3)
        start_date: Starting date for the timeline
        output_dir: Directory for output files
    """

    def __init__(
        self,
        persona: str,
        num_sessions: int = DEFAULT_NUM_SESSIONS,
        num_preferences: int = DEFAULT_NUM_PREFERENCES,
        start_date: str | None = None,
        output_dir: str | None = None,
    ):
        """Initialize MultiSession generator.

        Args:
            persona: User persona description (required)
            num_sessions: Number of sessions to generate
            num_preferences: Initial number of preferences
            start_date: Start date (MM/DD/YYYY), defaults to today
            output_dir: Output directory (optional)
        """
        # We don't need a topic - it's inferred from persona
        super().__init__(topic="", output_dir=output_dir)
        self.persona = persona
        self.num_sessions = num_sessions
        self.num_preferences = num_preferences
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

        Returns:
            ExpandedPersona with facts across all domains
        """
        prompt = render_prompt(
            "data_generation/multisession/expand_persona",
            persona=self.persona,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=1500, temperature=0.8)

            if not isinstance(result, dict):
                raise GenerationError(f"Unexpected response type: {type(result)}")

            expanded = ExpandedPersona.from_dict(result)
            logger.info(
                f"Expanded persona: {expanded.age}yo {expanded.gender} in {expanded.location}"
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
        # Get domain-specific facts from expanded persona
        domain_facts = expanded_persona.get_domain_facts(domain)
        domain_facts_str = "\n".join(f"- {fact}" for fact in domain_facts) if domain_facts else "No specific facts available"

        # Format previous events for context
        if previous_events:
            previous_events_str = "\n".join(
                f"- Event {e.session_id + 1}: {e.event}" for e in previous_events
            )
        else:
            previous_events_str = "None (this is the first event)"

        prompt = render_prompt(
            "data_generation/multisession/generate_life_story",
            persona=expanded_persona.to_full_description(),
            domain=domain,
            domain_facts=domain_facts_str,
            previous_events=previous_events_str,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=500, temperature=0.8)

            if not isinstance(result, dict):
                raise GenerationError(f"Unexpected response type: {type(result)}")

            return LifeEvent(
                session_id=session_id,
                date=self.start_date,  # Could enhance to add time offsets
                event=result.get("event", ""),
                context=f"[{domain}] {result.get('context', '')}",
            )

        except Exception as e:
            logger.error(f"Failed to generate life event for domain {domain}: {e}")
            raise GenerationError(f"Life event generation failed: {e}") from e

    def _generate_initial_preferences(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        expanded_persona: ExpandedPersona,
    ) -> list[str]:
        """Generate initial preferences for session 0.

        Args:
            life_event: The first life event
            timeline: PreferenceTimeline to add preferences to
            expanded_persona: The expanded persona with domain facts

        Returns:
            List of preference IDs that were created
        """
        prompt = render_prompt(
            "data_generation/multisession/generate_initial_preferences",
            persona=expanded_persona.to_full_description(),
            life_event=f"{life_event.event}\nContext: {life_event.context}",
            event_date=life_event.date,
            num_preferences=self.num_preferences,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=800, temperature=0.7)

            # Handle response format
            if isinstance(result, dict):
                prefs_data = result.get("preferences", [])
            elif isinstance(result, list):
                prefs_data = result
            else:
                raise GenerationError(f"Unexpected response type: {type(result)}")

            created_ids = []
            for p in prefs_data:
                pref_id = timeline.add_preference(
                    fact=p.get("fact", ""),
                    category=p.get("category", "general"),
                    session_id=0,
                    date=life_event.date,
                )
                created_ids.append(pref_id)

            logger.info(f"Generated {len(created_ids)} initial preferences")
            return created_ids

        except Exception as e:
            logger.error(f"Failed to generate initial preferences: {e}")
            raise GenerationError(f"Initial preference generation failed: {e}") from e

    def _evolve_preferences(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        session_id: int,
        expanded_persona: ExpandedPersona,
    ) -> dict[str, str]:
        """Evolve preferences based on a new life event.

        Evolution is event-driven: the model decides which (0 or more)
        preferences naturally evolve given the life event.

        Args:
            life_event: The new life event
            timeline: PreferenceTimeline with current preferences
            session_id: Current session number
            expanded_persona: The expanded persona with domain facts

        Returns:
            Mapping of {old_preference_id: new_preference_id} for evolved preferences
        """
        # Pass ALL active preferences to the prompt
        active_prefs = timeline.get_active_preferences()
        current_prefs_json = json.dumps(
            [{"preference_id": p.preference_id, "fact": p.fact, "category": p.category}
             for p in active_prefs],
            indent=2,
            ensure_ascii=False,
        )

        prompt = render_prompt(
            "data_generation/multisession/evolve_preferences",
            persona=expanded_persona.to_full_description(),
            life_event=f"{life_event.event}\nContext: {life_event.context}",
            event_date=life_event.date,
            current_preferences=current_prefs_json,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=1000, temperature=0.7)

            if not isinstance(result, dict):
                result = {"evolutions": []}

            # Handle new response format with evolutions array
            evolutions = result.get("evolutions", [])
            unchanged_count = result.get("unchanged_count", 0)

            evolved_mapping: dict[str, str] = {}

            # Build lookup from original preference text to ID
            pref_text_to_id = {p.fact: p.preference_id for p in active_prefs}

            # Process evolutions (new format uses original_preference/evolved_preference)
            for evo in evolutions:
                # Support both old and new format
                original_text = evo.get("original_preference", "")
                evolved_text = evo.get("evolved_preference", evo.get("new_fact", ""))
                reason = evo.get("reason", "")

                # Find the preference ID from the original text
                old_id = evo.get("old_preference_id", pref_text_to_id.get(original_text, ""))

                if old_id and evolved_text and old_id in timeline.preferences:
                    try:
                        new_id = timeline.evolve_preference(
                            old_id=old_id,
                            new_fact=evolved_text,
                            session_id=session_id,
                            date=life_event.date,
                            reason=reason,
                        )
                        evolved_mapping[old_id] = new_id
                        logger.info(f"Evolved {old_id} -> {new_id}: {reason}")
                    except ValueError as e:
                        logger.warning(f"Failed to evolve preference: {e}")
                elif original_text and not old_id:
                    logger.warning(f"Could not find preference matching: {original_text[:50]}...")

            logger.info(
                f"Evolution complete: {len(evolved_mapping)} evolved, "
                f"{unchanged_count} unchanged"
            )
            return evolved_mapping

        except Exception as e:
            logger.error(f"Failed to evolve preferences: {e}")
            # Return empty evolution on failure (preferences stay the same)
            return {}

    def _generate_session_preferences(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        session_id: int,
        expanded_persona: ExpandedPersona,
    ) -> list[str]:
        """Generate new preferences for a session (session > 0).

        Creates new preferences relevant to the life event, distinct from evolution.
        This ensures each session introduces fresh preferences tied to its event.

        Args:
            life_event: The life event for this session
            timeline: PreferenceTimeline with current state
            session_id: Current session number
            expanded_persona: The expanded persona with domain facts

        Returns:
            List of new preference IDs that were created
        """
        # Get current active preferences to avoid duplication
        active_prefs = timeline.get_active_preferences()
        current_prefs_json = json.dumps(
            [{"preference_id": p.preference_id, "fact": p.fact, "category": p.category}
             for p in active_prefs],
            indent=2,
            ensure_ascii=False,
        )

        prompt = render_prompt(
            "data_generation/multisession/generate_session_preferences",
            persona=expanded_persona.to_full_description(),
            life_event=f"{life_event.event}\nContext: {life_event.context}",
            event_date=life_event.date,
            current_preferences=current_prefs_json,
            num_preferences=self.num_preferences,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=800, temperature=0.7)

            # Handle response format
            if isinstance(result, dict):
                prefs_data = result.get("preferences", [])
            elif isinstance(result, list):
                prefs_data = result
            else:
                raise GenerationError(f"Unexpected response type: {type(result)}")

            created_ids = []
            for p in prefs_data:
                pref_id = timeline.add_preference(
                    fact=p.get("fact", ""),
                    category=p.get("category", "general"),
                    session_id=session_id,
                    date=life_event.date,
                )
                created_ids.append(pref_id)

            logger.info(f"Generated {len(created_ids)} new preferences for session {session_id}")
            return created_ids

        except Exception as e:
            logger.error(f"Failed to generate session preferences: {e}")
            # Return empty on failure
            return []

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
            [{"preference_id": p.preference_id, "fact": p.fact, "category": p.category}
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
            "data_generation/multisession/generate_session_conversation",
            persona=expanded_persona.to_full_description(),
            life_event=f"{life_event.event}\nContext: {life_event.context}",
            event_date=life_event.date,
            active_preferences=active_prefs_json,
            evolved_preferences=evolved_prefs_json,
            session_id=session_id,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=2000, temperature=0.8)

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

        # Initialize timeline
        timeline = PreferenceTimeline()
        sessions: list[Session] = []

        # Step 3: Process each session
        for idx, event in enumerate(life_events):
            logger.info(f"Step 3.{idx}: Processing session {idx}...")

            if idx == 0:
                # First session: generate initial preferences only
                new_pref_ids = self._generate_initial_preferences(event, timeline, expanded_persona)
                evolved_mapping: dict[str, str] = {}
            else:
                # Later sessions: first evolve existing, then create new
                # Order matters: evolution considers pre-existing preferences,
                # new preferences are created aware of the evolved state
                evolved_mapping = self._evolve_preferences(
                    event, timeline, idx, expanded_persona
                )
                new_pref_ids = self._generate_session_preferences(
                    event, timeline, idx, expanded_persona
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
            f"{len(timeline.preferences)} total preferences"
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
                topic=p.category,
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
    num_preferences: int = DEFAULT_NUM_PREFERENCES,
    output_dir: str | None = None,
    domains: list[str] | None = None,
) -> MultiSessionOutput:
    """Generate multi-session data with preference evolution.

    Args:
        persona: User persona description
        num_sessions: Number of sessions to generate
        num_preferences: Initial number of preferences
        output_dir: Output directory
        domains: Optional list of domains to use for life events

    Returns:
        MultiSessionOutput with full timeline and sessions
    """
    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=num_sessions,
        num_preferences=num_preferences,
        output_dir=output_dir,
    )
    return generator.generate_multi_session(domains=domains)
