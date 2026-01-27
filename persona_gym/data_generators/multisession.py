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
from datetime import datetime

from persona_gym.client import LLMClient, get_client
from persona_gym.data_generators.base import BaseDataGenerator, GenerationError
from persona_gym.prompts import render_prompt
from persona_gym.schemas import (
    DataGenerationMetadata,
    DataGenerationOutput,
    LifeEvent,
    MultiSessionOutput,
    PreferenceTimeline,
    Session,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_NUM_SESSIONS = 2
DEFAULT_NUM_PREFERENCES = 5
DEFAULT_NUM_TO_EVOLVE = 2  # How many preferences to evolve per session


class MultiSessionGenerator(BaseDataGenerator):
    """Generate multi-session conversation data with preference evolution.

    This generator creates realistic user journeys where preferences
    evolve over time due to life events. Each session represents a
    conversation that happens after a significant life event.

    Attributes:
        persona: User persona description
        num_sessions: Number of sessions to generate (default: 2)
        num_preferences: Initial number of preferences (default: 5)
        num_to_evolve: Target preferences to evolve per session (default: 2)
        start_date: Starting date for the timeline
        output_dir: Directory for output files
    """

    def __init__(
        self,
        persona: str,
        num_sessions: int = DEFAULT_NUM_SESSIONS,
        num_preferences: int = DEFAULT_NUM_PREFERENCES,
        num_to_evolve: int = DEFAULT_NUM_TO_EVOLVE,
        start_date: str | None = None,
        output_dir: str | None = None,
    ):
        """Initialize MultiSession generator.

        Args:
            persona: User persona description (required)
            num_sessions: Number of sessions to generate
            num_preferences: Initial number of preferences
            num_to_evolve: How many preferences to evolve per session
            start_date: Start date (MM/DD/YYYY), defaults to today
            output_dir: Output directory (optional)
        """
        # We don't need a topic - it's inferred from persona
        super().__init__(topic="", output_dir=output_dir)
        self.persona = persona
        self.num_sessions = num_sessions
        self.num_preferences = num_preferences
        self.num_to_evolve = num_to_evolve
        self.start_date = start_date or datetime.now().strftime("%m/%d/%Y")
        self._llm: LLMClient | None = None

    @property
    def llm(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm is None:
            self._llm = get_client()
        return self._llm

    def _generate_life_story(self) -> list[LifeEvent]:
        """Generate a sequence of life events for the persona.

        Returns:
            List of LifeEvent objects in chronological order
        """
        prompt = render_prompt(
            "data_generation/multisession/generate_life_story",
            persona=self.persona,
            num_events=self.num_sessions,
            start_date=self.start_date,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=800, temperature=0.8)

            # Handle various response formats
            if isinstance(result, dict):
                events_data = result.get("events", result.get("life_events", []))
            elif isinstance(result, list):
                events_data = result
            else:
                raise GenerationError(f"Unexpected response type: {type(result)}")

            events = []
            for e in events_data:
                events.append(
                    LifeEvent(
                        session_id=e.get("session_id", len(events)),
                        date=e.get("date", self.start_date),
                        event=e.get("event", ""),
                        context=e.get("context", ""),
                    )
                )

            if len(events) < self.num_sessions:
                logger.warning(
                    f"Generated {len(events)} events, expected {self.num_sessions}"
                )

            return events

        except Exception as e:
            logger.error(f"Failed to generate life story: {e}")
            raise GenerationError(f"Life story generation failed: {e}") from e

    def _generate_initial_preferences(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
    ) -> list[str]:
        """Generate initial preferences for session 0.

        Args:
            life_event: The first life event
            timeline: PreferenceTimeline to add preferences to

        Returns:
            List of preference IDs that were created
        """
        prompt = render_prompt(
            "data_generation/multisession/generate_initial_preferences",
            persona=self.persona,
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
    ) -> tuple[list[str], dict[str, str]]:
        """Evolve preferences based on a new life event.

        Args:
            life_event: The new life event
            timeline: PreferenceTimeline with current preferences
            session_id: Current session number

        Returns:
            Tuple of (new_preference_ids, evolved_mapping {old_id: new_id})
        """
        # Format current preferences for the prompt
        active_prefs = timeline.get_active_preferences()
        current_prefs_json = json.dumps(
            [{"preference_id": p.preference_id, "fact": p.fact, "category": p.category}
             for p in active_prefs],
            indent=2,
        )

        prompt = render_prompt(
            "data_generation/multisession/evolve_preferences",
            persona=self.persona,
            life_event=f"{life_event.event}\nContext: {life_event.context}",
            event_date=life_event.date,
            current_preferences=current_prefs_json,
            num_to_evolve=self.num_to_evolve,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=800, temperature=0.7)

            if not isinstance(result, dict):
                result = {"evolutions": [], "new_preferences": []}

            evolutions = result.get("evolutions", [])
            new_prefs = result.get("new_preferences", [])

            evolved_mapping: dict[str, str] = {}
            new_pref_ids: list[str] = []

            # Process evolutions
            for evo in evolutions:
                old_id = evo.get("old_preference_id", "")
                new_fact = evo.get("new_fact", "")
                reason = evo.get("reason", "")

                if old_id and new_fact and old_id in timeline.preferences:
                    try:
                        new_id = timeline.evolve_preference(
                            old_id=old_id,
                            new_fact=new_fact,
                            session_id=session_id,
                            date=life_event.date,
                            reason=reason,
                        )
                        evolved_mapping[old_id] = new_id
                        logger.info(f"Evolved {old_id} -> {new_id}")
                    except ValueError as e:
                        logger.warning(f"Failed to evolve preference: {e}")

            # Process new preferences
            for np in new_prefs:
                fact = np.get("fact", "")
                category = np.get("category", "general")
                if fact:
                    pref_id = timeline.add_preference(
                        fact=fact,
                        category=category,
                        session_id=session_id,
                        date=life_event.date,
                    )
                    new_pref_ids.append(pref_id)
                    logger.info(f"Added new preference {pref_id}")

            return new_pref_ids, evolved_mapping

        except Exception as e:
            logger.error(f"Failed to evolve preferences: {e}")
            # Return empty evolution on failure (preferences stay the same)
            return [], {}

    def _generate_session_conversation(
        self,
        life_event: LifeEvent,
        timeline: PreferenceTimeline,
        session_id: int,
        evolved_mapping: dict[str, str],
    ) -> list[dict[str, str]]:
        """Generate a conversation for a session.

        Args:
            life_event: The life event for this session
            timeline: PreferenceTimeline with current state
            session_id: Current session number
            evolved_mapping: Mapping of old_id -> new_id for recently evolved prefs

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
            persona=self.persona,
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

    def generate_multi_session(self, persona_id: str = "") -> MultiSessionOutput:
        """Generate multi-session data with preference evolution.

        This is the primary method for generating multi-session data.
        For pipeline compatibility, use generate() instead.

        Args:
            persona_id: Optional identifier for the persona

        Returns:
            MultiSessionOutput with full timeline and sessions
        """
        logger.info(f"Starting multi-session generation for persona: {self.persona[:50]}...")

        # Step 1: Generate life story
        logger.info("Step 1: Generating life story...")
        life_events = self._generate_life_story()
        logger.info(f"Generated {len(life_events)} life events")

        # Initialize timeline
        timeline = PreferenceTimeline()
        sessions: list[Session] = []

        # Step 2: Process each session
        for idx, event in enumerate(life_events):
            logger.info(f"Step 2.{idx}: Processing session {idx}...")

            if idx == 0:
                # First session: generate initial preferences
                new_pref_ids = self._generate_initial_preferences(event, timeline)
                evolved_mapping: dict[str, str] = {}
            else:
                # Later sessions: evolve preferences
                new_pref_ids, evolved_mapping = self._evolve_preferences(
                    event, timeline, idx
                )

            # Generate conversation for this session
            conversation = self._generate_session_conversation(
                event, timeline, idx, evolved_mapping
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
    num_to_evolve: int = DEFAULT_NUM_TO_EVOLVE,
    output_dir: str | None = None,
) -> MultiSessionOutput:
    """Generate multi-session data with preference evolution.

    Args:
        persona: User persona description
        num_sessions: Number of sessions to generate
        num_preferences: Initial number of preferences
        num_to_evolve: Preferences to evolve per session
        output_dir: Output directory

    Returns:
        MultiSessionOutput with full timeline and sessions
    """
    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=num_sessions,
        num_preferences=num_preferences,
        num_to_evolve=num_to_evolve,
        output_dir=output_dir,
    )
    return generator.generate_multi_session()
