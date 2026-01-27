"""
PersonaMem V2 data generator.

This generator implements the PersonaMem v2 pipeline which generates conversations
based on user preferences and topics. Unlike v1, v2:
- Uses an 8k token budget for conversation generation
- Generates exactly 5 preferences per user
- Creates sessions that fit within the token budget
- Generates topics based on user personas
- Supports preference evolution (40% of preferences change over time)

The v2 approach focuses on generating meaningful preferences first, then
creating conversations that naturally reveal those preferences.
"""

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

import tiktoken

from persona_gym.client import LLMClient, get_client
from persona_gym.data_generators.base import BaseDataGenerator, GenerationError
from persona_gym.prompts import render_prompt
from persona_gym.schemas import (
    ConversationTurn,
    DataGenerationMetadata,
    DataGenerationOutput,
    GeneratedConversation,
    PreferenceItem,
    SideNote,
)

logger = logging.getLogger(__name__)

# Default token budget for conversation generation
DEFAULT_TOKEN_BUDGET = 8000
DEFAULT_NUM_PREFERENCES = 5


class PersonaMemV2Generator(BaseDataGenerator):
    """Generate conversation data using the PersonaMem V2 pipeline.

    PersonaMem V2 creates synthetic conversations with:
    - Fixed number of user preferences (default: 5)
    - Token-budgeted conversation generation (default: 8k tokens)
    - Topic generation based on user persona
    - Natural preference revelation in conversations

    Attributes:
        topic: Conversation topic (e.g., "travel", "cooking", "therapy")
        persona: User persona description (if None, will generate one)
        token_budget: Maximum tokens for conversation (default: 8000)
        num_preferences: Number of preferences to generate (default: 5)
        output_dir: Directory for output files
    """

    def __init__(
        self,
        topic: str,
        persona: Optional[str] = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        num_preferences: int = DEFAULT_NUM_PREFERENCES,
        output_dir: Optional[str] = None,
    ):
        """Initialize PersonaMem V2 generator.

        Args:
            topic: Conversation topic domain
            persona: User persona description (optional, will generate if not provided)
            token_budget: Maximum tokens for conversation (default: 8000)
            num_preferences: Number of preferences to generate (default: 5)
            output_dir: Output directory (optional)
        """
        super().__init__(topic=topic, output_dir=output_dir)
        self.persona = persona
        self.token_budget = token_budget
        self.num_preferences = num_preferences
        self._llm: Optional[LLMClient] = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def llm(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm is None:
            self._llm = get_client()
        return self._llm

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self._tokenizer.encode(text))

    def _generate_persona(self) -> str:
        """Generate a user persona if not provided."""
        prompt = render_prompt(
            "data_generation/personamemv2/generate_persona",
            topic=self.topic,
        )
        return self.llm.complete(prompt, max_tokens=300, temperature=0.8)

    def _generate_topics_for_persona(self, persona: str) -> list[str]:
        """Generate relevant topics based on user persona.

        Args:
            persona: The user persona description

        Returns:
            List of topic areas relevant to this persona
        """
        prompt = render_prompt(
            "data_generation/personamemv2/generate_subtopics",
            persona=persona,
            topic=self.topic,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=200, temperature=0.7)
            if isinstance(result, dict):
                # Handle case where LLM wraps in a dict
                topics = result.get("topics", result.get("subtopics", []))
            elif isinstance(result, list):
                topics = result
            else:
                topics = [self.topic]
            return topics[:5] if topics else [self.topic]
        except Exception as e:
            logger.warning(f"Failed to generate topics: {e}, using default topic")
            return [self.topic]

    def _generate_preferences(self, persona: str, topics: list[str]) -> list[PreferenceItem]:
        """Generate user preferences based on persona and topics.

        Args:
            persona: The user persona description
            topics: List of relevant topics

        Returns:
            List of PreferenceItem objects
        """
        topics_str = ", ".join(topics)
        prompt = render_prompt(
            "data_generation/personamemv2/generate_preferences",
            persona=persona,
            topics_str=topics_str,
            num_preferences=self.num_preferences,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=800, temperature=0.7)
            preferences = []

            # Handle both array and dict responses
            pref_list = result if isinstance(result, list) else result.get("preferences", [])

            for p in pref_list[:self.num_preferences]:
                pref_type = p.get("type", "current")
                preferences.append(PreferenceItem(
                    fact=p.get("fact", ""),
                    preference_type=pref_type,
                    source_date=datetime.now().strftime("%m/%d/%Y"),
                    topic=p.get("topic", self.topic),
                    old_value=p.get("old_value") if pref_type == "updated" else None,
                    reason_of_change=p.get("reason_of_change") if pref_type == "updated" else None,
                    preference_id=str(uuid.uuid4())[:8],  # Short UUID for readability
                ))

            # Ensure we have the right number of preferences
            while len(preferences) < self.num_preferences:
                preferences.append(PreferenceItem(
                    fact=f"has a general preference related to {self.topic}",
                    preference_type="current",
                    source_date=datetime.now().strftime("%m/%d/%Y"),
                    topic=self.topic,
                    preference_id=str(uuid.uuid4())[:8],
                ))

            return preferences[:self.num_preferences]

        except Exception as e:
            logger.warning(f"Failed to generate preferences: {e}, creating defaults")
            return [
                PreferenceItem(
                    fact=f"has preferences related to {self.topic}",
                    preference_type="current",
                    source_date=datetime.now().strftime("%m/%d/%Y"),
                    topic=self.topic,
                    preference_id=str(uuid.uuid4())[:8],
                )
                for _ in range(self.num_preferences)
            ]

    def _evolve_preferences(
        self,
        preferences: list[PreferenceItem],
        persona: str,
        evolution_rate: float = 0.4,
    ) -> list[PreferenceItem]:
        """Evolve a subset of preferences to simulate change over time.

        Takes existing preferences and creates updated versions for ~40% of them,
        simulating how user preferences change. Updates can be:
        - Direct contradiction: "I don't like X anymore"
        - Indirect implication: "I was diagnosed with allergy" (implies dietary changes)

        Args:
            preferences: List of original preferences
            persona: User persona for context
            evolution_rate: Fraction of preferences to evolve (default: 0.4 = 40%)

        Returns:
            List containing both original and evolved preferences, with evolved
            preferences having supersedes_id pointing to the original.
        """
        num_to_evolve = max(1, int(len(preferences) * evolution_rate))

        # Select preferences to evolve (avoid already-updated ones)
        candidates = [p for p in preferences if p.preference_type == "current"]
        if len(candidates) < num_to_evolve:
            candidates = preferences[:num_to_evolve]

        # Randomly select which preferences to evolve
        to_evolve = random.sample(candidates, min(num_to_evolve, len(candidates)))

        if not to_evolve:
            return preferences

        # Format preferences for prompt
        prefs_to_evolve = "\n".join([
            f"- ID: {p.preference_id}, Preference: {p.fact}, Category: {p.topic}"
            for p in to_evolve
        ])

        prompt = render_prompt(
            "data_generation/personamemv2/evolve_preferences",
            persona=persona,
            prefs_to_evolve=prefs_to_evolve,
            num_evolutions=len(to_evolve),
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=800, temperature=0.7)
            logger.debug(f"Evolution LLM response: {result}")

            # Handle various response formats
            if isinstance(result, list):
                evolutions = result
            elif isinstance(result, dict):
                # Try common key names the LLM might use
                evolutions = (
                    result.get("evolutions") or
                    result.get("data") or
                    result.get("preferences") or
                    result.get("updates") or
                    result.get("evolved_preferences") or
                    result.get("results") or
                    []
                )
            else:
                evolutions = []

            logger.debug(f"Parsed evolutions: {evolutions}")

            # Build map of original preferences by ID
            pref_map = {p.preference_id: p for p in preferences}
            logger.debug(f"Preference map keys: {list(pref_map.keys())}")

            # Create evolved preferences
            evolved_prefs = []
            evolved_ids = set()

            for evo in evolutions:
                original_id = evo.get("original_id")
                logger.debug(f"Processing evolution for original_id: {original_id}")
                if original_id not in pref_map:
                    logger.debug("  -> original_id not found in pref_map")
                    continue
                if original_id in evolved_ids:
                    logger.debug("  -> original_id already evolved")
                    continue

                original = pref_map[original_id]
                evolved_ids.add(original_id)

                # Generate a new date that's later than the original
                # Parse original date and add some time
                try:
                    original_date = datetime.strptime(original.source_date, "%m/%d/%Y")
                    new_date = original_date + timedelta(days=random.randint(30, 180))
                    new_date_str = new_date.strftime("%m/%d/%Y")
                except ValueError:
                    new_date_str = evo.get("new_source_date", datetime.now().strftime("%m/%d/%Y"))

                evolved_prefs.append(PreferenceItem(
                    fact=evo.get("new_fact", f"no longer: {original.fact}"),
                    preference_type="updated",
                    source_date=new_date_str,
                    topic=original.topic,
                    old_value=original.fact,
                    reason_of_change=evo.get("reason_of_change", "preference changed"),
                    preference_id=str(uuid.uuid4())[:8],
                    supersedes_id=original_id,
                ))

            logger.info(f"Created {len(evolved_prefs)} evolved preferences")

            # Return original preferences + evolved ones
            # The evolved ones will have supersedes_id linking to originals
            return preferences + evolved_prefs

        except Exception as e:
            logger.warning(f"Failed to evolve preferences: {e}, returning originals")
            return preferences

    def _generate_conversation(
        self,
        persona: str,
        preferences: list[PreferenceItem],
    ) -> GeneratedConversation:
        """Generate a conversation that reveals the given preferences.

        Args:
            persona: The user persona description
            preferences: List of preferences to reveal in conversation

        Returns:
            GeneratedConversation object
        """
        # Format preferences for prompt
        pref_descriptions = []
        for i, p in enumerate(preferences, 1):
            desc = f"{i}. {p.fact}"
            if p.preference_type == "updated" and p.old_value:
                desc += f" (changed from: {p.old_value}, reason: {p.reason_of_change})"
            pref_descriptions.append(desc)

        prefs_text = "\n".join(pref_descriptions)

        # Calculate target conversation length based on token budget
        # Reserve some tokens for structure overhead
        target_tokens = self.token_budget - 500
        current_date = datetime.now().strftime("%m/%d/%Y")

        prompt = render_prompt(
            "data_generation/personamemv2/generate_conversation",
            topic=self.topic,
            persona=persona,
            prefs_text=prefs_text,
            target_tokens=target_tokens,
            current_date=current_date,
        )

        try:
            result = self.llm.complete_json(prompt, max_tokens=4000, temperature=0.7)

            turns = []
            for t in result.get("turns", []):
                side_note = None
                if t.get("side_note") and isinstance(t["side_note"], dict):
                    side_note = SideNote(
                        event=t["side_note"].get("event", ""),
                        date=t["side_note"].get("date", datetime.now().strftime("%m/%d/%Y")),
                    )

                turns.append(ConversationTurn(
                    role=t.get("role", "user"),
                    content=t.get("content", ""),
                    side_note=side_note,
                ))

            return GeneratedConversation(
                turns=turns,
                topic=self.topic,
                period="INIT",
            )

        except Exception as e:
            logger.error(f"Failed to generate conversation: {e}")
            # Return minimal valid conversation
            return GeneratedConversation(
                turns=[
                    ConversationTurn(role="user", content=f"I'd like help with {self.topic}.", side_note=None),
                    ConversationTurn(role="assistant", content=f"I'd be happy to help with {self.topic}. What would you like to know?", side_note=None),
                ],
                topic=self.topic,
                period="INIT",
            )

    def _validate_token_budget(self, conversation: GeneratedConversation) -> GeneratedConversation:
        """Ensure conversation fits within token budget.

        If over budget, truncates from the middle while preserving
        the beginning (context) and end (conclusion).

        Args:
            conversation: The generated conversation

        Returns:
            Conversation within token budget
        """
        # Calculate current token count
        total_text = " ".join(t.content for t in conversation.turns)
        current_tokens = self._count_tokens(total_text)

        if current_tokens <= self.token_budget:
            return conversation

        logger.info(f"Conversation exceeds token budget ({current_tokens} > {self.token_budget}), truncating")

        # Keep first 2 and last 2 turns, remove from middle
        turns = conversation.turns
        if len(turns) <= 4:
            return conversation

        # Binary search for how many middle turns to remove
        kept_turns = turns[:2] + turns[-2:]
        middle_turns = turns[2:-2]

        while middle_turns:
            test_turns = turns[:2] + middle_turns + turns[-2:]
            total_text = " ".join(t.content for t in test_turns)
            if self._count_tokens(total_text) <= self.token_budget:
                return GeneratedConversation(
                    turns=test_turns,
                    topic=conversation.topic,
                    period=conversation.period,
                )
            # Remove turns from middle
            mid_idx = len(middle_turns) // 2
            # Remove a pair (user + assistant) if possible
            if mid_idx > 0 and mid_idx < len(middle_turns):
                middle_turns = middle_turns[:mid_idx-1] + middle_turns[mid_idx+1:]
            else:
                middle_turns = middle_turns[:-1]

        return GeneratedConversation(
            turns=kept_turns,
            topic=conversation.topic,
            period=conversation.period,
        )

    def generate(self) -> DataGenerationOutput:
        """Generate conversation data using PersonaMem V2 pipeline.

        Returns:
            DataGenerationOutput with conversation, preferences, and metadata

        Raises:
            GenerationError: If generation fails
        """
        logger.info(f"Starting PersonaMem V2 generation for topic: {self.topic}")
        logger.info(f"Token budget: {self.token_budget}, Num preferences: {self.num_preferences}")

        try:
            # Step 1: Get or generate persona
            persona = self.persona or self._generate_persona()
            logger.info(f"Using persona: {persona[:100]}...")

            # Step 2: Generate topics based on persona
            topics = self._generate_topics_for_persona(persona)
            logger.info(f"Generated topics: {topics}")

            # Step 3: Generate initial preferences
            preferences = self._generate_preferences(persona, topics)
            logger.info(f"Generated {len(preferences)} initial preferences")

            # Step 4: Evolve preferences (40% will change over time)
            preferences = self._evolve_preferences(preferences, persona)
            num_evolved = len([p for p in preferences if p.supersedes_id])
            logger.info(f"Evolved {num_evolved} preferences, total now: {len(preferences)}")

            # Step 5: Generate conversation revealing preferences
            conversation = self._generate_conversation(persona, preferences)
            logger.info(f"Generated conversation with {len(conversation.turns)} turns")

            # Step 6: Validate token budget
            conversation = self._validate_token_budget(conversation)
            final_tokens = self._count_tokens(" ".join(t.content for t in conversation.turns))
            logger.info(f"Final conversation: {len(conversation.turns)} turns, {final_tokens} tokens")

            # Convert to output format
            conv_messages = [
                {"role": t.role, "content": t.content}
                for t in conversation.turns
            ]

            metadata = DataGenerationMetadata(
                topic=self.topic,
                persona_id=f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                source_file="",
            )

            return DataGenerationOutput(
                conversation=conv_messages,
                preferences=preferences,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"PersonaMem V2 generation failed: {e}")
            raise GenerationError(f"PersonaMem V2 generation failed: {e}") from e

    def generate_with_artifacts(self) -> tuple[DataGenerationOutput, dict[str, Any]]:
        """Generate data and return additional artifacts for debugging.

        Returns:
            Tuple of (DataGenerationOutput, artifacts_dict)
        """
        logger.info(f"Starting PersonaMem V2 generation with artifacts for topic: {self.topic}")

        try:
            # Step 1: Get or generate persona
            persona = self.persona or self._generate_persona()

            # Step 2: Generate topics based on persona
            topics = self._generate_topics_for_persona(persona)

            # Step 3: Generate initial preferences
            initial_preferences = self._generate_preferences(persona, topics)

            # Step 4: Evolve preferences (40% will change over time)
            preferences = self._evolve_preferences(initial_preferences, persona)

            # Step 5: Generate conversation revealing preferences
            conversation = self._generate_conversation(persona, preferences)

            # Step 6: Validate token budget
            conversation = self._validate_token_budget(conversation)

            # Convert to output format
            conv_messages = [
                {"role": t.role, "content": t.content}
                for t in conversation.turns
            ]

            metadata = DataGenerationMetadata(
                topic=self.topic,
                persona_id=f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                source_file="",
            )

            output = DataGenerationOutput(
                conversation=conv_messages,
                preferences=preferences,
                metadata=metadata,
            )

            # Build artifacts for debugging
            evolved_prefs = [p for p in preferences if p.supersedes_id]
            artifacts = {
                "persona": persona,
                "generated_topics": topics,
                "initial_preferences": [p.to_dict() for p in initial_preferences],
                "evolved_preferences": [p.to_dict() for p in evolved_prefs],
                "raw_conversation": conversation.model_dump(),
                "token_count": self._count_tokens(" ".join(t.content for t in conversation.turns)),
                "config": {
                    "token_budget": self.token_budget,
                    "num_preferences": self.num_preferences,
                    "topic": self.topic,
                },
            }

            return output, artifacts

        except Exception as e:
            logger.error(f"PersonaMem V2 generation failed: {e}")
            raise GenerationError(f"PersonaMem V2 generation failed: {e}") from e
