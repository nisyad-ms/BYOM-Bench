"""
PersonaMem V2 data generator.

This generator implements the PersonaMem v2 pipeline which generates conversations
based on user preferences and topics. Unlike v1, v2:
- Uses an 8k token budget for conversation generation
- Generates exactly 5 preferences per user
- Creates sessions that fit within the token budget
- Generates topics based on user personas

The v2 approach focuses on generating meaningful preferences first, then
creating conversations that naturally reveal those preferences.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import tiktoken

from persona_gym.client import LLMClient, get_client
from persona_gym.data_generators.base import BaseDataGenerator, GenerationError
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
        prompt = f"""Generate a detailed user persona for someone who might seek assistance with {self.topic}.

Include:
- A name
- Age range (e.g., "30s", "mid-40s")
- Occupation or background
- Personality traits (2-3 traits)
- A brief life context that makes them interested in {self.topic}

Output only the persona description as a single paragraph. No JSON, no labels."""

        return self.llm.complete(prompt, max_tokens=300, temperature=0.8)

    def _generate_topics_for_persona(self, persona: str) -> list[str]:
        """Generate relevant topics based on user persona.

        Args:
            persona: The user persona description

        Returns:
            List of topic areas relevant to this persona
        """
        prompt = f"""Given this user persona:

{persona}

And the main topic area: {self.topic}

Generate 3-5 specific subtopics or aspects of {self.topic} that this person would likely discuss.
These should be natural areas of interest based on their background.

Output as a JSON array of strings, e.g.: ["subtopic1", "subtopic2", "subtopic3"]"""

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
        prompt = f"""Given this user persona:

{persona}

And these topic areas: {topics_str}

Generate exactly {self.num_preferences} specific, actionable preferences this person has.
Mix of likes and dislikes. Be specific (not generic).

For each preference:
- Make it specific and testable (e.g., "prefers window seats" not "likes comfortable seating")
- Include a mix of likes (positive) and dislikes (negative)
- At least one should be a preference that could change over time (for "updated" type)

Output as JSON array:
[
  {{"fact": "specific preference statement", "type": "current", "topic": "relevant topic"}},
  {{"fact": "another preference", "type": "current", "topic": "relevant topic"}},
  {{"fact": "preference that changed", "type": "updated", "topic": "relevant topic", "old_value": "previous preference", "reason_of_change": "why it changed"}}
]

Include {self.num_preferences} preferences total, with at least 1 "updated" type."""

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
                ))

            # Ensure we have the right number of preferences
            while len(preferences) < self.num_preferences:
                preferences.append(PreferenceItem(
                    fact=f"has a general preference related to {self.topic}",
                    preference_type="current",
                    source_date=datetime.now().strftime("%m/%d/%Y"),
                    topic=self.topic,
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
                )
                for _ in range(self.num_preferences)
            ]

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

        prompt = f"""Generate a natural conversation between a user and an assistant about {self.topic}.

USER PERSONA:
{persona}

PREFERENCES TO REVEAL (user should naturally mention ALL of these during conversation):
{prefs_text}

REQUIREMENTS:
1. Generate a multi-turn conversation (aim for 8-12 turns total)
2. The user should naturally reveal each preference during the conversation
3. For "updated" preferences, show the change naturally (e.g., "I used to X but now I prefer Y because...")
4. Keep the total conversation length around {target_tokens} tokens
5. Make the conversation feel natural, not like a survey

OUTPUT FORMAT (JSON):
{{
  "turns": [
    {{"role": "user", "content": "message", "side_note": {{"event": "what preference is revealed", "date": "{datetime.now().strftime('%m/%d/%Y')}"}}}},
    {{"role": "assistant", "content": "response", "side_note": null}},
    ...
  ],
  "topic": "{self.topic}",
  "period": "INIT"
}}

CRITICAL:
- Every turn must have role as exactly "user" or "assistant"
- Every turn must have side_note field (null if no preference revealed)
- Strictly alternate user/assistant
- Output ONLY valid JSON"""

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

            # Step 3: Generate preferences
            preferences = self._generate_preferences(persona, topics)
            logger.info(f"Generated {len(preferences)} preferences")

            # Step 4: Generate conversation revealing preferences
            conversation = self._generate_conversation(persona, preferences)
            logger.info(f"Generated conversation with {len(conversation.turns)} turns")

            # Step 5: Validate token budget
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

            # Step 3: Generate preferences
            preferences = self._generate_preferences(persona, topics)

            # Step 4: Generate conversation revealing preferences
            conversation = self._generate_conversation(persona, preferences)

            # Step 5: Validate token budget
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
            artifacts = {
                "persona": persona,
                "generated_topics": topics,
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
