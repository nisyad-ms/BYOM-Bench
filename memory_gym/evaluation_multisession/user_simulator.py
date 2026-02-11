"""
Multi-Session User Simulator.

The user simulator acts as a realistic user during evaluation dialogues.
It knows its current preferences and will naturally correct the agent
if recommendations don't match those preferences.
"""

import re

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import EvaluationTask


class MultiSessionUserSimulator:
    """Simulates a user in evaluation dialogues.

    The user simulator:
    - Acts naturally based on persona and current situation
    - Knows its current preferences (from required_preferences)
    - Corrects the agent when recommendations don't match preferences
    - Does NOT know which preferences are "stale" - it just knows what it wants now

    This creates realistic dialogue where the user's corrections emerge
    naturally from preference mismatches, not from explicit testing.
    """

    def __init__(
        self,
        evaluation_task: EvaluationTask,
        client: LLMClient | PooledLLMClient | None = None,
    ):
        """Initialize user simulator for a specific evaluation task.

        Args:
            evaluation_task: The evaluation task containing persona, event, and preferences
            client: LLM client for generation. If None, creates a new one.
        """
        self.task = evaluation_task
        self.client = client or PooledLLMClient()

        # Extract preferences from rubric.required_preferences
        # These are dicts with {id, fact, supersedes?: ...}
        # User only needs to know their current preferences (ignores supersedes)
        self.required_preferences = evaluation_task.rubric.required_preferences

        # Build system prompt once
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the user simulator."""
        prefs_formatted = "\n".join(f"- {p['fact']}" for p in self.required_preferences)

        return render_prompt(
            "evaluation/user_simulator_system",
            persona_summary=self.task.persona_summary,
            evaluation_event=self.task.evaluation_event.event,
            required_preferences=prefs_formatted,
        )

    def get_initial_message(self) -> str:
        """Get the initial user message to start the dialogue.

        This is the pre-generated user prompt from the orchestrator.
        """
        return self.task.user_prompt

    def respond(
        self,
        agent_message: str,
        conversation_history: list[dict[str, str]],
    ) -> tuple[str, str | None]:
        """Generate user response to agent message.

        Args:
            agent_message: The agent's most recent message
            conversation_history: Full conversation so far (including agent_message)

        Returns:
            Tuple of (user_response, scratchpad_or_none)
            - user_response: Clean response to add to conversation
            - scratchpad: Raw scratchpad content for debugging, or None if not present
        """
        conv_text = self._format_conversation_as_string(conversation_history)
        user_prompt = render_prompt(
            "evaluation/user_simulator_user",
            conversation=conv_text,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["user_simulator"],
        )

        raw_response = response.strip()
        clean_response, scratchpad = self._extract_scratchpad(raw_response)
        return clean_response, scratchpad

    def _extract_scratchpad(self, response: str) -> tuple[str, str | None]:
        """Extract scratchpad and clean response separately.

        Returns:
            Tuple of (clean_response, scratchpad_content_or_none)
        """
        match = re.search(r"<scratchpad>(.*?)</scratchpad>", response, flags=re.DOTALL)
        scratchpad = match.group(1).strip() if match else None
        clean = re.sub(r"<scratchpad>.*?</scratchpad>\s*", "", response, flags=re.DOTALL)
        return clean.strip(), scratchpad

    def should_end_conversation(
        self,
        agent_message: str,
        conversation_history: list[dict[str, str]],
    ) -> bool:
        """Determine if the conversation should end naturally.

        The conversation ends when:
        - User explicitly thanks and ends (detected in their response)
        - Task appears complete
        - Maximum turns reached (handled externally)
        """
        # Check for common ending phrases in the last user response
        if conversation_history:
            last_user = None
            for turn in reversed(conversation_history):
                if turn.get("role") == "user":
                    last_user = turn.get("content", "").lower()
                    break

            if last_user:
                end_phrases = [
                    "thank you",
                    "thanks",
                    "that's all",
                    "that's perfect",
                    "sounds good",
                    "i'm good",
                    "that works",
                    "perfect",
                    "great, thanks",
                    "appreciate it",
                ]
                if any(phrase in last_user for phrase in end_phrases):
                    # Check it's actually an ending, not mid-conversation thanks
                    if len(last_user) < 100 and ("?" not in last_user or last_user.strip().endswith("!")):
                        return True

        return False

    def _format_conversation_as_string(self, history: list[dict[str, str]]) -> str:
        """Format conversation history as labeled string for the prompt."""
        lines = []
        for turn in history:
            role = "User" if turn.get("role") == "user" else "Agent"
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _format_conversation(self, history: list[dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        lines = []
        for turn in history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


def create_user_simulator(
    evaluation_task: EvaluationTask,
    client: LLMClient | PooledLLMClient | None = None,
) -> MultiSessionUserSimulator:
    """Convenience function to create a user simulator.

    Args:
        evaluation_task: The evaluation task
        client: Optional LLM client

    Returns:
        Configured MultiSessionUserSimulator
    """
    return MultiSessionUserSimulator(evaluation_task, client)
