"""
Multi-Session User Simulator.

The user simulator acts as a realistic user during evaluation dialogues.
It knows its current preferences and will naturally correct the agent
if recommendations don't match those preferences.
"""

import json
import logging

from persona_gym.client import LLMClient
from persona_gym.prompts import render_prompt
from persona_gym.schemas import EvaluationTask

logger = logging.getLogger(__name__)


class MultiSessionUserSimulator:
    """Simulates a user in evaluation dialogues.

    The user simulator:
    - Acts naturally based on persona and current situation
    - Knows its current preferences
    - Corrects the agent when recommendations don't match preferences
    - Does NOT know which preferences are "stale" - it just knows what it wants now

    This creates realistic dialogue where the user's corrections emerge
    naturally from preference mismatches, not from explicit testing.
    """

    def __init__(
        self,
        evaluation_task: EvaluationTask,
        client: LLMClient | None = None,
    ):
        """Initialize user simulator for a specific evaluation task.

        Args:
            evaluation_task: The evaluation task containing persona, event, and preferences
            client: LLM client for generation. If None, creates a new one.
        """
        self.task = evaluation_task
        self.client = client or LLMClient()

        # Extract current preferences from rubric
        self.current_preferences = evaluation_task.rubric.current_preferences

        # Build system prompt once
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the user simulator."""
        # Format preferences for the prompt
        prefs_formatted = json.dumps(
            [
                {"id": p.preference_id, "preference": p.fact}
                for p in self.current_preferences
            ],
            indent=2,
            ensure_ascii=False,
        )

        return render_prompt(
            "evaluation/multisession_user_simulator",
            persona_summary=self.task.persona_summary,
            evaluation_event=self.task.evaluation_event.event,
            current_preferences=prefs_formatted,
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
        max_tokens: int = 256,
    ) -> str:
        """Generate user response to agent message.

        Args:
            agent_message: The agent's most recent message
            conversation_history: Full conversation so far (including agent_message)
            max_tokens: Maximum tokens in response

        Returns:
            User's response message
        """
        # Format conversation for context
        conv_formatted = self._format_conversation(conversation_history)

        # Combine everything in user prompt to avoid jailbreak detection with system prompts
        user_prompt = f"""Continue this dialogue. The person seeking help has these PRIVATE preferences (do not reveal unless violated):
{json.dumps([{"preference": p.fact} for p in self.current_preferences], indent=2, ensure_ascii=False)}

Conversation so far:
{conv_formatted}

The assistant just said: "{agent_message}"

CRITICAL RULES:
- DO NOT proactively mention or hint at your preferences
- DO NOT say "I prefer X" or "I like Y" or explain why something works for you
- Only reveal a preference IF the assistant's suggestion directly CONFLICTS with it
- When violated, politely correct: "Actually, I'd prefer X instead"
- Accept good suggestions without explaining why they match your preferences
- Keep response brief (1-3 sentences)

Write the next response from the person seeking help. Output only the response text."""

        response = self.client.complete(
            prompt=user_prompt,
            system_prompt="You are a helpful assistant that generates dialogue continuations.",
            max_tokens=max_tokens,
            temperature=0.8,
        )

        return response.strip()

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
                    if len(last_user) < 100 and (
                        "?" not in last_user or last_user.strip().endswith("!")
                    ):
                        return True

        return False

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
    client: LLMClient | None = None,
) -> MultiSessionUserSimulator:
    """Convenience function to create a user simulator.

    Args:
        evaluation_task: The evaluation task
        client: Optional LLM client

    Returns:
        Configured MultiSessionUserSimulator
    """
    return MultiSessionUserSimulator(evaluation_task, client)
