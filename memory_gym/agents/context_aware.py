"""Context-aware agent with full preference history."""

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.formatting import format_preference_history
from memory_gym.prompts import render_prompt
from memory_gym.schemas import MultiSessionOutput


class ContextAwareAgent:
    """Agent with access to full user preference history.

    Expected behavior:
    - preference_score ~1.0 (proactively uses preferences)
    - efficiency_score ~1.0 (no corrections needed)
    """

    def __init__(self, client: LLMClient | PooledLLMClient):
        self.client = client
        self._system_prompt: str | None = None

    def build_context(
        self,
        multisession_data: MultiSessionOutput,
    ) -> str:
        """Build system prompt with structured preference history."""
        preference_history = format_preference_history(multisession_data, include_ids=False)
        self._system_prompt = render_prompt(
            "agents/agent_system_with_context",
            preference_history=preference_history,
        )
        return self._system_prompt

    def respond(self, conversation: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Generate a response given the conversation history.

        Returns:
            Tuple of (response_text, retrieved_memories).
            retrieved_memories is always [] for context-aware agents.
        """
        if self._system_prompt is None:
            raise ValueError("Agent context not built. Call build_context first.")

        messages = [{"role": "system", "content": self._system_prompt}] + conversation

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["agent"],
        )

        return response.strip(), []
