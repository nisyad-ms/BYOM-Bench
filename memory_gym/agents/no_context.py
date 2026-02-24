"""No-context agent (baseline) with no memory of past conversations."""

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import MultiSessionOutput


class NoContextAgent:
    """Agent with no access to user preference history.

    Used as baseline for evaluation.

    Expected behavior:
    - preference_score ~0.0 (all IGNORED - user mentions everything first)
    - efficiency_score low (most turns classified as GENERIC - no personalization)
    """

    def __init__(self, client: LLMClient | PooledLLMClient):
        self.client = client
        self._system_prompt: str | None = None

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Build system prompt without any preference history."""
        self._system_prompt = render_prompt("agents/agent_system_no_context")
        return self._system_prompt

    def respond(self, conversation: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Generate a response given the conversation history.

        Returns:
            Tuple of (response_text, retrieved_memories).
            retrieved_memories is always [] for no-context agents.
        """
        if self._system_prompt is None:
            raise ValueError("Agent context not built. Call build_context first.")

        messages = [{"role": "system", "content": self._system_prompt}] + conversation

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["agent"],
        )

        return response.strip(), []
