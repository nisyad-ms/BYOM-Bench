"""Shared base for ContextAwareAgent and NoContextAgent.

Both baseline agents share the same ``__init__`` and ``respond`` logic —
only ``build_context`` differs.  This module extracts the common parts so
each agent file stays minimal.
"""

from ream_bench.client import CONFIG, LLMClient, PooledLLMClient
from ream_bench.schemas import MultiSessionOutput


class _BaselineAgent:
    """Common plumbing for non-memory agents (context-aware / no-context).

    Subclasses must override :meth:`build_context` to set ``_system_prompt``.
    """

    def __init__(self, client: LLMClient | PooledLLMClient):
        self.client = client
        self._system_prompt: str | None = None

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Build the system prompt.  **Override in subclasses.**"""
        raise NotImplementedError

    def respond(self, conversation: list[dict[str, str]], memory_token_budget: int | None = None) -> tuple[str, list[dict]]:
        """Generate a response given the conversation history.

        Returns:
            Tuple of (response_text, retrieved_memories).
            retrieved_memories is always ``[]`` for baseline agents.
        """
        if self._system_prompt is None:
            raise ValueError("Agent context not built. Call build_context first.")

        messages = [{"role": "system", "content": self._system_prompt}] + conversation

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["agent"],
        )

        return response.strip(), []
