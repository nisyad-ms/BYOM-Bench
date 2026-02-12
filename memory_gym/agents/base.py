"""Base agent interface."""

from abc import ABC, abstractmethod

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.schemas import MultiSessionOutput


class BaseAgent(ABC):
    """Abstract base class for evaluation agents."""

    def __init__(self, client: LLMClient | PooledLLMClient):
        self.client = client
        self._system_prompt: str | None = None

    @abstractmethod
    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Build the agent's system prompt from multi-session data."""
        pass

    def respond(self, conversation: list[dict[str, str]]) -> str:
        """Generate a response given the conversation history."""
        if self._system_prompt is None:
            raise ValueError("Agent context not built. Call build_context first.")

        messages = [{"role": "system", "content": self._system_prompt}] + conversation

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["agent"],
        )

        return response.strip()
