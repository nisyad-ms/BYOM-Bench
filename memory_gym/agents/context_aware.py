"""Context-aware agent with full preference history."""

from memory_gym.formatting import format_preference_history
from memory_gym.prompts import render_prompt
from memory_gym.schemas import MultiSessionOutput

from .base import BaseAgent


class ContextAwareAgent(BaseAgent):
    """Agent with access to full user preference history.

    Expected behavior:
    - preference_score ~1.0 (proactively uses preferences)
    - efficiency_score ~1.0 (no corrections needed)
    """

    def build_context(
        self,
        multisession_data: MultiSessionOutput,
        event_summaries: dict[int, str] | None = None,
    ) -> str:
        """Build system prompt with structured preference history."""
        if event_summaries is None:
            event_summaries = {}
        preference_history = format_preference_history(multisession_data, event_summaries)
        self._system_prompt = render_prompt(
            "agents/agent_system_with_context",
            preference_history=preference_history,
        )
        return self._system_prompt
