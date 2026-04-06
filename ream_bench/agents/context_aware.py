"""Context-aware agent with full preference history."""

from byom_bench.formatting import format_preference_history
from byom_bench.prompts import render_prompt
from byom_bench.schemas import MultiSessionOutput

from ._baseline import _BaselineAgent


class ContextAwareAgent(_BaselineAgent):
    """Agent with access to full user preference history.

    Expected behavior:
    - preference_score ~1.0 (proactively uses preferences)
    - efficiency_score ~1.0 (no corrections needed)
    """

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
