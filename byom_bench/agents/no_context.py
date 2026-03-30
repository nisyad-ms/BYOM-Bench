"""No-context agent (baseline) with no memory of past conversations."""

from byom_bench.prompts import render_prompt
from byom_bench.schemas import MultiSessionOutput

from ._baseline import _BaselineAgent


class NoContextAgent(_BaselineAgent):
    """Agent with no access to user preference history.

    Used as baseline for evaluation.

    Expected behavior:
    - preference_score ~0.0 (all IGNORED - user mentions everything first)
    - efficiency_score low (most turns classified as GENERIC - no personalization)
    """

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Build system prompt without any preference history."""
        self._system_prompt = render_prompt("agents/agent_system_no_context")
        return self._system_prompt
