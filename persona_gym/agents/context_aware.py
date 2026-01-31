"""Context-aware agent with full preference history."""

from persona_gym.prompts import render_prompt
from persona_gym.schemas import MultiSessionOutput

from .base import BaseAgent


class ContextAwareAgent(BaseAgent):
    """Agent with access to full user preference history.

    Expected behavior:
    - preference_score ~1.0 (proactively uses preferences)
    - efficiency_score ~1.0 (no corrections needed)
    """

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Build system prompt with structured preference history."""
        preference_history = _format_preference_history(multisession_data)
        self._system_prompt = render_prompt(
            "agents/agent_system_with_context",
            preference_history=preference_history,
        )
        return self._system_prompt


def _format_preference_history(data: MultiSessionOutput) -> str:
    """Format preference history as structured memory for the agent.

    Structure:
    1. Core Preferences - baseline preferences (created_at_session=-1)
    2. Preference Evolution History - per session changes
    """
    parts = ["## User Preference History\n"]

    baseline_prefs = [
        p for p in data.timeline.preferences.values()
        if p.created_at_session == -1
    ]
    if baseline_prefs:
        parts.append("### Core Preferences (before user started talking to the agent)\n")
        by_domain: dict[str, list] = {}
        for pref in baseline_prefs:
            by_domain.setdefault(pref.domain, []).append(pref)

        for domain in sorted(by_domain.keys()):
            parts.append(f"**{domain}:**")
            for pref in sorted(by_domain[domain], key=lambda p: p.preference_id):
                if pref.is_active:
                    parts.append(f"- {pref.fact}")
                else:
                    parts.append(f"- {pref.fact} [UPDATED in session {pref.superseded_at_session}]")
            parts.append("")

    if data.sessions:
        parts.append("### Preference Evolution History\n")
        for session in data.sessions:
            event = session.life_event
            parts.append(f"**Session {session.session_id} - {event.event}**\n")

            if session.new_preference_ids:
                non_evolved_new = [
                    pid for pid in session.new_preference_ids
                    if pid not in session.evolved_preference_ids.values()
                ]
                if non_evolved_new:
                    parts.append("New preferences:")
                    for pref_id in non_evolved_new:
                        pref = data.timeline.preferences.get(pref_id)
                        if pref:
                            parts.append(f"- [{pref.domain}] {pref.fact}")
                    parts.append("")

            if session.evolved_preference_ids:
                parts.append("Evolved preferences:")
                for old_id, new_id in session.evolved_preference_ids.items():
                    old_pref = data.timeline.preferences.get(old_id)
                    new_pref = data.timeline.preferences.get(new_id)
                    if old_pref and new_pref:
                        old_origin = "baseline" if old_pref.created_at_session == -1 else f"session {old_pref.created_at_session}"
                        parts.append(f"- EVOLVED: \"{old_pref.fact}\" [from {old_origin}]")
                        parts.append(f"  → \"{new_pref.fact}\"")
                parts.append("")

    return "\n".join(parts)
