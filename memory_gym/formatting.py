"""Shared formatting utilities for preference history."""

from memory_gym.schemas import MultiSessionOutput, Preference


def format_preference_history(
    data: MultiSessionOutput,
    *,
    include_ids: bool = True,
) -> str:
    """Format preference history in canonical format.

    Args:
        data: Multi-session output with timeline and sessions
        include_ids: Whether to include preference IDs (e.g. [pref_004]) in the output.
            Set to False for agent-facing prompts so the agent treats preferences as
            natural knowledge rather than labeled data points.

    Returns:
        Formatted preference history string
    """
    parts: list[str] = []
    timeline = data.timeline

    def _fmt_id(pref_id: str) -> str:
        return f"[{pref_id}] " if include_ids else ""

    baseline_prefs = [p for p in timeline.preferences.values() if p.created_at_session == -1]
    if baseline_prefs:
        parts.append("BASELINE PREFERENCES:\n")
        by_domain: dict[str, list[Preference]] = {}
        for pref in baseline_prefs:
            by_domain.setdefault(pref.domain, []).append(pref)

        for domain in sorted(by_domain.keys()):
            parts.append(f"[{domain}]")
            for pref in sorted(by_domain[domain], key=lambda p: p.preference_id):
                status = (
                    f" [STALE \u2192 replaced in session {pref.superseded_at_session}]" if not pref.is_active else ""
                )
                parts.append(f"  - {_fmt_id(pref.preference_id)}{pref.fact}{status}")
            parts.append("")

    if data.sessions:
        parts.append("EVOLUTION HISTORY:\n")
        for session in data.sessions:
            parts.append(f"Session {session.session_id}: {session.life_event.event}")

            non_evolved_new = [
                pid for pid in session.new_preference_ids if pid not in session.evolved_preference_ids.values()
            ]
            for pref_id in non_evolved_new:
                pref = timeline.preferences.get(pref_id)
                if pref:
                    parts.append(f"  + {_fmt_id(pref.preference_id)}{pref.fact}")

            for old_id, new_id in session.evolved_preference_ids.items():
                old_pref = timeline.preferences.get(old_id)
                new_pref = timeline.preferences.get(new_id)
                if old_pref and new_pref:
                    if include_ids:
                        parts.append(f"  ~ [{new_id}] {new_pref.fact} (was [{old_id}]: {old_pref.fact})")
                    else:
                        parts.append(f"  ~ {new_pref.fact} (was: {old_pref.fact})")

            parts.append("")

    return "\n".join(parts)
