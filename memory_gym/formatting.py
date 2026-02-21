"""Shared formatting utilities for preference history."""

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import MultiSessionOutput, Preference


def summarize_events(
    data: MultiSessionOutput,
    client: LLMClient | PooledLLMClient,
) -> dict[int, str]:
    """Summarize all session events in a single LLM call.

    Args:
        data: Multi-session output with sessions to summarize
        client: LLM client for generation

    Returns:
        Dict mapping session_id to 1-3 sentence summary
    """
    if not data.sessions:
        return {}

    events_block = "\n".join(f"Session {s.session_id}: {s.life_event.event}" for s in data.sessions)

    prompt = render_prompt(
        "data_generation/event_summary_instruction",
        events=events_block,
    )

    response = client.complete_json(
        prompt=prompt,
        max_tokens=CONFIG["max_tokens"].get("event_summary", 4096),
    )

    return {int(k): v for k, v in response.items()}


def format_preference_history(
    data: MultiSessionOutput,
    event_summaries: dict[int, str],
    *,
    include_ids: bool = True,
) -> str:
    """Format preference history in canonical format.

    Args:
        data: Multi-session output with timeline and sessions
        event_summaries: Dict mapping session_id to summarized event text
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
            summary = event_summaries.get(session.session_id, "")
            parts.append(f"Session {session.session_id}: {summary}")

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
