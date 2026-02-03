#!/usr/bin/env python
"""Render generate_session_conversation_instruction prompt using actual session data."""

import json
import sys

sys.path.insert(0, ".")

from memory_gym.prompts import render_prompt
from memory_gym.schemas import ExpandedPersona, MultiSessionOutput
from test_prompts._utils import load_latest_session, save_prompt


def main():
    raw, path = load_latest_session()
    print(f"Using session: {path}")

    data = MultiSessionOutput.from_dict(raw)
    expanded_persona = ExpandedPersona.from_dict(raw.get("expanded_persona", {}))

    if not data.sessions:
        print("No sessions found in data")
        return

    session = data.sessions[0]
    life_event = session.life_event

    active_prefs = [
        {"id": p.preference_id, "fact": p.fact, "domain": p.domain} for p in data.get_current_preferences()[:10]
    ]
    active_prefs_json = json.dumps(active_prefs, indent=2, ensure_ascii=False)

    evolved_prefs = []
    pref_by_id = data.timeline.preferences
    for old_id, new_id in session.evolved_preference_ids.items():
        old_pref = pref_by_id.get(old_id)
        new_pref = pref_by_id.get(new_id)
        if old_pref and new_pref:
            evolved_prefs.append(
                {
                    "old_fact": old_pref.fact,
                    "new_fact": new_pref.fact,
                    "reason": new_pref.reason_for_change or old_pref.reason_for_change or "",
                }
            )
    evolved_prefs_json = json.dumps(evolved_prefs, indent=2) if evolved_prefs else "None"

    prompt = render_prompt(
        "data_generation/multisession/generate_session_conversation_instruction",
        persona=expanded_persona.to_full_description(),
        life_event=life_event.event,
        event_date=life_event.date,
        active_preferences=active_prefs_json,
        evolved_preferences=evolved_prefs_json,
        session_id=session.session_id,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "generate_session_conversation_instruction")


if __name__ == "__main__":
    main()
