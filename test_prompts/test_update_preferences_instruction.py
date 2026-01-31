#!/usr/bin/env python
"""Render update_preferences_instruction prompt using actual session data."""

import json
import sys

sys.path.insert(0, ".")

from persona_gym.prompts import render_prompt
from persona_gym.schemas import ExpandedPersona, MultiSessionOutput
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
        {"id": p.preference_id, "fact": p.fact, "domain": p.domain}
        for p in data.get_current_preferences()[:10]
    ]
    active_prefs_json = json.dumps(active_prefs, indent=2, ensure_ascii=False)

    domain_facts_lines = []
    for domain in ["work_education", "health_wellness", "travel_experiences", "relationships_personal", "hobbies_interests"]:
        facts = expanded_persona.get_domain_facts(domain)
        if facts:
            domain_facts_lines.append(f"{domain}:")
            for fact in facts:
                domain_facts_lines.append(f"  - {fact}")
    domain_facts_str = "\n".join(domain_facts_lines)

    current_event_str = life_event.event
    previous_events_str = "None (this is the first event)"
    evolution_history_str = "No evolutions yet (session 0)"

    prompt = render_prompt(
        "data_generation/multisession/update_preferences_instruction",
        persona=expanded_persona.to_full_description(),
        domain_facts=domain_facts_str,
        current_event=current_event_str,
        event_date=life_event.date,
        previous_events=previous_events_str,
        active_preferences=active_prefs_json,
        evolution_history=evolution_history_str,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "update_preferences_instruction")


if __name__ == "__main__":
    main()
