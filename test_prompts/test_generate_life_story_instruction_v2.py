#!/usr/bin/env python
"""Render generate_life_story_instruction_v2 prompt using actual session data."""

import sys

sys.path.insert(0, ".")

from memory_gym.prompts import render_prompt
from memory_gym.schemas import ExpandedPersona
from test_prompts._utils import load_latest_session, save_prompt


def main():
    raw, path = load_latest_session()
    print(f"Using session: {path}")

    expanded_persona = ExpandedPersona.from_dict(raw.get("expanded_persona", {}))

    domain = "hobbies_interests"
    previous_events_str = "None (this is the first event)"

    prompt = render_prompt(
        "data_generation/multisession/generate_life_story_instruction_v2",
        persona=expanded_persona.to_full_description(),
        domain=domain,
        previous_events=previous_events_str,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "generate_life_story_instruction_v2")


if __name__ == "__main__":
    main()
