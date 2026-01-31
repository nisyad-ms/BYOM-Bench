#!/usr/bin/env python
"""Render expand_persona_instruction prompt using actual session data."""

import sys

sys.path.insert(0, ".")

from persona_gym.prompts import render_prompt
from test_prompts._utils import load_latest_session, save_prompt


def main():
    raw, path = load_latest_session()
    print(f"Using session: {path}")

    persona = raw.get("persona", "A software engineer.")

    prompt = render_prompt(
        "data_generation/multisession/expand_persona_instruction",
        persona=persona,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "expand_persona_instruction")


if __name__ == "__main__":
    main()
