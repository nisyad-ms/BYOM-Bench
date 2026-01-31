#!/usr/bin/env python
"""Render agent_system_no_context prompt."""

import sys

sys.path.insert(0, ".")

from persona_gym.prompts import render_prompt
from test_prompts._utils import save_prompt


def main():
    prompt = render_prompt("agents/agent_system_no_context")

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "agent_system_no_context")


if __name__ == "__main__":
    main()
