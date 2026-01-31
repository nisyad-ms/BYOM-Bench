#!/usr/bin/env python
"""Render evaluation_task_system prompt."""

import sys

sys.path.insert(0, ".")

from persona_gym.prompts import render_prompt
from test_prompts._utils import save_prompt


def main():
    prompt = render_prompt("task_generation/evaluation_task_system")

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "evaluation_task_system")


if __name__ == "__main__":
    main()
