#!/usr/bin/env python
"""Render preference_judge_system prompt."""

import sys

sys.path.insert(0, ".")

from memory_gym.prompts import render_prompt
from test_prompts._utils import save_prompt


def main():
    prompt = render_prompt("evaluation/preference_judge_system")

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "preference_judge_system")


if __name__ == "__main__":
    main()
