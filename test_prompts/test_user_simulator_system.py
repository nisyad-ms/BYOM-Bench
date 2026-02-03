#!/usr/bin/env python
"""Render user_simulator_system prompt using latest task data."""

import json
import sys

sys.path.insert(0, ".")

from memory_gym.prompts import render_prompt
from test_prompts._utils import load_latest_task, save_prompt


def main():
    task_data, path = load_latest_task()
    print(f"Using task: {path}")

    eval_event = task_data.get("evaluation_event", {})
    rubric = task_data.get("rubric", {})
    required_prefs = rubric.get("required_preferences", [])

    prefs_formatted = json.dumps(
        [{"id": p["id"], "preference": p["fact"]} for p in required_prefs],
        indent=2,
        ensure_ascii=False,
    )

    prompt = render_prompt(
        "evaluation/user_simulator_system",
        persona_summary=task_data.get("persona_summary", ""),
        evaluation_event=eval_event.get("event", ""),
        required_preferences=prefs_formatted,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "user_simulator_system")


if __name__ == "__main__":
    main()
