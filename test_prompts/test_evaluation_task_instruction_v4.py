#!/usr/bin/env python
"""Render evaluation_task_instruction_v4 prompt using latest session data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from persona_gym.client import LLMClient
from persona_gym.prompts import render_prompt
from persona_gym.schemas import MultiSessionOutput
from persona_gym.task_generators.evaluation_task import EvaluationTaskGenerator
from utils import get_latest_session_dir, get_session_path


def save_prompt(prompt: str, name: str):
    out_path = Path("outputs/prompts") / f"{name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(prompt)
    print(f"Saved to {out_path}")


def main():
    session_dir = get_latest_session_dir()
    if not session_dir:
        raise FileNotFoundError("No session directories found")
    session_path = get_session_path(session_dir)
    with open(session_path) as f:
        raw = json.load(f)
    print(f"Using session: {session_path}")

    data = MultiSessionOutput.from_dict(raw)

    generator = EvaluationTaskGenerator(LLMClient())
    preference_story = generator._build_preference_evolution_story(data)

    previous_events = [
        "The user is planning a month-long creative sabbatical alone in Japan.",
        "The user is preparing for a two-week, cross-country train journey across the United States.",
    ]
    previous_events_str = "\n".join(f"- {e}" for e in previous_events)

    prompt = render_prompt(
        "task_generation/evaluation_task_instruction",
        persona=data.persona,
        preference_evolution_story=preference_story,
        num_evolved_required=3,
        num_baseline_required=3,
        previous_events=previous_events_str,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "evaluation_task_instruction_v4")


if __name__ == "__main__":
    main()
