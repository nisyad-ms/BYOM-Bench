#!/usr/bin/env python
"""Render evaluation_task_instruction_v2 prompt using latest session data."""

import sys

sys.path.insert(0, ".")

from memory_gym.client import LLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import MultiSessionOutput
from memory_gym.task_generators.evaluation_task import EvaluationTaskGenerator
from test_prompts._utils import load_latest_session, save_prompt


def main():
    raw, path = load_latest_session()
    print(f"Using session: {path}")

    data = MultiSessionOutput.from_dict(raw)

    generator = EvaluationTaskGenerator(LLMClient())
    preference_story = generator._build_preference_evolution_story(data)

    prompt = render_prompt(
        "task_generation/evaluation_task_instruction_v2",
        persona=data.persona,
        preference_evolution_story=preference_story,
        num_evolved_required=3,
        num_baseline_required=3,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "evaluation_task_instruction_v2")


if __name__ == "__main__":
    main()
