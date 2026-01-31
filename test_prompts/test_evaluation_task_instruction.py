#!/usr/bin/env python
"""Render evaluation_task_instruction prompt using latest session data."""

import json
import random
import sys

sys.path.insert(0, ".")

from persona_gym.client import LLMClient
from persona_gym.prompts import render_prompt
from persona_gym.schemas import MultiSessionOutput
from persona_gym.task_generators.evaluation_task import EvaluationTaskGenerator
from test_prompts._utils import load_latest_session, save_prompt


def main():
    raw, path = load_latest_session()
    print(f"Using session: {path}")

    data = MultiSessionOutput.from_dict(raw)

    generator = EvaluationTaskGenerator(LLMClient())
    preference_story = generator._build_preference_evolution_story(data)

    current_prefs = data.get_current_preferences()
    evolved_ids = data.get_evolved_preference_ids()
    evolutions = data.get_evolved_preferences()

    evolved_prefs = [p for p in current_prefs if p.preference_id in evolved_ids]
    baseline_prefs = [p for p in current_prefs if p.preference_id not in evolved_ids]

    random.seed(42)
    num_evolved = min(3, len(evolved_prefs))
    num_baseline = min(3, len(baseline_prefs))

    selected_evolved = random.sample(evolved_prefs, num_evolved)
    selected_baseline = random.sample(baseline_prefs, num_baseline)
    required_prefs = selected_evolved + selected_baseline

    selected_evolved_ids = {p.preference_id for p in selected_evolved}
    required_evolutions = [(old, new) for old, new in evolutions if new.preference_id in selected_evolved_ids]
    required_evolved_ids = {new.preference_id for _, new in required_evolutions}

    required_prefs_list = []
    for p in required_prefs:
        pref_info = {"id": p.preference_id, "fact": p.fact, "domain": p.domain}
        if p.preference_id in required_evolved_ids:
            pref_info["is_evolved"] = True
            for old, new in required_evolutions:
                if new.preference_id == p.preference_id:
                    pref_info["evolved_from"] = old.fact
                    break
        required_prefs_list.append(pref_info)

    prompt = render_prompt(
        "task_generation/evaluation_task_instruction",
        persona=data.persona,
        preference_evolution_story=preference_story,
        required_preferences=json.dumps(required_prefs_list, indent=2, ensure_ascii=False),
        num_required=len(required_prefs),
        num_evolved=num_evolved,
        num_baseline=num_baseline,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "evaluation_task_instruction")


if __name__ == "__main__":
    main()
