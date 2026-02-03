#!/usr/bin/env python
"""Render preference_judge_instruction prompt using actual task and evaluation data."""

import json
import sys

sys.path.insert(0, ".")

from memory_gym.prompts import render_prompt
from test_prompts._utils import load_latest_task, save_prompt


def load_latest_evaluation():
    """Load the latest evaluation result if available."""
    from pathlib import Path

    from utils import extract_session_id, get_latest_session

    session_path = get_latest_session()
    if not session_path:
        return None, None

    session_id = extract_session_id(session_path)
    eval_dir = Path("outputs/evaluations")
    if not eval_dir.exists():
        return None, None

    pattern = f"eval_{session_id}_*.json"
    eval_files = sorted(eval_dir.glob(pattern), reverse=True)
    if not eval_files:
        eval_files = sorted(eval_dir.glob("eval_*.json"), reverse=True)

    if eval_files:
        with open(eval_files[0]) as f:
            return json.load(f), eval_files[0]
    return None, None


def main():
    task_data, task_path = load_latest_task()
    print(f"Using task: {task_path}")

    eval_data, eval_path = load_latest_evaluation()
    if eval_data and "conversation" in eval_data:
        transcript = eval_data["conversation"]
        print(f"Using evaluation: {eval_path}")
    else:
        transcript = [
            {
                "role": "user",
                "content": task_data.get("evaluation_event", {}).get("user_prompt", "I need help with something."),
            },
            {
                "role": "assistant",
                "content": "I'd be happy to help! Based on your interest in public video gaming history exhibits, would you like to incorporate that theme?",
            },
            {"role": "user", "content": "Yes, that sounds great! It's also my daughter's birthday weekend."},
            {
                "role": "assistant",
                "content": "Perfect! I know you prefer larger, inclusive gatherings now, so perhaps we could combine the workshop with birthday celebrations?",
            },
        ]
        print("No evaluation found, using sample transcript based on task")

    required_prefs = task_data.get("rubric", {}).get("required_preferences", [])

    prompt = render_prompt(
        "evaluation/preference_judge_instruction",
        required_preferences=json.dumps(required_prefs, indent=2, ensure_ascii=False),
        transcript=json.dumps(transcript, indent=2, ensure_ascii=False),
        num_required=len(required_prefs),
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "preference_judge_instruction")


if __name__ == "__main__":
    main()
