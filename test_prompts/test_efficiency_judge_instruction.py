#!/usr/bin/env python
"""Render efficiency_judge_instruction prompt using actual task and evaluation data."""

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
            {"role": "assistant", "content": "I'd be happy to help! What specifically do you need?"},
            {"role": "user", "content": "Can you suggest some activities for this weekend?"},
            {"role": "assistant", "content": "Sure! What kind of activities do you enjoy?"},
            {"role": "user", "content": "I prefer larger gatherings with family and friends now."},
            {"role": "assistant", "content": "Great, I'll plan for a larger event then."},
        ]
        print("No evaluation found, using sample transcript based on task")

    required_prefs = task_data.get("rubric", {}).get("required_preferences", [])
    agent_turns = sum(1 for t in transcript if t["role"] == "assistant")

    prompt = render_prompt(
        "evaluation/efficiency_judge_instruction",
        required_preferences=json.dumps(required_prefs, indent=2, ensure_ascii=False),
        transcript=json.dumps(transcript, indent=2, ensure_ascii=False),
        agent_turns=agent_turns,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "efficiency_judge_instruction")


if __name__ == "__main__":
    main()
