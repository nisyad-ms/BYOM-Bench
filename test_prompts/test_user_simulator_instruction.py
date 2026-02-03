#!/usr/bin/env python
"""Render user_simulator_instruction prompt using actual evaluation data."""

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
        conversation = eval_data["conversation"]
        print(f"Using evaluation: {eval_path}")
    else:
        user_prompt = task_data.get("evaluation_event", {}).get("user_prompt", "I need help planning my weekend.")
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "I'd be happy to help! What kind of activities are you thinking of?"},
        ]
        print("No evaluation found, using sample conversation based on task")

    conv_formatted = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation)

    agent_message = ""
    for turn in reversed(conversation):
        if turn["role"] == "assistant":
            agent_message = turn["content"]
            break

    prompt = render_prompt(
        "evaluation/user_simulator_instruction",
        conversation=conv_formatted,
        agent_message=agent_message,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "user_simulator_instruction")


if __name__ == "__main__":
    main()
