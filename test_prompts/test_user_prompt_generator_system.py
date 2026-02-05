"""Test script to render user_prompt_generator_system prompt (v1 and v2)."""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from memory_gym.prompts import render_prompt
from utils import get_latest_session_dir, get_latest_task


def save_prompt(prompt: str, name: str):
    """Save rendered prompt to outputs/prompts/ directory."""
    out_path = Path("outputs/prompts") / f"{name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(prompt)
    print(f"Saved to {out_path}")


def main():
    import json

    session_dir = get_latest_session_dir()
    if not session_dir:
        print("No session found")
        return

    task_path = get_latest_task(session_dir)
    if not task_path:
        print("No task found")
        return

    with open(task_path) as f:
        task_data = json.load(f)
    print(f"Loaded task from: {task_path}")

    persona_summary = task_data.get("persona_summary", "Unknown persona")
    evaluation_event = task_data["evaluation_event"]["event"]
    required_preferences = task_data["rubric"]["required_preferences"]

    prefs_formatted = "\n".join(
        f"- [{p.get('id', 'unknown')}] {p.get('fact', '')}" for p in required_preferences
    )

    prompt_v1 = render_prompt(
        "task_generation/user_prompt_generator_system",
        persona_summary=persona_summary,
        evaluation_event=evaluation_event,
        required_preferences=prefs_formatted,
    )
    print("\n=== V1 (Specific Question) ===")
    print(prompt_v1)
    save_prompt(prompt_v1, "user_prompt_generator_system_v1")

    prompt_v2 = render_prompt(
        "task_generation/user_prompt_generator_system_v2",
        persona_summary=persona_summary,
        evaluation_event=evaluation_event,
        required_preferences=prefs_formatted,
    )
    print("\n=== V2 (Broad Composite) ===")
    print(prompt_v2)
    save_prompt(prompt_v2, "user_prompt_generator_system_v2")


if __name__ == "__main__":
    main()
