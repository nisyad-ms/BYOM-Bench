"""Test script to generate user prompts using v1 and v2 templates."""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from memory_gym.client import LLMClient
from memory_gym.prompts import render_prompt
from utils import get_latest_session_dir, get_latest_task


def generate_user_prompt(client, prompt_version: str, persona_summary: str, evaluation_event: str, prefs_formatted: str) -> str:
    """Generate a user prompt using the specified template version."""
    system_prompt = render_prompt(
        f"task_generation/user_prompt_generator_system{prompt_version}",
        persona_summary=persona_summary,
        evaluation_event=evaluation_event,
        required_preferences=prefs_formatted,
        use_config=False,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate the user's opening message."},
    ]

    response = client.complete_chat(messages=messages, max_tokens=256)
    return response.strip()


def main():
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
    print(f"Loaded task from: {task_path}\n")

    persona_summary = task_data.get("persona_summary", "Unknown persona")
    evaluation_event = task_data["evaluation_event"]["event"]
    required_preferences = task_data["rubric"]["required_preferences"]

    prefs_formatted = "\n".join(
        f"- [{p.get('id', 'unknown')}] {p.get('fact', '')}" for p in required_preferences
    )

    print("=" * 80)
    print("EVALUATION EVENT:")
    print(evaluation_event)
    print("=" * 80)
    print("\nREQUIRED PREFERENCES:")
    for p in required_preferences:
        print(f"  - [{p.get('id')}] {p.get('fact', '')[:80]}...")
    print("=" * 80)

    client = LLMClient()

    print("\n>>> Generating with V1 (Specific Question)...")
    user_prompt_v1 = generate_user_prompt(client, "", persona_summary, evaluation_event, prefs_formatted)
    print("\n=== V1 OUTPUT ===")
    print(user_prompt_v1)

    print("\n>>> Generating with V2 (Broad Composite)...")
    user_prompt_v2 = generate_user_prompt(client, "_v2", persona_summary, evaluation_event, prefs_formatted)
    print("\n=== V2 OUTPUT ===")
    print(user_prompt_v2)

    out_dir = Path("outputs/prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "user_prompt_v1_output.txt", "w") as f:
        f.write(user_prompt_v1)
    with open(out_dir / "user_prompt_v2_output.txt", "w") as f:
        f.write(user_prompt_v2)
    print(f"\nSaved outputs to {out_dir}/")


if __name__ == "__main__":
    main()
