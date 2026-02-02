"""Shared utilities for prompt test scripts."""

import json
from pathlib import Path

from utils import extract_session_id, get_latest_session, get_latest_task_for_session


def load_latest_session():
    """Load the latest session data."""
    session_path = get_latest_session()
    if not session_path:
        raise FileNotFoundError("No session files found in outputs/sessions/")

    with open(session_path) as f:
        return json.load(f), session_path


def load_latest_task():
    """Load the latest task for the latest session."""
    session_path = get_latest_session()
    if not session_path:
        raise FileNotFoundError("No session files found in outputs/sessions/")

    session_id = extract_session_id(session_path)
    task_path = get_latest_task_for_session(session_id)
    if not task_path:
        raise FileNotFoundError(f"No task files found for session {session_id}")

    with open(task_path) as f:
        return json.load(f), task_path


def save_prompt(prompt: str, name: str):
    """Save rendered prompt to outputs/prompts/ directory."""
    out_path = Path("outputs/prompts") / f"{name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(prompt)
    print(f"Saved to {out_path}")
