#!/usr/bin/env python
"""Test task generation - Generate evaluation tasks from multi-session data.

Usage:
    python test_task_generation.py                    # Uses latest session, creates new task
    python test_task_generation.py --session <file>   # Use specific session
"""

import argparse
import json
import sys
from pathlib import Path

from utils import (
    add_file_logging,
    extract_session_id,
    get_latest_session,
    get_next_task_num,
    get_task_path,
    setup_logging,
)

logger = setup_logging("task_generation")


def main():
    parser = argparse.ArgumentParser(description="Test evaluation task generation")
    parser.add_argument("--session", type=str, default=None,
                        help="Path to session file (default: latest)")
    args = parser.parse_args()

    # Find session file
    if args.session:
        input_path = Path(args.session)
        if not input_path.exists():
            logger.error(f"Session file not found: {input_path}")
            sys.exit(1)
    else:
        input_path = get_latest_session()
        if input_path is None:
            logger.error("No session files found in outputs/conversation/")
            logger.info("Run test_data_generation.py first")
            sys.exit(1)
        logger.info(f"Using latest session: {input_path.name}")

    # Extract session ID and determine output path
    session_id = extract_session_id(input_path)
    if session_id is None:
        logger.error(f"Could not extract session ID from: {input_path}")
        logger.info("Expected filename format: sessions_XX.json")
        sys.exit(1)

    # Get next task number for this session
    task_num = get_next_task_num(session_id)
    output_path = get_task_path(session_id, task_num)

    from persona_gym.schemas import MultiSessionOutput
    from persona_gym.task_generators import generate_evaluation_task

    # Load multi-session data
    logger.info(f"Loading session data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)
    logger.info(f"Loaded {len(data.sessions)} sessions with {len(data.timeline.preferences)} preferences")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # Display current vs stale preferences
    current_prefs = data.get_current_preferences()
    stale_prefs = data.get_superseded_preferences()

    logger.info("=" * 60)
    logger.info("PREFERENCE ANALYSIS")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Current preferences ({len(current_prefs)}):")
    logger.info("")
    for p in current_prefs:
        logger.info(f"  [{p.preference_id}] [{p.domain}]")
        logger.info(f"    {p.fact}")
        logger.info("")

    logger.info(f"Stale/superseded preferences ({len(stale_prefs)}):")
    logger.info("")
    for p in stale_prefs:
        logger.info(f"  [{p.preference_id}] [{p.domain}] -> superseded by {p.superseded_by}")
        logger.info(f"    {p.fact}")
        logger.info("")

    # Generate evaluation task
    logger.info("=" * 60)
    logger.info("GENERATING EVALUATION TASK")
    logger.info("=" * 60)
    logger.info("")
    task = generate_evaluation_task(data)

    # Display task details
    logger.info("-" * 40)
    logger.info("EVALUATION EVENT")
    logger.info("-" * 40)
    logger.info("")
    logger.info(f"Date: {task.evaluation_event.date}")
    logger.info("")
    logger.info("Event:")
    logger.info(f"  {task.evaluation_event.event}")
    logger.info("")
    logger.info("Domain:")
    logger.info(f"  {task.evaluation_event.domain}")
    logger.info("")
    if task.evaluation_event.task_internal:
        logger.info("Task (internal for judge):")
        for line in task.evaluation_event.task_internal.split('\n'):
            logger.info(f"  {line}")
        logger.info("")

    logger.info("-" * 40)
    logger.info("USER PROMPT")
    logger.info("-" * 40)
    logger.info("")
    for line in task.user_prompt.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    logger.info("-" * 40)
    logger.info("EVALUATION RUBRIC")
    logger.info("-" * 40)
    logger.info("")
    logger.info(f"Required Preferences ({len(task.rubric.required_preferences)}):")
    for p in task.rubric.required_preferences:
        logger.info(f"  [{p['id']}] {p['fact']}")
        if "supersedes" in p:
            logger.info(f"    ↳ supersedes: {p['supersedes']['fact']}")
    logger.info("")

    logger.info("-" * 40)
    logger.info("PERSONA SUMMARY (for user simulator)")
    logger.info("-" * 40)
    logger.info("")
    for line in task.persona_summary.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    # Save task
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Task saved to: {output_path}")
    logger.info("")

    logger.info("=" * 60)
    logger.info("TASK GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")


if __name__ == "__main__":
    add_file_logging(logger)
    main()
