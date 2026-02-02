#!/usr/bin/env python
"""Test task generation - Generate evaluation tasks from multi-session data.

Usage:
    python test_task_generation.py                    # Single task, latest session
    python test_task_generation.py --count 3          # Generate 3 tasks in parallel
    python test_task_generation.py --session <file>   # Use specific session
"""

import argparse
import asyncio
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


def log_task_details(task, task_num: int, output_path: Path):
    """Log details for a single task."""
    logger.info("-" * 40)
    logger.info(f"TASK {task_num}")
    logger.info("-" * 40)
    logger.info("")
    logger.info(f"Date: {task.evaluation_event.date}")
    logger.info(f"Event: {task.evaluation_event.event}")
    logger.info("")
    logger.info("User Prompt:")
    for line in task.user_prompt.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    logger.info(f"Required Preferences ({len(task.rubric.required_preferences)}):")
    for p in task.rubric.required_preferences:
        logger.info(f"  [{p['id']}] {p['fact'][:80]}...")
    logger.info("")
    logger.info(f"Saved to: {output_path}")
    logger.info("")


def generate_single(data, session_id: str, task_num: int):
    """Generate a single task synchronously."""
    from persona_gym.task_generators import generate_evaluation_task

    output_path = get_task_path(session_id, task_num)
    task = generate_evaluation_task(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

    log_task_details(task, task_num, output_path)
    return task


async def generate_multiple(data, session_id: str, start_task_num: int, count: int):
    """Generate multiple tasks in parallel."""
    from persona_gym.task_generators import generate_evaluation_tasks_parallel

    logger.info(f"Generating {count} tasks in parallel...")
    logger.info("")

    tasks = await generate_evaluation_tasks_parallel(data, num_tasks=count)

    for i, task in enumerate(tasks):
        task_num = start_task_num + i
        output_path = get_task_path(session_id, task_num)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

        log_task_details(task, task_num, output_path)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Test evaluation task generation")
    parser.add_argument("--session", type=str, default=None,
                        help="Path to session file (default: latest)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of tasks to generate (default: 1, uses parallel if > 1)")
    args = parser.parse_args()

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

    session_id = extract_session_id(input_path)
    if session_id is None:
        logger.error(f"Could not extract session ID from: {input_path}")
        logger.info("Expected filename format: sessions_XX.json")
        sys.exit(1)

    from persona_gym.schemas import MultiSessionOutput

    logger.info(f"Loading session data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)
    logger.info(f"Loaded {len(data.sessions)} sessions with {len(data.timeline.preferences)} preferences")
    logger.info("")

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

    logger.info("=" * 60)
    logger.info(f"GENERATING {args.count} EVALUATION TASK(S)")
    logger.info("=" * 60)
    logger.info("")

    start_task_num = get_next_task_num(session_id)

    if args.count == 1:
        generate_single(data, session_id, start_task_num)
    else:
        asyncio.run(generate_multiple(data, session_id, start_task_num, args.count))

    logger.info("=" * 60)
    logger.info("TASK GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")


if __name__ == "__main__":
    add_file_logging(logger)
    main()
