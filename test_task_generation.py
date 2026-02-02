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


def generate_single(data, session_id: str, task_num: int, total: int = 1):
    """Generate a single task synchronously."""
    from persona_gym.task_generators import generate_evaluation_task

    logger.info(f"Generated task {task_num}/{task_num + total - 1}")
    output_path = get_task_path(session_id, task_num)
    task = generate_evaluation_task(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

    return task


async def generate_multiple(data, session_id: str, start_task_num: int, count: int):
    """Generate multiple tasks in parallel."""
    from persona_gym.task_generators import generate_evaluation_tasks_parallel

    tasks = await generate_evaluation_tasks_parallel(data, num_tasks=count)

    for i, task in enumerate(tasks):
        task_num = start_task_num + i
        logger.info(f"Generated task {task_num}/{start_task_num + count - 1}")
        output_path = get_task_path(session_id, task_num)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

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
            sys.exit(1)

    session_id = extract_session_id(input_path)
    if session_id is None:
        logger.error(f"Could not extract session ID from: {input_path}")
        sys.exit(1)

    from persona_gym.schemas import MultiSessionOutput

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    start_task_num = get_next_task_num(session_id)

    if args.count == 1:
        generate_single(data, session_id, start_task_num)
    else:
        asyncio.run(generate_multiple(data, session_id, start_task_num, args.count))


if __name__ == "__main__":
    add_file_logging(logger)
    main()
