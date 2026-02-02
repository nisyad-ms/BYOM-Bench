#!/usr/bin/env python
"""Test task generation - Generate evaluation tasks from multi-session data.

Usage:
    python test_task_generation.py                    # Single task, latest session
    python test_task_generation.py --count 3          # Generate 3 tasks in parallel
    python test_task_generation.py --session <path>   # Use specific session dir or file
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from utils import (
    add_file_logging,
    get_next_task_num,
    get_session_dir,
    get_session_path,
    get_task_path,
    setup_logging,
)

logger = setup_logging("task_generation")


def generate_single(data, session_dir: Path, task_num: int):
    """Generate a single task synchronously."""
    from persona_gym.task_generators import generate_evaluation_task

    task = generate_evaluation_task(data)
    output_path = get_task_path(session_dir, task_num)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Generated task {task_num}")
    return task


async def generate_multiple(data, session_dir: Path, start_task_num: int, count: int):
    """Generate multiple tasks in parallel."""
    from persona_gym.task_generators import generate_evaluation_tasks_parallel

    tasks = await generate_evaluation_tasks_parallel(data, num_tasks=count)

    for i, task in enumerate(tasks):
        task_num = start_task_num + i
        output_path = get_task_path(session_dir, task_num)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Generated task {task_num}/{start_task_num + count - 1}")

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Test evaluation task generation")
    parser.add_argument("--session", type=str, default=None,
                        help="Path to session dir or file (default: latest)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of tasks to generate (default: 1, uses parallel if > 1)")
    args = parser.parse_args()

    session_dir = get_session_dir(args.session)
    if session_dir is None:
        logger.error("No session found. Run test_data_generation.py first.")
        sys.exit(1)

    add_file_logging(logger, session_dir)

    session_file = get_session_path(session_dir)
    if not session_file.exists():
        logger.error(f"Session file not found: {session_file}")
        sys.exit(1)

    from persona_gym.schemas import MultiSessionOutput

    with open(session_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    start_task_num = get_next_task_num(session_dir)

    if args.count == 1:
        generate_single(data, session_dir, start_task_num)
    else:
        asyncio.run(generate_multiple(data, session_dir, start_task_num, args.count))


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
