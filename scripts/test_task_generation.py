#!/usr/bin/env python
"""Test task generation - Generate evaluation tasks from multi-session data.

Usage:
    python test_task_generation.py                    # Single task, latest session
    python test_task_generation.py --count 3          # Generate 3 tasks sequentially
    python test_task_generation.py --session <path>   # Use specific session dir or file
    python test_task_generation.py --version v2       # Use specific task version
"""

import argparse
import json
import sys
import time
from pathlib import Path

from byom_bench.utils import (
    get_next_task_num,
    get_next_task_version,
    get_session_dir,
    get_session_path,
    get_task_path,
)


def generate_tasks(data, session_dir: Path, version: str, start_task_num: int, count: int):
    """Generate tasks — pure random selection, no LLM calls."""
    from byom_bench.task_generators import generate_evaluation_tasks

    tasks = generate_evaluation_tasks(data, num_tasks=count)

    for i, task in enumerate(tasks):
        task_num = start_task_num + i
        output_path = get_task_path(session_dir, task_num, version)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Test evaluation task generation")
    parser.add_argument("--session", type=str, default=None, help="Path to session dir or file (default: latest)")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of tasks to generate (default: 1, uses parallel if > 1)"
    )
    parser.add_argument(
        "--version", type=str, default=None, help="Task version (e.g., v2). If not provided, auto-increments."
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help="Outputs directory (e.g., outputs/).",
    )
    args = parser.parse_args()

    import byom_bench.utils as _utils
    _utils.OUTPUTS_DIR = Path(args.outputs_dir)

    session_dir = get_session_dir(args.session)
    if session_dir is None:
        print("No session found. Run test_data_generation.py first.")
        sys.exit(1)

    if args.version:
        version = args.version
    else:
        version = get_next_task_version(session_dir)

    session_file = get_session_path(session_dir)
    if not session_file.exists():
        print(f"Session file not found: {session_file}")
        sys.exit(1)

    from byom_bench.schemas import MultiSessionOutput

    with open(session_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    start_task_num = get_next_task_num(session_dir, version)
    generate_tasks(data, session_dir, version, start_task_num, args.count)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
