#!/usr/bin/env python
"""Test evaluation - Run agent dialogue and scoring on generated task.

Usage:
    # Single task (default: latest task, context agent)
    python test_evaluation.py --agent context

    # All tasks for a session in parallel
    python test_evaluation.py --session 2026-02-02_1414 --task all --agent context
    python test_evaluation.py --session 2026-02-02_1414 --task all --agent nocontext
    python test_evaluation.py --session 2026-02-02_1414 --task all --agent foundry
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from utils import (
    add_file_logging,
    extract_task_num,
    get_all_tasks,
    get_eval_path,
    get_latest_task,
    get_session_dir,
    get_session_path,
    setup_logging,
    validate_task_session_match,
)

logger = setup_logging("evaluation")


def run_single_evaluation(
    session_dir: Path,
    task_path: Path,
    agent_type: str,
    max_agent_turns: int,
    force_recreate_memory: bool = False,
    run_id: int | None = None,
):
    """Run evaluation for a single task."""
    from memory_gym.evaluation_multisession import run_evaluation
    from memory_gym.schemas import EvaluationTask, MultiSessionOutput

    task_num = extract_task_num(task_path) or 1
    output_path = get_eval_path(session_dir, task_num, agent_type, run_id)
    session_file = get_session_path(session_dir)

    with open(session_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    with open(task_path, "r", encoding="utf-8") as f:
        raw_task = json.load(f)
    eval_task = EvaluationTask.from_dict(raw_task)

    memory_store_name = session_dir.name if agent_type == "foundry" else None

    result = run_evaluation(
        data,
        max_agent_turns=max_agent_turns,
        eval_task=eval_task,
        agent_type=agent_type,
        memory_store_name=memory_store_name,
        force_recreate_memory=force_recreate_memory,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation for task {task_num} completed")
    return result


async def run_all_evaluations(
    session_dir: Path,
    task_paths: list[Path],
    agent_type: str,
    max_agent_turns: int,
    run_id: int | None = None,
):
    """Run evaluations for all tasks in parallel."""
    from memory_gym.evaluation_multisession import run_evaluations_parallel
    from memory_gym.schemas import EvaluationTask, MultiSessionOutput

    session_file = get_session_path(session_dir)

    with open(session_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    memory_store_name = session_dir.name if agent_type == "foundry" else None

    contexts = []
    for task_path in task_paths:
        with open(task_path, "r", encoding="utf-8") as f:
            raw_task = json.load(f)
        eval_task = EvaluationTask.from_dict(raw_task)

        contexts.append(
            {
                "multisession_data": data,
                "eval_task": eval_task,
                "agent_type": agent_type,
                "memory_store_name": memory_store_name,
                "max_agent_turns": max_agent_turns,
                "task_path": task_path,
            }
        )

    def save_result(index: int, ctx: dict, result):
        """Save result immediately when evaluation completes."""
        task_path = ctx["task_path"]
        task_num = extract_task_num(task_path) or 1
        output_path = get_eval_path(session_dir, task_num, agent_type, run_id)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation for task {task_num} (run {run_id or 'single'}) completed")

    results = await run_evaluations_parallel(contexts, on_result=save_result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test evaluation system")
    parser.add_argument(
        "--session", type=str, default=None, help="Session name (e.g., 2026-02-02_1414). Default: latest"
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Path to task file, or 'all' for all tasks (default: latest)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["context", "nocontext", "foundry"],
        default="context",
        help="Agent type: context, nocontext, or foundry",
    )
    parser.add_argument("--max-agent-turns", type=int, default=10, help="Maximum agent turns in dialogue")
    parser.add_argument(
        "--no-cache", action="store_true", help="Force recreate memory store from scratch (foundry agent only)"
    )
    parser.add_argument(
        "--run", type=int, default=None, help="Run ID for multiple evaluation runs (e.g., 1, 2, 3)"
    )
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

    if args.task == "all":
        task_paths = get_all_tasks(session_dir)
        if not task_paths:
            logger.error("No task files found. Run test_task_generation.py first.")
            sys.exit(1)

        asyncio.run(
            run_all_evaluations(
                session_dir,
                task_paths,
                args.agent,
                args.max_agent_turns,
                args.run,
            )
        )
    else:
        if args.task:
            task_path = Path(args.task)
            if not task_path.exists():
                logger.error(f"Task file not found: {task_path}")
                sys.exit(1)
            if not validate_task_session_match(session_dir, task_path):
                logger.error("Task does not belong to the specified session")
                sys.exit(1)
        else:
            task_path = get_latest_task(session_dir)
            if task_path is None:
                logger.error("No task files found. Run test_task_generation.py first.")
                sys.exit(1)

        run_single_evaluation(
            session_dir,
            task_path,
            args.agent,
            args.max_agent_turns,
            args.no_cache,
            args.run,
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
