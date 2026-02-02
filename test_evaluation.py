#!/usr/bin/env python
"""Test evaluation - Run agent dialogue and scoring on generated task.

Usage:
    # Single task (default: latest task, context agent)
    python test_evaluation.py --task tasks_01_01.json --agent context

    # All tasks for a session in parallel
    python test_evaluation.py --task all --agent context
    python test_evaluation.py --task all --agent nocontext

    # Specify session explicitly
    python test_evaluation.py --task all --agent context --session sessions_01.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from utils import (
    add_file_logging,
    extract_session_id,
    extract_task_num,
    get_all_tasks_for_session,
    get_eval_path,
    get_latest_session,
    get_latest_task_for_session,
    setup_logging,
    validate_task_session_match,
)

logger = setup_logging("evaluation")


def run_single_evaluation(
    session_path: Path,
    task_path: Path,
    agent_type: str,
    max_agent_turns: int,
):
    """Run evaluation for a single task."""
    from persona_gym.evaluation_multisession import run_evaluation
    from persona_gym.schemas import EvaluationTask, MultiSessionOutput

    session_id = extract_session_id(session_path)
    task_num = extract_task_num(task_path) or 1
    output_path = get_eval_path(session_id, task_num, agent_type)

    with open(session_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    with open(task_path, "r", encoding="utf-8") as f:
        raw_task = json.load(f)
    eval_task = EvaluationTask.from_dict(raw_task)

    include_history = agent_type == "context"

    result = run_evaluation(
        data,
        max_agent_turns=max_agent_turns,
        include_history=include_history,
        eval_task=eval_task,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation for task {task_num} completed")
    return result


async def run_all_evaluations(
    session_path: Path,
    task_paths: list[Path],
    agent_type: str,
    max_agent_turns: int,
):
    """Run evaluations for all tasks in parallel."""
    from persona_gym.evaluation_multisession import run_evaluations_parallel
    from persona_gym.schemas import EvaluationTask, MultiSessionOutput

    with open(session_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    include_history = agent_type == "context"

    contexts = []
    for task_path in task_paths:
        with open(task_path, "r", encoding="utf-8") as f:
            raw_task = json.load(f)
        eval_task = EvaluationTask.from_dict(raw_task)

        contexts.append({
            "multisession_data": data,
            "eval_task": eval_task,
            "include_history": include_history,
            "max_agent_turns": max_agent_turns,
            "task_path": task_path,
        })

    results = await run_evaluations_parallel(contexts)

    session_id = extract_session_id(session_path)
    for ctx, result in zip(contexts, results):
        task_path = ctx["task_path"]
        task_num = extract_task_num(task_path) or 1
        output_path = get_eval_path(session_id, task_num, agent_type)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation for task {task_num} completed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test evaluation system")
    parser.add_argument("--session", type=str, default=None,
                        help="Path to session file (default: latest)")
    parser.add_argument("--task", type=str, default=None,
                        help="Path to task file, or 'all' for all tasks (default: latest for session)")
    parser.add_argument("--agent", type=str, choices=["context", "nocontext"], default="context",
                        help="Agent type: context (full history) or nocontext (baseline)")
    parser.add_argument("--max-agent-turns", type=int, default=10,
                        help="Maximum agent turns in dialogue")
    args = parser.parse_args()

    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            logger.error(f"Session file not found: {session_path}")
            sys.exit(1)
    else:
        session_path = get_latest_session()
        if session_path is None:
            logger.error("No session files found in outputs/conversation/")
            sys.exit(1)

    session_id = extract_session_id(session_path)
    if session_id is None:
        logger.error(f"Could not extract session ID from: {session_path}")
        sys.exit(1)

    if args.task == "all":
        task_paths = get_all_tasks_for_session(session_id)
        if not task_paths:
            logger.error(f"No task files found for session: {session_id}")
            sys.exit(1)

        asyncio.run(run_all_evaluations(
            session_path,
            task_paths,
            args.agent,
            args.max_agent_turns,
        ))
    else:
        if args.task:
            task_path = Path(args.task)
            if not task_path.exists():
                logger.error(f"Task file not found: {task_path}")
                sys.exit(1)
            if not validate_task_session_match(session_path, task_path):
                logger.error("Task session ID does not match session ID")
                sys.exit(1)
        else:
            task_path = get_latest_task_for_session(session_id)
            if task_path is None:
                logger.error(f"No task files found for session: {session_id}")
                sys.exit(1)

        run_single_evaluation(
            session_path,
            task_path,
            args.agent,
            args.max_agent_turns,
        )


if __name__ == "__main__":
    add_file_logging(logger)
    main()
