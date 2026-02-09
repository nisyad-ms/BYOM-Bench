#!/usr/bin/env python
"""Test evaluation - Run agent dialogue and scoring on generated task.

Usage:
    # All sessions, all tasks
    python test_evaluation.py --session all --agent context --num-runs 3

    # One session, all tasks
    python test_evaluation.py --session 2026-02-02_1414 --agent foundry --num-runs 3

    # One session, specific tasks
    python test_evaluation.py --session 2026-02-02_1414 --task 01,02,03 --agent context
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import (
    create_eval_run_dir,
    extract_task_num,
    get_all_session_dirs,
    get_all_tasks,
    get_eval_path,
    get_latest_task_version,
    get_session_dir,
    get_session_path,
    get_tasks_by_nums,
    save_eval_run_config,
)


async def run_session_evals(
    session_dir: Path,
    task_paths: list[Path],
    agent_type: str,
    max_agent_turns: int,
    task_version: str,
    eval_run_dir: Path,
    force_recreate_memory: bool = False,
    num_runs: int = 1,
    embedding_model: str | None = None,
):
    from memory_gym.evaluation_multisession import run_evaluations_parallel
    from memory_gym.schemas import EvaluationTask, MultiSessionOutput

    session_file = get_session_path(session_dir)

    with open(session_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    memory_store_name = session_dir.name if agent_type == "foundry" else None

    shared_foundry_agent = None
    if agent_type == "foundry":
        from memory_gym.agents import FoundryMemoryAgent

        shared_foundry_agent = FoundryMemoryAgent(
            memory_store_name=memory_store_name,
            embedding_model=embedding_model,
        )
        shared_foundry_agent.build_context(data, force_recreate=force_recreate_memory)

    run_ids = list(range(1, num_runs + 1)) if num_runs > 1 else [None]

    contexts = []
    for task_path in task_paths:
        with open(task_path, "r", encoding="utf-8") as f:
            raw_task = json.load(f)
        eval_task = EvaluationTask.from_dict(raw_task)

        for run_id in run_ids:
            contexts.append(
                {
                    "multisession_data": data,
                    "eval_task": eval_task,
                    "agent_type": agent_type,
                    "memory_store_name": memory_store_name,
                    "max_agent_turns": max_agent_turns,
                    "task_path": task_path,
                    "foundry_agent": shared_foundry_agent,
                    "run_id": run_id,
                }
            )

    def save_result(index: int, ctx: dict, result):
        task_path = ctx["task_path"]
        run_id = ctx["run_id"]
        task_num = extract_task_num(task_path) or 1
        output_path = get_eval_path(eval_run_dir, task_num, agent_type, run_id)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    results = await run_evaluations_parallel(contexts, on_result=save_result)
    return results


async def run_all_sessions(
    session_dirs: list[Path],
    agent_type: str,
    max_agent_turns: int,
    force_recreate_memory: bool,
    num_runs: int,
    task_version: str | None,
):
    embedding_models = ["text-embedding-3-small-001"]
    if agent_type == "foundry":
        from memory_gym.agents import get_foundry_embedding_models

        embedding_models = get_foundry_embedding_models()

    eval_configs: list[tuple[Path, dict]] = []
    tasks = []
    for i, session_dir in enumerate(session_dirs):
        session_file = get_session_path(session_dir)
        if not session_file.exists():
            print(f"Skipping {session_dir.name}: no sessions.json")
            continue

        resolved_version = task_version
        if resolved_version is None:
            resolved_version = get_latest_task_version(session_dir)
            if resolved_version is None:
                print(f"Skipping {session_dir.name}: no task files")
                continue

        task_paths = get_all_tasks(session_dir, resolved_version)
        if not task_paths:
            print(f"Skipping {session_dir.name}: no task files")
            continue

        eval_run_dir = create_eval_run_dir(session_dir)
        emb_model = embedding_models[i % len(embedding_models)] if agent_type == "foundry" else None

        eval_configs.append(
            (
                eval_run_dir,
                {
                    "agent_type": agent_type,
                    "task_version": resolved_version,
                    "num_runs": num_runs,
                    "max_agent_turns": max_agent_turns,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

        tasks.append(
            run_session_evals(
                session_dir,
                task_paths,
                agent_type,
                max_agent_turns,
                resolved_version,
                eval_run_dir,
                force_recreate_memory,
                num_runs,
                emb_model,
            )
        )

    await asyncio.gather(*tasks)

    for eval_run_dir, config in eval_configs:
        save_eval_run_config(eval_run_dir, config)


def main():
    parser = argparse.ArgumentParser(description="Test evaluation system")
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session name (e.g., 2026-02-02_1414) or 'all' for every session",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="'all' for all tasks, or comma-separated task numbers (e.g., '01,02,03'). Only with single session.",
    )
    parser.add_argument(
        "--task-version",
        type=str,
        default=None,
        help="Task version (e.g., 'v1', 'v2'). Defaults to latest version.",
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
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per task (default: 1)")
    args = parser.parse_args()

    if args.session == "all":
        if args.task != "all":
            print("--task cannot be used with --session all")
            sys.exit(1)

        session_dirs = get_all_session_dirs()
        if not session_dirs:
            print("No sessions found in outputs/")
            sys.exit(1)

        asyncio.run(
            run_all_sessions(
                session_dirs,
                args.agent,
                args.max_agent_turns,
                args.no_cache,
                args.num_runs,
                args.task_version,
            )
        )
    else:
        session_dir = get_session_dir(args.session)
        if session_dir is None:
            print(f"Session not found: {args.session}")
            sys.exit(1)

        session_file = get_session_path(session_dir)
        if not session_file.exists():
            print(f"Session file not found: {session_file}")
            sys.exit(1)

        resolved_task_version = args.task_version
        if resolved_task_version is None:
            resolved_task_version = get_latest_task_version(session_dir)
            if resolved_task_version is None:
                print("No task files found. Run test_task_generation.py first.")
                sys.exit(1)

        if args.task == "all":
            task_paths = get_all_tasks(session_dir, resolved_task_version)
        else:
            task_paths = get_tasks_by_nums(session_dir, args.task, resolved_task_version)

        if not task_paths:
            print("No task files found. Run test_task_generation.py first.")
            sys.exit(1)

        eval_run_dir = create_eval_run_dir(session_dir)

        asyncio.run(
            run_session_evals(
                session_dir,
                task_paths,
                args.agent,
                args.max_agent_turns,
                resolved_task_version,
                eval_run_dir,
                args.no_cache,
                args.num_runs,
            )
        )

        config = {
            "agent_type": args.agent,
            "task_version": resolved_task_version,
            "num_runs": args.num_runs,
            "max_agent_turns": args.max_agent_turns,
            "timestamp": datetime.now().isoformat(),
        }
        save_eval_run_config(eval_run_dir, config)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
