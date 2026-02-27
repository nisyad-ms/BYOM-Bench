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
    num_runs: int = 1,
    foundry_config: tuple[str, str, str] | None = None,
    memory_semaphore: asyncio.Semaphore | None = None,
):
    from memory_gym.evaluation_multisession import run_evaluations_parallel
    from memory_gym.schemas import EvaluationTaskSpec, MultiSessionOutput

    session_file = get_session_path(session_dir)

    with open(session_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    # Build list of pending tasks (skip already-completed ones)
    run_ids = list(range(1, num_runs + 1))

    pending_tasks: list[tuple[Path, EvaluationTaskSpec, int | None]] = []
    for task_path in task_paths:
        with open(task_path, "r", encoding="utf-8") as f:
            raw_task = json.load(f)
        eval_task = EvaluationTaskSpec.from_dict(raw_task)

        for run_id in run_ids:
            task_num = extract_task_num(task_path) or 1
            output_path = get_eval_path(eval_run_dir, task_num, agent_type, run_id)
            if output_path.exists():
                run_label = f" run {run_id}" if run_id else ""
                print(f"Skipping task {task_num:02d}{run_label}: already completed")
                continue
            pending_tasks.append((task_path, eval_task, run_id))

    if not pending_tasks:
        print(f"All tasks already completed for {session_dir.name}")
        return []

    print(f"{session_dir.name}: {len(pending_tasks)} tasks to run", flush=True)

    # Build agent context only if there are pending tasks
    memory_store_name = session_dir.name if agent_type in ("foundry", "aws") else None

    shared_foundry_agent = None
    shared_google_agent = None
    shared_aws_agent = None
    shared_foundry_local_agent = None
    session_start = time.time()
    try:
        if agent_type == "foundry":
            from memory_gym.agents import FoundryMemoryAgent

            endpoint, chat_model, emb_model = foundry_config or (None, None, None)
            shared_foundry_agent = FoundryMemoryAgent(
                memory_store_name=memory_store_name,  # type: ignore[arg-type]  # guaranteed str when agent_type="foundry"
                endpoint=endpoint,
                chat_model=chat_model,
                embedding_model=emb_model,
            )
            print(f"{session_dir.name}: building Foundry memory store...", flush=True)
            await asyncio.to_thread(shared_foundry_agent.build_context, data)

        if agent_type == "google":
            from memory_gym.agents import GoogleMemoryAgent

            shared_google_agent = GoogleMemoryAgent()
            print(f"{session_dir.name}: building Google memory store...", flush=True)
            if memory_semaphore:
                async with memory_semaphore:
                    await asyncio.to_thread(shared_google_agent.build_context, data)
            else:
                await asyncio.to_thread(shared_google_agent.build_context, data)

        if agent_type == "aws":
            from memory_gym.agents import AWSMemoryAgent

            shared_aws_agent = AWSMemoryAgent(memory_name=session_dir.name)
            print(f"{session_dir.name}: building AWS memory store...", flush=True)
            await asyncio.to_thread(shared_aws_agent.build_context, data)

        if agent_type == "foundry_local":
            from memory_gym.agents import FoundryLocalAgent

            shared_foundry_local_agent = FoundryLocalAgent(db_path=f"./.lancedb_foundry_local_{session_dir.name}")
            print(f"{session_dir.name}: building local Foundry memory store...", flush=True)
            await asyncio.to_thread(shared_foundry_local_agent.build_context, data)

        contexts = []
        for task_path, eval_task, run_id in pending_tasks:
            contexts.append(
                {
                    "multisession_data": data,
                    "eval_task": eval_task,
                    "agent_type": agent_type,
                    "memory_store_name": memory_store_name,
                    "max_agent_turns": max_agent_turns,
                    "task_path": task_path,
                    "foundry_agent": shared_foundry_agent,
                    "google_agent": shared_google_agent,
                    "aws_agent": shared_aws_agent,
                    "foundry_local_agent": shared_foundry_local_agent,
                    "run_id": run_id,
                }
            )

        def save_result(index: int, ctx: dict, result):
            task_path = ctx["task_path"]
            run_id = ctx["run_id"]
            task_num = extract_task_num(task_path) or 1
            output_path = get_eval_path(eval_run_dir, task_num, agent_type, run_id)
            run_label = f" run {run_id}" if run_id else ""
            print(
                f"{session_dir.name}: task {task_num:02d}{run_label} done "
                f"(pref={result.preference_score}, eff={result.efficiency_score})",
                flush=True,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        results = await run_evaluations_parallel(contexts, on_result=save_result)
        return results
    finally:
        if shared_foundry_agent is not None:
            await asyncio.to_thread(shared_foundry_agent.cleanup)
        if shared_google_agent is not None:
            await asyncio.to_thread(shared_google_agent.cleanup)
        if shared_aws_agent is not None:
            await asyncio.to_thread(shared_aws_agent.cleanup)
        if shared_foundry_local_agent is not None:
            await asyncio.to_thread(shared_foundry_local_agent.cleanup)
        session_elapsed = time.time() - session_start
        print(f"{session_dir.name}: completed in {session_elapsed:.1f}s")


async def run_all_sessions(
    session_dirs: list[Path],
    agent_type: str,
    max_agent_turns: int,
    num_runs: int,
    task_version: str | None,
    eval_run: str | None = None,
):
    foundry_configs: list[tuple[str, str, str]] = []
    if agent_type == "foundry":
        from memory_gym.agents import get_foundry_configs

        foundry_configs = get_foundry_configs()
        print(f"Foundry configs: {len(foundry_configs)} (endpoint, chat, embedding) triples")

    # Limit concurrent memory creation to avoid Google Vertex AI API rate limits
    memory_semaphore = asyncio.Semaphore(3) if agent_type == "google" else None

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

        if eval_run:
            eval_run_dir = session_dir / "evaluations" / eval_run
            eval_run_dir.mkdir(parents=True, exist_ok=True)
        else:
            eval_run_dir = create_eval_run_dir(session_dir)
        foundry_config = foundry_configs[i % len(foundry_configs)] if foundry_configs else None

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
                num_runs,
                foundry_config,
                memory_semaphore,
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
        choices=["context", "nocontext", "foundry", "google", "aws", "foundry_local"],
        default="context",
        help="Agent type: context, nocontext, foundry, google, aws, or foundry_local",
    )
    parser.add_argument("--max-agent-turns", type=int, default=10, help="Maximum agent turns in dialogue")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per task (default: 1)")
    parser.add_argument(
        "--eval-run",
        type=str,
        default=None,
        help="Resume into existing eval run (e.g., '2026-02-12_173909'). Skips already-completed tasks.",
    )
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
                args.num_runs,
                args.task_version,
                args.eval_run,
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

        if args.eval_run:
            eval_run_dir = session_dir / "evaluations" / args.eval_run
            eval_run_dir.mkdir(parents=True, exist_ok=True)
        else:
            eval_run_dir = create_eval_run_dir(session_dir)

        asyncio.run(
            run_session_evals(
                session_dir,
                task_paths,
                args.agent,
                args.max_agent_turns,
                resolved_task_version,
                eval_run_dir,
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
