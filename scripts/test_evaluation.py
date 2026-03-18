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
from typing import Any

from memory_gym.agents.stores import get_available_agent_types
from memory_gym.prompts import _load_prompt_config
from memory_gym.utils import (
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


def _get_prompt_versions() -> dict[str, str]:
    """Return the active prompt version map from configs/prompts.yaml."""
    config = _load_prompt_config()
    return {k: v if v else "(default)" for k, v in config.items()}


def _create_memory_agent(
    agent_type: str,
    session_dir: Path,
    foundry_config: tuple[str, str, str] | None = None,
    sentinel_dir: Path | None = None,
) -> Any:
    """Create a MemoryAgent wrapping the appropriate store for *agent_type*.

    Returns None for baseline agents (context / nocontext) — those are
    created inside the runner.
    """
    from memory_gym.agents import MemoryAgent

    # Special case: foundry needs endpoint config from multi-endpoint discovery
    if agent_type == "foundry":
        from memory_gym.agents import FoundryMemoryStore

        endpoint, chat_model, emb_model = foundry_config or (None, None, None)
        store = FoundryMemoryStore(
            memory_store_name=session_dir.name,
            endpoint=endpoint,
            chat_model=chat_model,
            embedding_model=emb_model,
        )
        return MemoryAgent(store)

    # Generic: look up store class from registry
    from memory_gym.agents.stores import get_store_class

    store_cls = get_store_class(agent_type)
    if store_cls is not None:
        store = store_cls(session_dir=session_dir, sentinel_dir=sentinel_dir, session_name=session_dir.name)
        return MemoryAgent(store)

    # Baseline agents (context / nocontext) — handled by the runner
    return None


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
    sentinel_dir: Path | None = None,
    memory_token_budget: int | None = None,
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
        return [], None

    print(f"{session_dir.name}: {len(pending_tasks)} tasks to run", flush=True)

    shared_agent = _create_memory_agent(agent_type, session_dir, foundry_config, sentinel_dir)
    build_context_seconds: float | None = None
    session_start = time.time()
    try:
        if shared_agent is not None:
            print(f"{session_dir.name}: building {agent_type} memory store...", flush=True)
            t0 = time.monotonic()
            if memory_semaphore:
                async with memory_semaphore:
                    await asyncio.to_thread(shared_agent.build_context, data)
            else:
                await asyncio.to_thread(shared_agent.build_context, data)
            build_context_seconds = round(time.monotonic() - t0, 2)
            print(f"{session_dir.name}: build_context completed in {build_context_seconds:.1f}s", flush=True)

        contexts = []
        for task_path, eval_task, run_id in pending_tasks:
            contexts.append(
                {
                    "multisession_data": data,
                    "eval_task": eval_task,
                    "agent_type": agent_type,
                    "max_agent_turns": max_agent_turns,
                    "task_path": task_path,
                    "agent": shared_agent,
                    "run_id": run_id,
                    "memory_token_budget": memory_token_budget,
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
                f"(pref={result.preference_score}, "
                f"time={result.eval_seconds}s)",
                flush=True,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        results = await run_evaluations_parallel(contexts, on_result=save_result)
        return results, build_context_seconds
    finally:
        if shared_agent is not None:
            if sentinel_dir is None:
                await asyncio.to_thread(shared_agent.cleanup)
            else:
                print(f"{session_dir.name}: skipping cleanup (--reuse-stores)")
        session_elapsed = time.time() - session_start
        print(f"{session_dir.name}: completed in {session_elapsed:.1f}s")


async def run_all_sessions(
    session_dirs: list[Path],
    agent_type: str,
    max_agent_turns: int,
    num_runs: int,
    task_version: str | None,
    eval_run: str | None = None,
    sentinel_dir: Path | None = None,
    memory_token_budget: int | None = None,
):
    foundry_configs: list[tuple[str, str, str]] = []
    if agent_type == "foundry":
        from memory_gym.agents import get_foundry_configs

        foundry_configs = get_foundry_configs()
        print(f"Foundry configs: {len(foundry_configs)} (endpoint, chat, embedding) triples")

    # Limit concurrency to avoid cloud API rate/quota limits.
    # Google: semaphore on build_context only (retrieval quota is generous).
    # AWS/mem0/foundry_local: semaphore on entire session.
    from memory_gym.client import get_agent_config

    agent_cfg = (
        get_agent_config(agent_type) if agent_type in ("google", "aws", "mem0", "mem0_graph", "foundry_local") else {}
    )
    memory_semaphore = (
        asyncio.Semaphore(agent_cfg.get("max_concurrent_build_context", 3)) if agent_type == "google" else None
    )
    session_semaphore: asyncio.Semaphore | None = None
    if agent_type in ("aws", "mem0", "mem0_graph", "foundry_local"):
        session_semaphore = asyncio.Semaphore(agent_cfg.get("max_concurrent_sessions", 10))

    eval_configs: list[tuple[Path, dict, str]] = []
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
                    "memory_token_budget": memory_token_budget,
                    "timestamp": datetime.now().isoformat(),
                    "prompt_versions": _get_prompt_versions(),
                },
                session_dir.name,
            )
        )

        coro = run_session_evals(
            session_dir,
            task_paths,
            agent_type,
            max_agent_turns,
            resolved_version,
            eval_run_dir,
            num_runs,
            foundry_config,
            memory_semaphore,
            sentinel_dir,
            memory_token_budget,
        )

        if session_semaphore is not None:

            async def _gated(sem: asyncio.Semaphore, c):  # noqa: E501
                async with sem:
                    return await c

            coro = _gated(session_semaphore, coro)

        tasks.append(coro)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            print(f"Session {eval_configs[i][2]} FAILED: {type(result).__name__}: {result}")
        else:
            _, build_seconds = result
            if build_seconds is not None:
                eval_configs[i][1]["build_context_seconds"] = build_seconds

    for eval_run_dir, config, _ in eval_configs:
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
        choices=["context", "nocontext", "foundry"] + get_available_agent_types(),
        default="context",
        help="Agent type: context, nocontext, foundry, or any autodiscovered store",
    )
    parser.add_argument("--max-agent-turns", type=int, default=20, help="Maximum agent turns in dialogue")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per task (default: 1)")
    parser.add_argument(
        "--eval-run",
        type=str,
        default=None,
        help="Resume into existing eval run (e.g., '2026-02-12_173909'). Skips already-completed tasks.",
    )
    parser.add_argument(
        "--reuse-stores",
        action="store_true",
        help="Reuse existing memory stores if sentinel is valid. Skips cleanup so stores persist.",
    )
    parser.add_argument(
        "--memory-token-budget",
        type=int,
        default=None,
        help="Max tokens of retrieved memories per retrieval call. If set, truncates retrieved facts to this budget.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Override outputs directory (default: outputs/). E.g., outputs.v0.4/",
    )
    args = parser.parse_args()

    if args.outputs_dir:
        import memory_gym.utils as _utils

        _utils.OUTPUTS_DIR = Path(args.outputs_dir)

    sentinel_dir = Path(".memory_sentinels") if args.reuse_stores else None
    if sentinel_dir is not None:
        sentinel_dir.mkdir(exist_ok=True)

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
                sentinel_dir,
                args.memory_token_budget,
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

        _, build_context_seconds = asyncio.run(
            run_session_evals(
                session_dir,
                task_paths,
                args.agent,
                args.max_agent_turns,
                resolved_task_version,
                eval_run_dir,
                args.num_runs,
                sentinel_dir=sentinel_dir,
                memory_token_budget=args.memory_token_budget,
            )
        )

        config: dict[str, Any] = {
            "agent_type": args.agent,
            "task_version": resolved_task_version,
            "num_runs": args.num_runs,
            "max_agent_turns": args.max_agent_turns,
            "memory_token_budget": args.memory_token_budget,
            "timestamp": datetime.now().isoformat(),
            "prompt_versions": _get_prompt_versions(),
        }
        if build_context_seconds is not None:
            config["build_context_seconds"] = build_context_seconds
        save_eval_run_config(eval_run_dir, config)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
