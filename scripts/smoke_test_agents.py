#!/usr/bin/env python
"""Smoke test for refactored agents.

Runs 1 task / 1 run against each agent type using the tiny debug dataset.
Skips google (needs manual intervention for auth).

Usage:
    uv run python scripts/smoke_test_agents.py
    uv run python scripts/smoke_test_agents.py --agent context
    uv run python scripts/smoke_test_agents.py --agent foundry_local
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from memory_gym.agents.stores import get_available_agent_types, get_store_class

DATA_DIR = Path("data/debug_data")
SESSIONS_FILE = DATA_DIR / "sessions.json"
TASK_FILE = DATA_DIR / "task_01.json"

AGENT_TYPES = ["context", "nocontext"] + get_available_agent_types()


async def run_smoke(agent_type: str) -> None:
    from memory_gym.agents import MemoryAgent
    from memory_gym.evaluation_multisession import run_evaluation
    from memory_gym.schemas import EvaluationTaskSpec, MultiSessionOutput

    with open(SESSIONS_FILE, encoding="utf-8") as f:
        data = MultiSessionOutput.from_dict(json.load(f))
    with open(TASK_FILE, encoding="utf-8") as f:
        task = EvaluationTaskSpec.from_dict(json.load(f))

    # Memory agents get pre-built and passed via `agent=`; baselines use `agent_type=`
    agent = None
    store_cls = get_store_class(agent_type)
    if store_cls is not None:
        store = store_cls(session_dir=Path(f".debug_smoke/{agent_type}"))
        agent = MemoryAgent(store)
    elif agent_type not in ("context", "nocontext"):
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Running smoke test: {agent_type}")
    print(f"{'=' * 60}")

    t0 = time.time()

    try:
        # For memory agents: build_context up front (mirrors test_evaluation.py)
        if agent is not None:
            print("  Building context...", flush=True)
            await asyncio.to_thread(agent.build_context, data)

        # run_evaluation — baselines pass agent_type, memory agents pass agent
        print("  Running evaluation (1 task, max 5 turns)...", flush=True)
        result = await asyncio.to_thread(
            run_evaluation,
            multisession_data=data,
            eval_task=task,
            max_agent_turns=5,
            agent_type=agent_type if agent is None else "context",  # unused when agent is set
            agent=agent,
        )

        elapsed = time.time() - t0
        print(f"  pref_score={result.preference_score}, eff_score={result.efficiency_score}")
        print(f"  turns={result.total_turns}, elapsed={elapsed:.1f}s")

        # Save eval result to debug_data for inspection (overwrites previous run)
        output_path = DATA_DIR / f"eval_{agent_type}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"  Saved eval to {output_path}")

        print("  PASSED")
    finally:
        if agent is not None:
            print("  Cleaning up...", flush=True)
            await asyncio.to_thread(agent.cleanup)


async def main():
    parser = argparse.ArgumentParser(description="Smoke test refactored agents")
    parser.add_argument("--agent", type=str, default=None, help="Single agent to test (or all if omitted)")
    args = parser.parse_args()

    if not SESSIONS_FILE.exists():
        print(f"Missing {SESSIONS_FILE}. Run from repo root.")
        sys.exit(1)

    agents = [args.agent] if args.agent else AGENT_TYPES
    failed = []

    for agent_type in agents:
        try:
            await run_smoke(agent_type)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(agent_type)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(agents)} agent(s) passed.")


if __name__ == "__main__":
    asyncio.run(main())
