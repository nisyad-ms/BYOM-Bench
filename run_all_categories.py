#!/usr/bin/env python
"""Run first 2 personas from each category in base_personas.json with N sessions each.

Usage:
    python run_all_categories.py --sessions 10
    python run_all_categories.py --sessions 10 --max-workers 5
    python run_all_categories.py --sessions 10 --retry   # Only run missing personas
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from test_data_generation import run_one

DATA_DIR = Path(__file__).parent / "data"
BASE_PERSONAS_FILE = DATA_DIR / "base_personas.json"
OUTPUTS_DIR = Path("outputs")


def get_personas(per_category: int = 2) -> list[tuple[str, str]]:
    """Return list of (category, persona) tuples — first N from each category."""
    with open(BASE_PERSONAS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    result = []
    for category, personas in data.items():
        for p in personas[:per_category]:
            result.append((category, p))
    return result


def get_completed_personas() -> set[str]:
    """Return set of base_persona strings that already have 10+ sessions."""
    completed = set()
    if not OUTPUTS_DIR.exists():
        return completed
    for d in OUTPUTS_DIR.iterdir():
        session_file = d / "sessions.json"
        if session_file.exists():
            try:
                data = json.loads(session_file.read_text())
                if len(data.get("sessions", [])) >= 10:
                    bp = data.get("expanded_persona", {}).get("base_persona", "")
                    completed.add(bp.strip())
            except Exception:
                pass
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--retry", action="store_true", help="Only run personas not yet completed")
    args = parser.parse_args()

    personas = get_personas(args.per_category)

    if args.retry:
        completed = get_completed_personas()
        personas = [(cat, p) for cat, p in personas if p.strip() not in completed]
        print(f"Retry mode: {len(completed)} already done, {len(personas)} remaining\n")

    if not personas:
        print("All personas already completed!")
        return

    print(f"Running {len(personas)} personas, {args.sessions} sessions each, max {args.max_workers} parallel\n")

    failed = []
    completed = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for category, persona in personas:
            future = executor.submit(run_one, persona, args.sessions)
            futures[future] = (category, persona)

        for future in as_completed(futures):
            category, persona = futures[future]
            try:
                session_dir = future.result()
                completed.append((category, persona, session_dir))
                print(f"  DONE [{len(completed)}/{len(personas)}] [{category}] {persona}\n  -> {session_dir}\n")
            except Exception as e:
                failed.append((category, persona, str(e)))
                print(f"  FAIL [{category}] {persona}\n  -> {e}\n")

    print(f"\n{'='*60}")
    print(f"Completed: {len(completed)}/{len(personas)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for cat, persona, err in failed:
            print(f"  [{cat}] {persona}: {err[:100]}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
