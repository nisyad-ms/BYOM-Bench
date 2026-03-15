#!/usr/bin/env python
"""Re-run the preference judge on existing evaluation JSON files.

Overwrites preference scoring in-place. Does NOT regenerate dialogues.

Usage:
    # Dry run — show what would be re-judged
    python scripts/rerun_judge.py --session all --eval-run 2026-03-05_163038 --dry-run

    # Re-judge one session, all agents
    python scripts/rerun_judge.py --session 2026-03-02_093705_374604 --eval-run 2026-03-05_163038

    # Re-judge specific agent and tasks
    python scripts/rerun_judge.py --session all --eval-run 2026-03-05_163038 --agent context --task 01,03
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from memory_gym.client import PooledLLMClient
from memory_gym.evaluation_multisession.judge import (
    MultiSessionJudge,
    _extract_simulator_verdicts,
)
from memory_gym.utils import (
    EVAL_PATTERN,
    get_all_session_dirs,
    get_eval_run_dir,
    get_session_dir,
)


def _discover_eval_files(
    session_dirs: list[Path],
    eval_run: str,
    agent_filter: str | None,
    task_filter: set[int] | None,
) -> list[Path]:
    """Discover eval JSON files matching the filters."""
    files: list[Path] = []
    for session_dir in session_dirs:
        run_dir = get_eval_run_dir(session_dir, eval_run)
        if run_dir is None:
            continue
        for f in sorted(run_dir.glob("eval_*.json")):
            match = EVAL_PATTERN.match(f.name)
            if not match:
                continue
            task_num = int(match.group(1))
            agent = match.group(2)
            if agent_filter and agent != agent_filter:
                continue
            if task_filter and task_num not in task_filter:
                continue
            files.append(f)
    return files


def _reconstruct_required_preferences(preference_verdicts: list[dict]) -> list[dict]:
    """Reconstruct required_preferences from enriched preference_verdicts."""
    prefs = []
    for v in preference_verdicts:
        pref: dict = {
            "id": v["preference_id"],
            "fact": v.get("preference", ""),
        }
        if v.get("type") == "evolved" and "supersedes" in v:
            pref["supersedes"] = v["supersedes"]
        prefs.append(pref)
    return prefs


def _rejudge_file(eval_path: Path, judge: MultiSessionJudge) -> dict | None:
    """Re-run preference judge on a single eval file. Returns summary dict or None on error."""
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_pref_scoring = data.get("preference_scoring", {})
    old_verdicts = old_pref_scoring.get("preference_verdicts", [])
    if not old_verdicts:
        print(f"  SKIP {eval_path.name}: no preference_verdicts")
        return None

    old_score = data.get("scores", {}).get("preference_score", 0.0)

    # Reconstruct inputs
    required_preferences = _reconstruct_required_preferences(old_verdicts)
    conversation = data.get("conversation", [])

    # Extract corrected simulator verdicts from conversation scratchpads
    simulator_verdicts = _extract_simulator_verdicts(conversation, required_preferences)
    simulator_verdicts_json = json.dumps(simulator_verdicts, indent=2, ensure_ascii=False)

    required_prefs_json = json.dumps(required_preferences, indent=2, ensure_ascii=False)
    transcript_json = json.dumps(conversation, indent=2, ensure_ascii=False)
    num_required = len(required_preferences)

    # Call preference judge
    pref_result = judge._call_preference_judge(
        required_prefs_json, transcript_json, num_required, simulator_verdicts_json
    )

    # Calculate new scores
    new_verdicts = pref_result.get("preference_verdicts", [])

    # Enrich new verdicts with preference content and type
    pref_lookup = {p["id"]: p for p in required_preferences}
    for entry in new_verdicts:
        pref_id = entry.get("preference_id")
        if pref_id and pref_id in pref_lookup:
            pref = pref_lookup[pref_id]
            entry["preference"] = pref["fact"]
            if "supersedes" in pref:
                entry["type"] = "evolved"
                entry["supersedes"] = pref["supersedes"]
            else:
                entry["type"] = "baseline"

    recalled_count, stale_count, new_score = judge._calculate_preference_score(new_verdicts)

    new_preference_usage = {
        entry["preference_id"]: entry.get("final_verdict", "missed")
        for entry in new_verdicts
        if entry.get("preference_id")
    }
    stale_used = [
        entry["preference_id"]
        for entry in new_verdicts
        if entry.get("stale_used", False) and entry.get("preference_id")
    ]

    # Update in-place
    data["scores"]["preference_score"] = new_score
    data["preference_scoring"] = {
        "recalled_count": recalled_count,
        "stale_count": stale_count,
        "stale_preference_usage": stale_used,
        "preference_verdicts": new_verdicts,
    }
    data["preference_usage"] = new_preference_usage

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "file": eval_path.name,
        "old_score": old_score,
        "new_score": new_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-run preference judge on existing eval files")
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session name or 'all'",
    )
    parser.add_argument(
        "--eval-run",
        type=str,
        required=True,
        help="Eval run timestamp (e.g., '2026-03-05_163038')",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Filter to specific agent type (optional)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to specific task numbers, comma-separated (e.g., '01,03')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be re-judged without making changes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=25,
        help="Number of parallel workers (default: 25, one per deployment)",
    )
    args = parser.parse_args()

    # Resolve sessions
    if args.session == "all":
        session_dirs = get_all_session_dirs()
    else:
        sd = get_session_dir(args.session)
        if sd is None:
            print(f"Session not found: {args.session}")
            sys.exit(1)
        session_dirs = [sd]

    # Parse task filter
    task_filter: set[int] | None = None
    if args.task:
        task_filter = {int(t.strip()) for t in args.task.split(",")}

    # Discover files
    eval_files = _discover_eval_files(session_dirs, args.eval_run, args.agent, task_filter)
    if not eval_files:
        print("No matching eval files found.")
        sys.exit(1)

    print(f"Found {len(eval_files)} eval file(s) to re-judge")

    if args.dry_run:
        for f in eval_files:
            # Show session/file path relative to outputs/
            rel = f.relative_to(Path("outputs"))
            print(f"  {rel}")
        print(f"\nDry run: {len(eval_files)} files would be re-judged")
        return

    # Run in parallel
    client = PooledLLMClient()
    judge = MultiSessionJudge(client)
    start = time.time()

    changed = 0
    total = 0
    errors = 0
    counter_lock = threading.Lock()
    completed_count = 0

    def _process_file(eval_path: Path) -> tuple[Path, dict | None, str | None]:
        """Process a single file. Returns (path, result_dict, error_msg)."""
        try:
            result = _rejudge_file(eval_path, judge)
            return eval_path, result, None
        except Exception as e:
            return eval_path, None, str(e)

    num_files = len(eval_files)
    print(f"Running with {args.workers} workers...", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, p): p for p in eval_files}
        for future in as_completed(futures):
            eval_path, result, error = future.result()
            rel = eval_path.relative_to(Path("outputs"))

            with counter_lock:
                completed_count += 1
                idx = completed_count

            if error:
                print(f"[{idx}/{num_files}] {rel} ... ERROR: {error}", flush=True)
                errors += 1
                continue

            if result is None:
                continue

            total += 1
            delta = result["new_score"] - result["old_score"]
            sign = "+" if delta >= 0 else ""
            print(
                f"[{idx}/{num_files}] {rel} ... "
                f"{result['old_score']:.2f} -> {result['new_score']:.2f} ({sign}{delta:.2f})",
                flush=True,
            )
            if delta != 0:
                changed += 1

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s: {total} re-judged, {changed} changed, {errors} errors")


if __name__ == "__main__":
    main()
