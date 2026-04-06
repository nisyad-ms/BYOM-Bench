#!/usr/bin/env python
"""Gather all evaluation results into an Excel file with detail and summary sheets.

Metrics:
- preference_recall: (recalled - stale) / total  (from preference_score)
- stale_rate: stale_count / evolved_count per task
- task_completion: 1 if preference_score == 1.0, else 0

Summary uses macro-averaging: runs → task → session → overall.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook

from ream_bench.utils import (
    EVAL_PATTERN,
    SESSION_DIR_PATTERN,
    get_eval_run_dir,
    get_latest_eval_run_dir,
)


def _compute_stale_rate(pref_scoring: dict) -> float | None:
    """Compute stale_rate = stale_count / evolved_count from preference scoring data.

    Returns None if there are no evolved preferences (avoids division by zero).
    """
    verdicts = pref_scoring.get("preference_verdicts", pref_scoring.get("first_mention_trace")) or []
    evolved_count = sum(1 for v in verdicts if v.get("type") == "evolved")
    if evolved_count == 0:
        return None
    stale_count = pref_scoring.get("stale_count", 0)
    return stale_count / evolved_count


def main():
    parser = argparse.ArgumentParser(description="Gather all evaluation results into an Excel file")
    parser.add_argument(
        "--eval-run",
        type=str,
        help="Optional eval timestamp (e.g., 2026-02-09_143022). If not provided, use latest eval run per session.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help="Outputs directory containing session folders (e.g., outputs/ or outputs.v0.4/).",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    eval_run_dirs: list[Path] = []

    outputs_path = Path(args.outputs_dir)
    session_dirs = sorted(d for d in outputs_path.iterdir() if d.is_dir() and SESSION_DIR_PATTERN.match(d.name))
    for session_dir in sorted(session_dirs):
        if args.eval_run:
            eval_run_dir = get_eval_run_dir(session_dir, args.eval_run)
            if not eval_run_dir:
                continue
        else:
            eval_run_dir = get_latest_eval_run_dir(session_dir)
            if not eval_run_dir:
                continue

        eval_run_dirs.append(eval_run_dir)
        eval_run_name = eval_run_dir.name

        for eval_file in sorted(eval_run_dir.glob("eval_*.json")):
            match = EVAL_PATTERN.match(eval_file.name)
            if not match:
                continue

            task_num = match.group(1)
            agent = match.group(2)
            run_id = match.group(3) or "1"

            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            scores = data.get("scores", {})
            pref_scoring = data.get("preference_scoring", {})

            preference_score = scores.get("preference_score", 0.0)
            stale_rate = _compute_stale_rate(pref_scoring)
            task_completion = 1 if preference_score == 1.0 else 0

            rows.append(
                {
                    "session": session_dir.name,
                    "eval_run": eval_run_name,
                    "task": task_num,
                    "agent": agent,
                    "run": run_id,
                    "preference_score": preference_score,
                    "stale_rate": stale_rate if stale_rate is not None else "",
                    "task_completion": task_completion,
                }
            )

    if not rows:
        print("No evaluation files found.")
        sys.exit(1)

    wb = Workbook()

    # --- Detail sheet ---
    ws1 = wb.active
    assert ws1 is not None
    ws1.title = "results"
    fieldnames = list(rows[0].keys())
    ws1.append(fieldnames)
    for row in rows:
        ws1.append([row[f] for f in fieldnames])

    # --- Summary sheet: macro-averaged metrics per agent ---
    # Step 1: group rows by (agent, session, task) → list of run-level values
    run_data: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        run_data[row["agent"]][row["session"]][row["task"]].append(row)

    # Step 2: macro-average: runs → task mean → session mean → overall mean
    ws2 = wb.create_sheet("summary")
    ws2.append(["agent", "preference_recall", "stale_rate", "task_completion", "n_sessions"])

    for agent in sorted(run_data.keys()):
        session_pref_scores: list[float] = []
        session_stale_rates: list[float] = []
        session_completions: list[float] = []

        for session in run_data[agent]:
            task_pref_scores: list[float] = []
            task_stale_rates: list[float] = []
            task_completions: list[float] = []

            for task in run_data[agent][session]:
                runs = run_data[agent][session][task]

                # Average runs within a task
                run_prefs = [r["preference_score"] for r in runs if r["preference_score"] != ""]
                if run_prefs:
                    task_pref_scores.append(sum(run_prefs) / len(run_prefs))

                run_stales = [r["stale_rate"] for r in runs if r["stale_rate"] != ""]
                if run_stales:
                    task_stale_rates.append(sum(run_stales) / len(run_stales))

                run_completions = [r["task_completion"] for r in runs]
                task_completions.append(sum(run_completions) / len(run_completions))

            # Average tasks within a session
            if task_pref_scores:
                session_pref_scores.append(sum(task_pref_scores) / len(task_pref_scores))
            if task_stale_rates:
                session_stale_rates.append(sum(task_stale_rates) / len(task_stale_rates))
            if task_completions:
                session_completions.append(sum(task_completions) / len(task_completions))

        # Average sessions → overall
        n_sessions = len(run_data[agent])
        avg_pref = round(sum(session_pref_scores) / len(session_pref_scores), 4) if session_pref_scores else ""
        avg_stale = round(sum(session_stale_rates) / len(session_stale_rates), 4) if session_stale_rates else ""
        avg_comp = round(sum(session_completions) / len(session_completions), 4) if session_completions else ""

        ws2.append([agent, avg_pref, avg_stale, avg_comp, n_sessions])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    combined_output = outputs_path / f"results_{timestamp}.xlsx"
    wb.save(combined_output)
    print(f"Saved to {combined_output}")
    print(f"Wrote {len(rows)} results from {len(eval_run_dirs)} session(s) (2 sheets: results, summary)")


if __name__ == "__main__":
    main()
