#!/usr/bin/env python
"""Gather all evaluation results into an Excel file with detail and summary sheets."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook

from utils import (
    get_all_session_dirs,
    get_eval_run_dir,
    get_latest_eval_run_dir,
)


def main():
    parser = argparse.ArgumentParser(description="Gather all evaluation results into an Excel file")
    parser.add_argument(
        "--eval-run",
        type=str,
        help="Optional eval timestamp (e.g., 2026-02-09_143022). If not provided, use latest eval run per session.",
    )
    args = parser.parse_args()

    rows = []
    eval_run_dirs: list[Path] = []

    session_dirs = get_all_session_dirs()
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
            match = re.match(r"eval_(\d+)_(\w+?)(?:_(\d+))?\.json", eval_file.name)
            if not match:
                continue

            task_num = match.group(1)
            agent = match.group(2)
            run_id = match.group(3) or "1"

            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            scores = data.get("scores", {})
            pref = data.get("preference_scoring", {})
            eff = data.get("efficiency_scoring", {})

            rows.append(
                {
                    "session": session_dir.name,
                    "eval_run": eval_run_name,
                    "task": task_num,
                    "agent": agent,
                    "run": run_id,
                    "preference_score": scores.get("preference_score", ""),
                    "efficiency_score": scores.get("efficiency_score", ""),
                    "proactive_count": pref.get("proactive_count", ""),
                    "stale_count": pref.get("stale_count", ""),
                    "required_preferences": len(data.get("rubric", {}).get("required_preferences", [])),
                    "total_turns": eff.get("total_turns", ""),
                    "productive_turns": eff.get("productive_turns", ""),
                    "generic_turns": eff.get("generic_turns", eff.get("clarifying_turns", "")),
                    "correction_turns": eff.get("correction_turns", ""),
                    "ignored_turns": eff.get("ignored_turns", ""),
                }
            )

    if not rows:
        print("No evaluation files found.")
        sys.exit(1)

    wb = Workbook()

    ws1 = wb.active
    ws1.title = "results"
    fieldnames = list(rows[0].keys())
    ws1.append(fieldnames)
    for row in rows:
        ws1.append([row[f] for f in fieldnames])

    scores_by_key: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["session"], row["eval_run"], row["task"])
        scores_by_key[key][row["agent"]].append(float(row["preference_score"]))

    ws2 = wb.create_sheet("summary_results")
    ws2.append(["session", "eval_run", "task", "context", "foundry", "nocontext"])
    for (session, eval_run, task), agents in sorted(scores_by_key.items()):
        ws2.append(
            [
                session,
                eval_run,
                task,
                round(sum(agents.get("context", [0])) / max(len(agents.get("context", [0])), 1), 2),
                round(sum(agents.get("foundry", [0])) / max(len(agents.get("foundry", [0])), 1), 2),
                round(sum(agents.get("nocontext", [0])) / max(len(agents.get("nocontext", [0])), 1), 2),
            ]
        )

    for eval_run_dir in eval_run_dirs:
        output_file = eval_run_dir / "results.xlsx"
        wb.save(output_file)
        print(f"  Saved to {output_file}")
    print(f"Wrote {len(rows)} results to {len(eval_run_dirs)} eval run(s) (2 sheets)")


if __name__ == "__main__":
    main()
