#!/usr/bin/env python
"""Gather all evaluation results into an Excel file with detail and summary sheets."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook

OUTPUTS_DIR = Path("outputs")


def main():
    output_file = Path("results.xlsx")
    rows = []

    for session_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        eval_dir = session_dir / "evaluation"
        if not eval_dir.exists():
            continue

        for eval_file in sorted(eval_dir.glob("eval_*.json")):
            match = re.match(r"eval_(\d+)_(\w+?)_(\d+)\.json", eval_file.name)
            if not match:
                continue

            task_num = match.group(1)
            agent = match.group(2)
            run_id = match.group(3)

            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            scores = data.get("scores", {})
            pref = data.get("preference_scoring", {})
            eff = data.get("efficiency_scoring", {})

            rows.append({
                "session": session_dir.name,
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
                "clarifying_turns": eff.get("clarifying_turns", ""),
                "correction_turns": eff.get("correction_turns", ""),
                "ignored_turns": eff.get("ignored_turns", ""),
            })

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

    scores_by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["session"], row["task"])
        scores_by_key[key][row["agent"]].append(float(row["preference_score"]))

    ws2 = wb.create_sheet("summary_results")
    ws2.append(["session", "task", "context", "foundry", "nocontext"])
    for (session, task), agents in sorted(scores_by_key.items()):
        ws2.append([
            session,
            task,
            round(sum(agents.get("context", [0])) / max(len(agents.get("context", [0])), 1), 2),
            round(sum(agents.get("foundry", [0])) / max(len(agents.get("foundry", [0])), 1), 2),
            round(sum(agents.get("nocontext", [0])) / max(len(agents.get("nocontext", [0])), 1), 2),
        ])

    wb.save(output_file)
    print(f"Wrote {len(rows)} results to {output_file} (2 sheets)")


if __name__ == "__main__":
    main()
