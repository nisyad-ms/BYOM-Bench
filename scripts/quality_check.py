#!/usr/bin/env python
"""Post-evaluation quality and bug check for MemoryGym results.

Parses all eval files once, then runs modular checks on:
1. User simulator behavior (role reversal, early termination, state management)
2. Preference judge behavior (override frequency, stale accuracy, judge quality)

Usage:
    uv run python scripts/quality_check.py --eval-run 2026-03-16_215407 --outputs-dir outputs.v0.4
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

# ---------- Constants ----------

SESSION_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(_\d{6})?$")
EVAL_PATTERN = re.compile(r"^eval_(\d{2})_(\w+?)(?:_(\d{2}))?\.json$")

ROLE_REVERSAL_PATTERNS = [
    (re.compile(r"\bI(?:'d)?\s+(?:suggest|recommend)\b", re.IGNORECASE), "direct_advice"),
    (re.compile(r"\byou\s+should\s+(?:try|consider|start|look|check|use|switch)\b", re.IGNORECASE), "directive"),
    (re.compile(r"(?:^|\n)\s*(?:\d+[.\)]\s|[-*]\s).*(?:^|\n)\s*(?:\d+[.\)]\s|[-*]\s)", re.MULTILINE), "bullet_list"),
    (re.compile(r"\bhere(?:'s| is| are)\s+(?:what|how|a tip|some|my)\b", re.IGNORECASE), "here_is"),
    (re.compile(r"\?[^?]*\b(?:I think|probably|maybe you could|what works for me)\b", re.IGNORECASE), "self_answer"),
]

STALE_FP_SIGNALS = [
    re.compile(r"neither version", re.IGNORECASE),
    re.compile(r"did not (?:apply|use|reference) the (?:old|outdated|superseded)", re.IGNORECASE),
    re.compile(r"generic advice", re.IGNORECASE),
    re.compile(r"no (?:evidence|indication) of (?:stale|old|outdated)", re.IGNORECASE),
    re.compile(r"\bCase 3\b"),
]


# ---------- Data model ----------


@dataclass
class EvalRecord:
    session: str
    task_num: str
    agent: str
    run: str
    filepath: Path
    preference_score: float
    preference_verdicts: list[dict]
    recalled_count: int
    stale_count: int
    required_pref_ids: list[str]
    n_required: int
    n_evolved: int
    conversation: list[dict]
    user_turns: list[dict]
    assistant_turns: list[dict]
    error: str | None


# ---------- Phase 1: Data extraction ----------


def load_all_evals(outputs_dir: Path, eval_run: str) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    sessions = sorted(d.name for d in outputs_dir.iterdir() if d.is_dir() and SESSION_DIR_PATTERN.match(d.name))

    print(f"Loading from {outputs_dir}/ with eval_run={eval_run}")
    print(f"Sessions found: {len(sessions)}")

    errors = 0
    skipped_no_eval = 0

    for session in sessions:
        session_dir = outputs_dir / session
        eval_dir = session_dir / "evaluations" / eval_run
        if not eval_dir.is_dir():
            skipped_no_eval += 1
            continue

        # Load task files for this session
        task_lookup: dict[str, dict] = {}
        tasks_dir = session_dir / "tasks" / "v1"
        if tasks_dir.is_dir():
            for tf in tasks_dir.glob("task_*.json"):
                with open(tf, encoding="utf-8") as f:
                    task_data = json.load(f)
                task_lookup[task_data["task_id"]] = task_data

        for eval_file in sorted(eval_dir.glob("eval_*.json")):
            match = EVAL_PATTERN.match(eval_file.name)
            if not match:
                continue

            task_num = match.group(1)
            agent = match.group(2)
            run_id = match.group(3) or "1"

            with open(eval_file, encoding="utf-8") as f:
                data = json.load(f)

            error = data.get("error")
            if error:
                errors += 1

            scores = data.get("scores", {})
            pref_scoring = data.get("preference_scoring", {})
            conversation = data.get("conversation", [])

            # Get required prefs from task file
            task_id = data.get("task_id", "")
            task_data = task_lookup.get(task_id, {})
            rubric = task_data.get("rubric", {})
            required_prefs = rubric.get("required_preferences", [])
            required_pref_ids = [p["id"] for p in required_prefs]
            n_evolved = sum(1 for p in required_prefs if p.get("supersedes"))

            user_turns = [t for t in conversation if t.get("role") == "user"]
            assistant_turns = [t for t in conversation if t.get("role") == "assistant"]

            records.append(
                EvalRecord(
                    session=session,
                    task_num=task_num,
                    agent=agent,
                    run=run_id,
                    filepath=eval_file,
                    preference_score=scores.get("preference_score", 0.0),
                    preference_verdicts=pref_scoring.get("preference_verdicts") or [],
                    recalled_count=pref_scoring.get("recalled_count", 0),
                    stale_count=pref_scoring.get("stale_count", 0),
                    required_pref_ids=required_pref_ids,
                    n_required=len(required_pref_ids),
                    n_evolved=n_evolved,
                    conversation=conversation,
                    user_turns=user_turns,
                    assistant_turns=assistant_turns,
                    error=error,
                )
            )

    # Print loading summary
    agent_counts = Counter(r.agent for r in records)
    print(f"Loaded: {len(records)} eval files, {errors} with errors, {skipped_no_eval} sessions without eval dir")
    print(f"Per agent: {dict(sorted(agent_counts.items()))}")
    print()

    return records


# ---------- Phase 2: User simulator checks ----------


def check_role_reversal(records: list[EvalRecord], results: dict) -> None:
    print("=" * 70)
    print("2.1 ROLE REVERSAL DETECTION")
    print("=" * 70)

    total_user_turns = 0
    flagged_turns: list[dict] = []
    per_agent: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "flagged": 0})
    pattern_counts: Counter = Counter()

    for rec in records:
        for i, turn in enumerate(rec.user_turns):
            total_user_turns += 1
            per_agent[rec.agent]["total"] += 1
            content = turn.get("content", "")

            for pattern, label in ROLE_REVERSAL_PATTERNS:
                if pattern.search(content):
                    per_agent[rec.agent]["flagged"] += 1
                    pattern_counts[label] += 1
                    flagged_turns.append({
                        "file": f"{rec.session}/eval_{rec.task_num}_{rec.agent}_{rec.run}.json",
                        "turn_idx": i,
                        "pattern": label,
                        "content": content[:300],
                        "agent": rec.agent,
                    })
                    break  # One flag per turn

    flagged_count = len(flagged_turns)
    rate = flagged_count / total_user_turns if total_user_turns else 0
    print(f"Total user turns: {total_user_turns}")
    print(f"Flagged: {flagged_count} ({rate:.2%})")
    print(f"Pattern breakdown: {dict(pattern_counts.most_common())}")
    print()

    print("Per-agent rates:")
    for agent in sorted(per_agent):
        a = per_agent[agent]
        r = a["flagged"] / a["total"] if a["total"] else 0
        print(f"  {agent:15s}: {a['flagged']:4d}/{a['total']:5d} ({r:.2%})")
    print()

    print("Top 10 flagged examples:")
    for ex in flagged_turns[:10]:
        print(f"  [{ex['pattern']}] {ex['file']}")
        print(f"    {ex['content'][:200]}")
        print()

    results["role_reversal_flagged"] = flagged_count
    results["role_reversal_total"] = total_user_turns


def check_early_termination(records: list[EvalRecord], results: dict) -> None:
    print("=" * 70)
    print("2.2 EARLY TERMINATION ANALYSIS")
    print("=" * 70)

    untested_evals: list[dict] = []
    turn_counts_by_agent: dict[str, list[int]] = defaultdict(list)
    max_turns_hit = 0

    for rec in records:
        if rec.error:
            continue

        n_assistant_turns = len(rec.assistant_turns)
        turn_counts_by_agent[rec.agent].append(n_assistant_turns)

        verdict_pref_ids = {v["preference_id"] for v in rec.preference_verdicts}
        required_set = set(rec.required_pref_ids)
        untested = required_set - verdict_pref_ids

        if untested:
            reason = "max_turns" if n_assistant_turns >= 20 else "early_exit"
            untested_evals.append({
                "file": f"{rec.session}/eval_{rec.task_num}_{rec.agent}_{rec.run}.json",
                "agent": rec.agent,
                "untested": untested,
                "n_untested": len(untested),
                "n_required": rec.n_required,
                "assistant_turns": n_assistant_turns,
                "reason": reason,
            })
            if n_assistant_turns >= 20:
                max_turns_hit += 1

    print(f"Evals with untested preferences: {len(untested_evals)} (excl. error files)")
    print(f"  Hit max_agent_turns (20): {max_turns_hit}")
    print(f"  Early exit: {len(untested_evals) - max_turns_hit}")
    print()

    # Per-agent breakdown
    agent_untested = Counter(e["agent"] for e in untested_evals)
    agent_totals = Counter(r.agent for r in records if not r.error)
    print("Per-agent untested rates:")
    for agent in sorted(set(agent_untested) | set(agent_totals)):
        u = agent_untested.get(agent, 0)
        t = agent_totals.get(agent, 0)
        r = u / t if t else 0
        print(f"  {agent:15s}: {u:4d}/{t:5d} ({r:.2%})")
    print()

    # Turn count distribution
    print("Assistant turn count distribution (mean / median / min / max):")
    for agent in sorted(turn_counts_by_agent):
        counts = turn_counts_by_agent[agent]
        if counts:
            print(
                f"  {agent:15s}: {mean(counts):.1f} / {median(counts):.1f} / {min(counts)} / {max(counts)}"
            )
    print()

    if untested_evals:
        print("First 10 evals with untested preferences:")
        for e in untested_evals[:10]:
            print(f"  {e['file']} ({e['reason']}): {e['n_untested']}/{e['n_required']} untested, {e['assistant_turns']} agent turns")
        print()

    results["early_termination_count"] = len(untested_evals)


def check_state_management(records: list[EvalRecord], results: dict) -> None:
    print("=" * 70)
    print("2.3 STATE MANAGEMENT BUG DETECTION")
    print("=" * 70)

    violations: dict[str, list[dict]] = {
        "duplicate_probe": [],
        "order_continuity": [],
        "proactive_retested": [],
    }

    for rec in records:
        if rec.error:
            continue
        file_label = f"{rec.session}/eval_{rec.task_num}_{rec.agent}_{rec.run}.json"

        # Check 1: No duplicate current_pref_id
        current_pref_ids = [t.get("current_pref_id") for t in rec.user_turns if t.get("current_pref_id")]
        duplicates = [pid for pid, count in Counter(current_pref_ids).items() if count > 1]
        if duplicates:
            violations["duplicate_probe"].append({"file": file_label, "duplicates": duplicates})

        # Check 3: Testing order continuity (current should equal prev turn's next)
        prev_next = None
        for turn in rec.user_turns:
            current = turn.get("current_pref_id")
            if prev_next is not None and current is not None and current != prev_next:
                violations["order_continuity"].append({
                    "file": file_label,
                    "expected": prev_next,
                    "got": current,
                })
            prev_next = turn.get("next_pref_id")

        # Check 4: Proactive recall not re-tested
        proactively_recalled: set[str] = set()
        for turn in rec.user_turns:
            scratchpad = turn.get("scratchpad", {})
            if isinstance(scratchpad, dict):
                proactive = scratchpad.get("proactive_recall") or []
                proactively_recalled.update(proactive)

        for pid in proactively_recalled:
            if pid in current_pref_ids:
                violations["proactive_retested"].append({"file": file_label, "pref_id": pid})

    total_violations = sum(len(v) for v in violations.values())

    for check_name, items in violations.items():
        status = "PASS" if not items else f"FAIL ({len(items)} violations)"
        print(f"  {check_name}: {status}")
        if items:
            for item in items[:5]:
                print(f"    {item}")
    print()

    results["state_violations"] = total_violations


# ---------- Phase 3: Judge checks ----------


def check_override_frequency(records: list[EvalRecord], results: dict) -> None:
    print("=" * 70)
    print("3.1 OVERRIDE FREQUENCY ANALYSIS")
    print("=" * 70)

    # Aggregated counters
    overall: Counter = Counter()
    per_agent: dict[str, Counter] = defaultdict(Counter)
    per_type: dict[str, Counter] = defaultdict(Counter)

    for rec in records:
        if rec.error:
            continue
        for v in rec.preference_verdicts:
            sim = v.get("simulator_verdict", "")
            final = v.get("final_verdict", "")
            ptype = v.get("type", "unknown")

            if sim == "recalled" and final == "recalled":
                key = "agree_recalled"
            elif sim == "missed" and final == "missed":
                key = "agree_missed"
            elif sim == "recalled" and final == "missed":
                key = "R->M"
            elif sim == "missed" and final == "recalled":
                key = "M->R"
            elif sim == "recalled" and final == "proactive":
                key = "agree_recalled"  # proactive is a form of recalled
            elif sim == "missed" and final == "proactive":
                key = "M->R"  # judge upgraded to proactive (recalled)
            else:
                key = f"other({sim}->{final})"

            overall[key] += 1
            per_agent[rec.agent][key] += 1
            per_type[ptype][key] += 1

    total = sum(overall.values())
    print(f"Total preference verdicts: {total}")
    print("Overall breakdown:")
    for key in ["agree_recalled", "agree_missed", "R->M", "M->R"]:
        count = overall.get(key, 0)
        pct = count / total * 100 if total else 0
        print(f"  {key:20s}: {count:6d} ({pct:.1f}%)")
    other_keys = [k for k in overall if k not in {"agree_recalled", "agree_missed", "R->M", "M->R"}]
    for key in other_keys:
        count = overall[key]
        pct = count / total * 100 if total else 0
        print(f"  {key:20s}: {count:6d} ({pct:.1f}%)")
    print()

    # Per-agent table
    print("Per-agent override rates:")
    print(f"  {'Agent':15s} {'Total':>6s} {'R->M':>10s} {'M->R':>10s} {'AgreeR':>10s} {'AgreeM':>10s}")
    for agent in sorted(per_agent):
        c = per_agent[agent]
        t = sum(c.values())
        rm = c.get("R->M", 0)
        mr = c.get("M->R", 0)
        ar = c.get("agree_recalled", 0)
        am = c.get("agree_missed", 0)
        print(
            f"  {agent:15s} {t:6d} {rm:5d}({rm/t*100:4.1f}%) {mr:5d}({mr/t*100:4.1f}%) "
            f"{ar:5d}({ar/t*100:4.1f}%) {am:5d}({am/t*100:4.1f}%)"
        )
    print()

    # Nocontext sanity check
    nc = per_agent.get("nocontext", Counter())
    nc_sim_recalled = nc.get("agree_recalled", 0) + nc.get("R->M", 0)
    nc_rm = nc.get("R->M", 0)
    nc_override_rate = nc_rm / nc_sim_recalled * 100 if nc_sim_recalled else 0
    print(f"NOCONTEXT SANITY CHECK:")
    print(f"  Simulator said 'recalled': {nc_sim_recalled}")
    print(f"  Judge overrode to 'missed': {nc_rm} ({nc_override_rate:.1f}%)")
    print(f"  Expected: >90% override rate (nocontext has no memory)")
    print()

    # Per preference type
    print("By preference type:")
    for ptype in sorted(per_type):
        c = per_type[ptype]
        t = sum(c.values())
        rm = c.get("R->M", 0)
        mr = c.get("M->R", 0)
        print(f"  {ptype:10s}: total={t}, R->M={rm} ({rm/t*100:.1f}%), M->R={mr} ({mr/t*100:.1f}%)")
    print()

    results["override_rm"] = overall.get("R->M", 0)
    results["override_mr"] = overall.get("M->R", 0)
    results["nocontext_override_rate"] = nc_override_rate


def check_stale_accuracy(records: list[EvalRecord], results: dict) -> None:
    print("=" * 70)
    print("3.2 STALE ACCURACY ANALYSIS")
    print("=" * 70)

    # Stale rate per agent (evolved prefs only)
    agent_evolved: dict[str, int] = defaultdict(int)
    agent_stale: dict[str, int] = defaultdict(int)
    stale_cases: dict[str, list[dict]] = defaultdict(list)
    stale_fp_count = 0
    stale_fp_cases: list[dict] = []

    for rec in records:
        if rec.error:
            continue
        for v in rec.preference_verdicts:
            if v.get("type") != "evolved":
                continue
            agent_evolved[rec.agent] += 1
            if v.get("stale_used"):
                agent_stale[rec.agent] += 1
                case = {
                    "file": f"{rec.session}/eval_{rec.task_num}_{rec.agent}_{rec.run}.json",
                    "pref_id": v.get("preference_id"),
                    "preference": v.get("preference", "")[:150],
                    "supersedes": (v.get("supersedes") or {}).get("fact", "")[:150],
                    "quote": v.get("quote", "")[:150],
                    "reasoning": v.get("reasoning", "")[:200],
                }
                stale_cases[rec.agent].append(case)

                # Check for false positive
                reasoning = v.get("reasoning", "")
                for fp_signal in STALE_FP_SIGNALS:
                    if fp_signal.search(reasoning):
                        stale_fp_count += 1
                        stale_fp_cases.append(case)
                        break

    print("Stale rate per agent (evolved preferences only):")
    for agent in sorted(set(agent_evolved) | set(agent_stale)):
        evolved = agent_evolved.get(agent, 0)
        stale = agent_stale.get(agent, 0)
        rate = stale / evolved * 100 if evolved else 0
        print(f"  {agent:15s}: {stale:4d}/{evolved:5d} ({rate:.1f}%)")
    print()

    print(f"Stale false positive candidates (reasoning contradicts stale_used=true): {stale_fp_count}")
    for case in stale_fp_cases[:5]:
        print(f"  {case['file']} — {case['pref_id']}")
        print(f"    Reasoning: {case['reasoning']}")
        print()

    # Sample stale cases per agent
    print("Stale case samples (3 per agent):")
    for agent in sorted(stale_cases):
        print(f"  --- {agent} ---")
        for case in stale_cases[agent][:3]:
            print(f"  {case['file']} — {case['pref_id']}")
            print(f"    Current: {case['preference']}")
            print(f"    Old:     {case['supersedes']}")
            print(f"    Quote:   {case['quote']}")
            print(f"    Reason:  {case['reasoning']}")
            print()

    # Nocontext stale anomaly
    nc_stale = agent_stale.get("nocontext", 0)
    nc_evolved = agent_evolved.get("nocontext", 0)
    print(f"NOCONTEXT STALE ANOMALY: {nc_stale}/{nc_evolved} (should be ~0)")
    print()

    results["stale_fp_count"] = stale_fp_count


def check_judge_quality(records: list[EvalRecord], results: dict) -> None:
    print("=" * 70)
    print("3.3 OVERALL JUDGE QUALITY")
    print("=" * 70)

    # Score distribution per agent
    agent_scores: dict[str, list[float]] = defaultdict(list)
    error_count = 0
    perfect_counts: Counter = Counter()

    for rec in records:
        if rec.error:
            error_count += 1
            continue
        agent_scores[rec.agent].append(rec.preference_score)
        if rec.preference_score == 1.0:
            perfect_counts[rec.agent] += 1

    print("Score distribution per agent:")
    print(f"  {'Agent':15s} {'N':>5s} {'Mean':>7s} {'Std':>7s} {'Min':>6s} {'Q25':>6s} {'Med':>6s} {'Q75':>6s} {'Max':>6s}")
    for agent in sorted(agent_scores):
        scores = sorted(agent_scores[agent])
        n = len(scores)
        q25 = scores[n // 4]
        q75 = scores[3 * n // 4]
        std_val = stdev(scores) if n > 1 else 0
        print(
            f"  {agent:15s} {n:5d} {mean(scores):7.4f} {std_val:7.4f} "
            f"{min(scores):6.3f} {q25:6.3f} {median(scores):6.3f} {q75:6.3f} {max(scores):6.3f}"
        )
    print()

    # Nocontext ceiling check
    nc_scores = agent_scores.get("nocontext", [])
    nc_high = [s for s in nc_scores if s > 0.3]
    print(f"NOCONTEXT CEILING CHECK: {len(nc_high)}/{len(nc_scores)} evals scored > 0.3")
    if nc_high:
        # Find and display the worst offenders
        nc_records = [r for r in records if r.agent == "nocontext" and not r.error and r.preference_score > 0.3]
        nc_records.sort(key=lambda r: r.preference_score, reverse=True)
        print(f"Top 5 nocontext false positives:")
        for r in nc_records[:5]:
            print(f"  score={r.preference_score:.3f}  {r.session}/eval_{r.task_num}_{r.agent}_{r.run}.json")
            for v in r.preference_verdicts:
                if v.get("final_verdict") == "recalled":
                    print(f"    {v['preference_id']}: sim={v.get('simulator_verdict')} -> judge=recalled")
                    print(f"      quote: {v.get('quote', '')[:150]}")
    print()

    # Perfect score analysis
    print("Perfect scores (preference_score=1.0) per agent:")
    for agent in sorted(agent_scores):
        n = len(agent_scores[agent])
        p = perfect_counts.get(agent, 0)
        print(f"  {agent:15s}: {p:4d}/{n:5d} ({p/n*100:.1f}%)")
    print()

    # Error file analysis
    error_records = [r for r in records if r.error]
    print(f"Error files: {len(error_records)}")
    for r in error_records:
        print(f"  {r.session}/eval_{r.task_num}_{r.agent}_{r.run}.json: {r.error[:150]}")
    print()

    results["error_count"] = error_count


# ---------- Phase 4: Summary ----------


def print_summary(records: list[EvalRecord], results: dict) -> None:
    total = len(records)
    n_sessions = len(set(r.session for r in records))

    print()
    print("=" * 70)
    print("QUALITY CHECK SUMMARY")
    print("=" * 70)
    print()
    print(f"DATA: {n_sessions} sessions, {total} eval files, {results.get('error_count', 0)} errors")
    print()
    print("USER SIMULATOR:")
    rr = results.get("role_reversal_flagged", 0)
    rr_total = results.get("role_reversal_total", 0)
    rr_rate = rr / rr_total * 100 if rr_total else 0
    print(f"  Role reversal flags: {rr}/{rr_total} user turns ({rr_rate:.1f}%)")
    print(f"  Early termination: {results.get('early_termination_count', 0)} evals with untested prefs")
    sv = results.get("state_violations", 0)
    print(f"  State management: {'PASS' if sv == 0 else f'FAIL ({sv} violations)'}")
    print()
    print("PREFERENCE JUDGE:")
    print(f"  Override rate: R->M={results.get('override_rm', 0)}, M->R={results.get('override_mr', 0)}")
    print(f"  Nocontext sanity: {results.get('nocontext_override_rate', 0):.1f}% of sim-recalled overridden (expect >90%)")
    print(f"  Stale false positives: {results.get('stale_fp_count', 0)}")


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(description="Post-evaluation quality and bug check")
    parser.add_argument("--eval-run", type=str, required=True, help="Eval run timestamp (e.g., 2026-03-16_215407)")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs directory (default: outputs)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_dir():
        print(f"Error: {outputs_dir} is not a directory")
        return

    records = load_all_evals(outputs_dir, args.eval_run)
    if not records:
        print("No eval files found.")
        return

    results: dict = {}

    # Phase 2: User simulator checks
    check_role_reversal(records, results)
    check_early_termination(records, results)
    check_state_management(records, results)

    # Phase 3: Judge checks
    check_override_frequency(records, results)
    check_stale_accuracy(records, results)
    check_judge_quality(records, results)

    # Phase 4: Summary
    print_summary(records, results)


if __name__ == "__main__":
    main()
