"""Diagnostic: Analyze unchanged baseline preference leakage into conversations.

For each dataset:
1. Identify unchanged baseline preferences (never in any session delta)
2. Use LLM to classify whether each leaked into conversation text
3. Cross-reference with foundry_local eval results to find critical cases:
   preferences that are truly absent from conversations but scored as "proactive"
"""

import asyncio
import json
import glob
import os
import sys
from typing import Any

from memory_gym.client import AsyncLLMPool, LLMClient, PooledLLMClient

SYSTEM_PROMPT = """You are analyzing conversation transcripts to determine if a specific user preference is discussed or mentioned anywhere.

Look for:
- Direct mentions of the preference
- Paraphrases or restatements
- Indirect references that convey the same concept
- Any content from which a reader could infer this preference

Be thorough but precise. Only mark as present if the preference concept genuinely appears — not if a vaguely related topic is discussed."""

USER_PROMPT_TEMPLATE = """PREFERENCE TO FIND:
"{fact}"

CONVERSATIONS ({num_sessions} sessions):
{conversations}

Does this preference concept appear in any of these conversations? Consider mentions by both the user and the assistant.

Return JSON:
{{
  "present": true or false,
  "evidence": [
    {{"session": <session number>, "speaker": "user" or "assistant", "quote": "<exact quote, max 100 chars>"}}
  ]
}}

If not present, return {{"present": false, "evidence": []}}"""


def load_dataset(session_path: str) -> dict[str, Any]:
    """Load a dataset and extract key information."""
    with open(session_path) as f:
        data = json.load(f)

    # Get all preference IDs that appear in any session delta
    delta_ids: set[str] = set()
    for sess in data["sessions"]:
        prefs = sess["preferences"]
        for p in prefs.get("created", []):
            delta_ids.add(p["id"])
        for p in prefs.get("evolved", []):
            delta_ids.add(p["from"]["id"])
            delta_ids.add(p["to"]["id"])
        for p in prefs.get("dropped", []):
            delta_ids.add(p["id"])

    active = data["final_state"]["active_preferences"]
    baseline_active = [p for p in active if p.get("created_at_session", 999) <= 0]
    unchanged_baselines = [p for p in baseline_active if p["id"] not in delta_ids]
    unchanged_ids = {p["id"] for p in unchanged_baselines}

    # Build conversation text per session
    session_conversations: list[str] = []
    for sess in data["sessions"]:
        conv = sess.get("conversation", [])
        lines = []
        for msg in conv:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        session_conversations.append("\n".join(lines))

    return {
        "data": data,
        "delta_ids": delta_ids,
        "unchanged_baselines": unchanged_baselines,
        "unchanged_ids": unchanged_ids,
        "session_conversations": session_conversations,
        "active": active,
    }


def load_foundry_local_evals(session_dir: str) -> list[dict[str, Any]]:
    """Load foundry_local eval results (run 01 only) from the most recent eval dir."""
    eval_dirs = sorted(glob.glob(os.path.join(session_dir, "evaluations", "*")))
    foundry_local_dir = None
    for ed in eval_dirs:
        config_path = os.path.join(ed, "run_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            if config.get("agent_type") == "foundry_local":
                if glob.glob(os.path.join(ed, "eval_*.json")):
                    foundry_local_dir = ed

    if not foundry_local_dir:
        return []

    results = []
    for ef in sorted(glob.glob(os.path.join(foundry_local_dir, "eval_*_01.json"))):
        with open(ef) as f:
            results.append(json.load(f))
    return results


def extract_eval_verdicts(
    eval_results: list[dict[str, Any]], unchanged_ids: set[str], delta_ids: set[str]
) -> list[dict[str, Any]]:
    """Extract preference verdicts from eval results, tagged by type.

    Classification uses delta membership (not the judge's type field, which
    mislabels newly-created preferences as "baseline" because they lack a
    supersedes field).

    Categories:
      - "in_delta": preference appeared in a session delta (created/evolved/dropped)
      - "unchanged_baseline": active baseline that was never in any delta
    """
    verdicts = []
    for ev in eval_results:
        trace = ev.get("preference_scoring", {}).get("first_mention_trace", [])
        for t in trace:
            pid = t.get("preference_id", "")
            usage = t.get("usage", "unknown")
            fact = t.get("preference", "")
            is_unchanged = pid in unchanged_ids

            # Use delta membership for classification, not judge's type field
            if is_unchanged:
                category = "unchanged_baseline"
            else:
                category = "in_delta"

            verdicts.append({
                "preference_id": pid,
                "category": category,
                "judge_type": t.get("type", "unknown"),
                "usage": usage,
                "fact": fact,
                "is_unchanged_baseline": is_unchanged,
                "in_delta": pid in delta_ids,
                "reasoning": t.get("reasoning", ""),
            })
    return verdicts


def classify_preference(
    client: LLMClient | PooledLLMClient,
    item: dict[str, Any],
) -> dict[str, Any]:
    """LLM call to classify whether a preference leaked into conversations."""
    pref_id = item["preference_id"]
    fact = item["fact"]
    conversations = item["conversations"]
    num_sessions = item["num_sessions"]

    prompt = USER_PROMPT_TEMPLATE.format(
        fact=fact,
        num_sessions=num_sessions,
        conversations=conversations,
    )

    try:
        result = client.complete_json(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=500)
        return {
            "preference_id": pref_id,
            "fact": fact,
            "present": result.get("present", False),
            "evidence": result.get("evidence", []),
        }
    except Exception as e:
        print(f"  LLM call failed for {pref_id}: {e}")
        return {
            "preference_id": pref_id,
            "fact": fact,
            "present": None,
            "evidence": [],
            "error": str(e),
        }


async def run_analysis() -> None:
    session_files = sorted(glob.glob("outputs/*/sessions.json"))
    if not session_files:
        print("No datasets found in outputs/")
        sys.exit(1)

    print(f"Found {len(session_files)} datasets\n")

    # Phase 1: Collect all data and build LLM call items
    all_datasets: list[dict[str, Any]] = []
    all_llm_items: list[dict[str, Any]] = []

    for sf in session_files:
        sdir = os.path.dirname(sf)
        ds_name = os.path.basename(sdir)

        ds = load_dataset(sf)
        eval_results = load_foundry_local_evals(sdir)

        if not eval_results:
            print(f"  {ds_name}: No foundry_local evals found, skipping")
            continue

        verdicts = extract_eval_verdicts(eval_results, ds["unchanged_ids"], ds["delta_ids"])

        # Build conversation text for LLM calls
        conv_text = ""
        for i, conv in enumerate(ds["session_conversations"]):
            conv_text += f"\n--- [Session {i}] ---\n{conv}\n"

        # Find unchanged baselines that were tested
        tested_unchanged = [v for v in verdicts if v["is_unchanged_baseline"]]

        # Build LLM items for each tested unchanged baseline
        for v in tested_unchanged:
            all_llm_items.append({
                "preference_id": v["preference_id"],
                "fact": v["fact"],
                "conversations": conv_text,
                "num_sessions": len(ds["session_conversations"]),
                "dataset": ds_name,
            })

        all_datasets.append({
            "name": ds_name,
            "persona_id": ds["data"].get("persona_id", "?"),
            "unchanged_baselines": ds["unchanged_baselines"],
            "unchanged_ids": ds["unchanged_ids"],
            "verdicts": verdicts,
            "tested_unchanged": tested_unchanged,
        })

    print(f"Total LLM classification calls needed: {len(all_llm_items)}")
    print("Running LLM-based leak classification...\n")

    # Phase 2: Run LLM calls in parallel
    pool = AsyncLLMPool()
    completed = [0]

    def on_result(index: int, item: Any, result: Any) -> None:
        completed[0] += 1
        status = "LEAKED" if result.get("present") else "ABSENT"
        print(f"  [{completed[0]}/{len(all_llm_items)}] {item['dataset']}/{item['preference_id']}: {status}")

    llm_results = await pool.run_parallel(
        items=all_llm_items,
        func=classify_preference,
        on_result=on_result,
    )

    # Index LLM results by (dataset, preference_id)
    leak_map: dict[tuple[str, str], dict[str, Any]] = {}
    for item, result in zip(all_llm_items, llm_results):
        leak_map[(item["dataset"], item["preference_id"])] = result

    # Phase 3: Report
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    # Aggregate counters
    agg: dict[str, dict[str, int]] = {
        "in_delta": {"tested": 0, "proactive": 0, "ignored": 0},
        "baseline_leaked": {"tested": 0, "proactive": 0, "ignored": 0},
        "baseline_absent": {"tested": 0, "proactive": 0, "ignored": 0},
    }
    critical_cases: list[dict[str, Any]] = []

    for ds in all_datasets:
        ds_name = ds["name"]
        persona = ds["persona_id"]
        n_unchanged = len(ds["unchanged_baselines"])
        n_tested = len(ds["tested_unchanged"])

        print(f"\n{'─' * 100}")
        print(f"Dataset: {ds_name} ({persona})")
        print(f"Unchanged baselines: {n_unchanged} total, {n_tested} tested in evals")
        print()

        if n_tested > 0:
            print(f"  {'Pref ID':<12} {'Fact':<50} {'In Convos?':<12} {'Verdict':<10} {'Critical?'}")
            print(f"  {'─' * 12} {'─' * 50} {'─' * 12} {'─' * 10} {'─' * 10}")

        for v in ds["tested_unchanged"]:
            pid = v["preference_id"]
            fact = v["fact"][:48] + "..." if len(v["fact"]) > 48 else v["fact"]
            usage = v["usage"]

            llm_result = leak_map.get((ds_name, pid), {})
            present = llm_result.get("present")
            if present is None:
                leak_status = "ERROR"
            elif present:
                leak_status = "LEAKED"
            else:
                leak_status = "ABSENT"

            is_critical = leak_status == "ABSENT" and usage == "proactive"
            critical_marker = "*** YES" if is_critical else ""

            print(f"  {pid:<12} {fact:<50} {leak_status:<12} {usage:<10} {critical_marker}")

            # Aggregate
            if leak_status == "LEAKED":
                agg["baseline_leaked"]["tested"] += 1
                agg["baseline_leaked"]["proactive" if usage == "proactive" else "ignored"] += 1
            elif leak_status == "ABSENT":
                agg["baseline_absent"]["tested"] += 1
                agg["baseline_absent"]["proactive" if usage == "proactive" else "ignored"] += 1

            if is_critical:
                critical_cases.append({
                    "dataset": ds_name,
                    "preference_id": pid,
                    "fact": v["fact"],
                    "reasoning": v["reasoning"],
                    "evidence": llm_result.get("evidence", []),
                })

        # Count in-delta verdicts (all preferences that were in session deltas)
        for v in ds["verdicts"]:
            if v["category"] == "in_delta":
                agg["in_delta"]["tested"] += 1
                agg["in_delta"]["proactive" if v["usage"] == "proactive" else "ignored"] += 1

    # Aggregate summary
    print(f"\n{'=' * 100}")
    print("AGGREGATE SUMMARY (across all datasets, foundry_local run 01)")
    print(f"{'=' * 100}")
    print()
    print(f"  {'Category':<35} {'Tested':>8} {'Proactive':>10} {'Ignored':>9} {'Proactive Rate':>15}")
    print(f"  {'─' * 35} {'─' * 8} {'─' * 10} {'─' * 9} {'─' * 15}")

    for label, key in [
        ("In session deltas (discussed)", "in_delta"),
        ("Unchanged baseline — leaked", "baseline_leaked"),
        ("Unchanged baseline — absent", "baseline_absent"),
    ]:
        d = agg[key]
        total = d["tested"]
        pro = d["proactive"]
        ign = d["ignored"]
        rate = f"{pro / total * 100:.0f}%" if total > 0 else "N/A"
        print(f"  {label:<35} {total:>8} {pro:>10} {ign:>9} {rate:>15}")

    # Critical cases detail
    if critical_cases:
        print(f"\n{'=' * 100}")
        print(f"CRITICAL CASES: {len(critical_cases)} truly absent preferences scored as PROACTIVE")
        print(f"{'=' * 100}")
        for c in critical_cases:
            print(f"\n  Dataset: {c['dataset']}")
            print(f"  Pref:    {c['preference_id']}: {c['fact']}")
            print(f"  Judge:   {c['reasoning']}")
    else:
        print(f"\nNo critical cases found — all truly absent baselines were correctly marked as ignored.")

    # Save full results
    output = {
        "aggregate": agg,
        "critical_cases": critical_cases,
        "datasets": [
            {
                "name": ds["name"],
                "persona_id": ds["persona_id"],
                "unchanged_count": len(ds["unchanged_baselines"]),
                "tested_count": len(ds["tested_unchanged"]),
                "preferences": [
                    {
                        "preference_id": v["preference_id"],
                        "fact": v["fact"],
                        "usage": v["usage"],
                        "leak_status": (
                            "LEAKED" if leak_map.get((ds["name"], v["preference_id"]), {}).get("present")
                            else "ABSENT" if leak_map.get((ds["name"], v["preference_id"]), {}).get("present") is not None
                            else "ERROR"
                        ),
                        "evidence": leak_map.get((ds["name"], v["preference_id"]), {}).get("evidence", []),
                        "judge_reasoning": v["reasoning"],
                    }
                    for v in ds["tested_unchanged"]
                ],
            }
            for ds in all_datasets
        ],
    }
    with open("debug_score_inflation_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to debug_score_inflation_results.json")


if __name__ == "__main__":
    asyncio.run(run_analysis())
