#!/usr/bin/env python
"""Test event generation — persona + life events + preference updates (no conversations).

Usage:
    python test_event_generation.py                                          # 1 random persona, 3 events
    python test_event_generation.py --num-events 5                           # 1 random persona, 5 events
    python test_event_generation.py --persona test --num-events 3            # All test personas, 3 events each
    python test_event_generation.py --persona test --num 2 --num-events 4    # First 2 test personas, 4 events
    python test_event_generation.py --persona "software development"         # All from that domain
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from test_data_generation import resolve_personas

from memory_gym.data_generators import MultiSessionGenerator
from memory_gym.schemas import PreferenceTimeline

OUTPUTS_DIR = Path("outputs")


def _timeline_to_dict(timeline: PreferenceTimeline) -> dict:
    """Serialize preference timeline to a readable dict."""
    active = []
    evolved = []
    for p in timeline.preferences.values():
        entry = {
            "id": p.preference_id,
            "fact": p.fact,
            "domain": p.domain,
            "created_session": p.created_at_session,
        }
        if p.is_active:
            active.append(entry)
        else:
            entry["superseded_by"] = p.superseded_by
            entry["superseded_at_session"] = p.superseded_at_session
            entry["reason"] = p.reason_for_change
            evolved.append(entry)
    return {"active": active, "evolved_from": evolved}


def run_one(persona: str, num_events: int, max_retries: int = 3) -> tuple[str, Path]:
    """Generate persona + life events + preference updates. Returns (persona, output_path)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _run_one_attempt(persona, num_events)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                print(f"  [{persona[:50]}] Attempt {attempt} failed: {e}. Retrying...", flush=True)
    raise last_exc  # type: ignore[misc]


def _run_one_attempt(persona: str, num_events: int) -> tuple[str, Path]:
    """Single attempt at generating persona + life events + preference updates."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    session_dir = OUTPUTS_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    output_path = session_dir / "event_generation.json"

    generator = MultiSessionGenerator(persona=persona, num_sessions=num_events)

    # Step 1: Expand persona
    print(f"  [{persona[:50]}] Expanding persona...", flush=True)
    expanded = generator._expand_persona()

    # Step 2: Generate life events (sequential — each event sees previous ones)
    print(f"  [{persona[:50]}] Generating {num_events} life events...", flush=True)
    life_events = generator._generate_life_events(expanded)

    # Step 3: Run event loop — update preferences after each event
    timeline = PreferenceTimeline()
    generator._load_baseline_preferences(timeline, expanded)

    sessions = []
    for idx, event in enumerate(life_events):
        print(f"  [{persona[:50]}] Event {idx + 1}/{num_events}: updating preferences ({event.domain})...", flush=True)
        evolved_mapping, new_pref_ids, dropped_pref_ids = generator._update_preferences(
            event, timeline, idx, expanded
        )

        sessions.append({
            "session_id": idx,
            "life_event": {"date": event.date, "event": event.event, "domain": event.domain},
            "evolutions": [
                {
                    "from_id": old_id,
                    "from": timeline.preferences[old_id].fact if old_id in timeline.preferences else old_id,
                    "to_id": new_id,
                    "to": timeline.preferences[new_id].fact,
                    "reason": timeline.preferences[old_id].reason_for_change if old_id in timeline.preferences else "",
                }
                for old_id, new_id in evolved_mapping.items()
            ],
            "dropped": [
                {
                    "id": pid,
                    "fact": timeline.preferences[pid].fact,
                    "domain": timeline.preferences[pid].domain,
                    "reason": timeline.preferences[pid].reason_for_change or "",
                }
                for pid in dropped_pref_ids
            ],
            "new_preferences": [
                {"id": pid, "fact": timeline.preferences[pid].fact, "domain": timeline.preferences[pid].domain}
                for pid in new_pref_ids
            ],
            "active_preference_count": len(timeline.get_active_preferences()),
        })

    output = {
        "persona": persona,
        "expanded_persona": expanded.to_dict(),
        "sessions": sessions,
        "preference_timeline": _timeline_to_dict(timeline),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return persona, output_path


def main():
    parser = argparse.ArgumentParser(description="Test event generation (persona + events + preference updates)")
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="'test', 'all', or a domain name. Default: 1 random persona.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Use first N personas from the resolved list.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=3,
        help="Number of life events per persona (default: 3).",
    )
    args = parser.parse_args()

    personas = resolve_personas(args.persona)

    if args.num is not None:
        if args.persona is None:
            parser.error("--num requires --persona (test, all, or a domain name)")
        personas = personas[: args.num]

    print(f"Generating {args.num_events} events for {len(personas)} persona(s)\n")

    with ThreadPoolExecutor(max_workers=len(personas)) as executor:
        futures = {executor.submit(run_one, p, args.num_events): p for p in personas}
        failed = []
        for future in as_completed(futures):
            persona = futures[future]
            try:
                _, output_path = future.result()
                print(f"\n  {persona}\n  -> {output_path}\n")
            except Exception as e:
                failed.append((persona, e))
                print(f"\n  FAIL: {persona}\n  -> {e}\n")

    if failed:
        print(f"\n{len(failed)} persona(s) failed:")
        for persona, err in failed:
            print(f"  {persona}: {err}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
