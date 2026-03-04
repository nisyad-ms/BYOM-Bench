#!/usr/bin/env python
"""Minimal smoke test for Hindsight store (populate + retrieve only).

Does NOT run the full evaluation pipeline (which requires DefaultAzureCredential).
Designed for Docker environments where only AZURE_TOKEN env var is available.
"""

import json
import sys
import time
from pathlib import Path

DATA_DIR = Path("data/debug_data")
SESSIONS_FILE = DATA_DIR / "sessions.json"


def main() -> None:
    if not SESSIONS_FILE.exists():
        print(f"Missing {SESSIONS_FILE}. Run from repo root.")
        sys.exit(1)

    from memory_gym.agents import HindsightMemoryStore
    from memory_gym.schemas import MultiSessionOutput

    with open(SESSIONS_FILE, encoding="utf-8") as f:
        data = MultiSessionOutput.from_dict(json.load(f))

    t0 = time.time()
    print("Creating HindsightMemoryStore...")
    store = HindsightMemoryStore(profile="debug_smoke")
    t_init = time.time()
    print(f"  Server startup: {t_init - t0:.1f}s")

    try:
        print(f"Populating {len(data.sessions)} sessions...")
        store.populate(data)
        t_populate = time.time()
        print(f"  Populate: {t_populate - t_init:.1f}s ({(t_populate - t_init) / len(data.sessions):.1f}s/session)")

        print("Retrieving...")
        results = store.retrieve("What are the user preferences?")
        t_retrieve = time.time()
        print(f"  Retrieve: {t_retrieve - t_populate:.1f}s")
        print(f"  Got {len(results)} memories")
        for i, r in enumerate(results):
            print(f"  [{i}] {r[:120]}...")

        if not results:
            print("FAILED: No memories retrieved")
            sys.exit(1)

        t_total = time.time() - t0
        print(f"\nPASSED: Hindsight store smoke test")
        print(f"Total: {t_total:.1f}s (startup={t_init - t0:.1f}s, populate={t_populate - t_init:.1f}s, retrieve={t_retrieve - t_populate:.1f}s)")
    finally:
        print("Cleaning up...")
        store.cleanup()


if __name__ == "__main__":
    main()
